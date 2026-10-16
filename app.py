"""FastAPI 服务入口:把 MeetingRAG 暴露为 HTTP API。

启动方式(二选一):
    uvicorn app:app --host 0.0.0.0 --port 8000
    python app.py

鉴权:若设置了环境变量 APP_API_KEY,则所有写接口需带请求头 X-API-Key。
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag import MeetingRAG


# ---------- 日志 ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")


# ---------- 请求/响应模型 ----------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str


class DeleteRequest(BaseModel):
    filename: str = Field(..., min_length=1)


class BuildResponse(BaseModel):
    added: int


# ---------- 鉴权(可选)----------
APP_API_KEY = os.getenv("APP_API_KEY", "")


def require_api_key(request: Request, x_api_key: str = Header(default="")) -> None:
    """设置了 APP_API_KEY 时,校验请求头 X-API-Key。"""
    if APP_API_KEY and x_api_key != APP_API_KEY:
        logger.warning("鉴权失败:非法 API key,来源 IP=%s", _client_ip(request))
        raise HTTPException(status_code=401, detail="invalid api key")


def get_rag(request: Request) -> MeetingRAG:
    return request.app.state.rag


# ---------- 限流(令牌桶)----------
# /chat 配额:突发 20 个令牌,每秒补充 0.5 个(约 30 次/分钟)
CHAT_CAPACITY = 20
CHAT_REFILL = 0.5
# /build、/delete 运维接口配额:突发 3 个,每 60 秒补充 1 个
OPS_CAPACITY = 3
OPS_REFILL = 1.0 / 60.0


class TokenBucket:
    """令牌桶:以固定速率补充令牌,允许突发,超限拒绝。"""

    def __init__(self, capacity: int, refill_per_second: float):
        self.capacity = float(capacity)
        self.refill_per_second = float(refill_per_second)
        self.tokens = self.capacity
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last_refill) * self.refill_per_second)
        self.last_refill = now

    def try_acquire(self) -> bool:
        with self._lock:
            self._refill()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

    def retry_after(self) -> int:
        """估算还需多少秒才能拿到一个令牌(用于 Retry-After)。"""
        with self._lock:
            self._refill()
            if self.tokens >= 1.0:
                return 0
            return int((1.0 - self.tokens) / self.refill_per_second) + 1


_buckets: dict = {}
_buckets_lock = threading.Lock()


def _get_bucket(key: str, capacity: int, refill_per_second: float) -> TokenBucket:
    with _buckets_lock:
        bucket = _buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(capacity, refill_per_second)
            _buckets[key] = bucket
        return bucket


def _client_ip(request: Request) -> str:
    """取客户端 IP;反代场景优先用 X-Forwarded-For 首个地址。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(name: str, capacity: int, refill_per_second: float):
    """返回一个 FastAPI 依赖:按「接口 + 客户端 IP」做令牌桶限流。"""

    def limiter(request: Request) -> None:
        ip = _client_ip(request)
        key = f"{name}:{ip}"
        bucket = _get_bucket(key, capacity, refill_per_second)
        if not bucket.try_acquire():
            logger.warning("限流触发:接口=%s IP=%s", name, ip)
            raise HTTPException(
                status_code=429,
                detail="请求过于频繁,请稍后再试",
                headers={"Retry-After": str(bucket.retry_after())},
            )

    return limiter


# ---------- 应用生命周期 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时创建一次,进程内单实例,避免每次请求重复开库
    logger.info("初始化 MeetingRAG ...")
    try:
        app.state.rag = MeetingRAG()
    except Exception:
        logger.exception("MeetingRAG 初始化失败(请检查 DASHSCOPE_API_KEY 等配置)")
        raise
    logger.info("MeetingRAG 初始化完成")
    yield
    logger.info("服务关闭")


app = FastAPI(title="会议纪要 RAG 问答服务", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """兜住所有未捕获异常:记录堆栈 + 返回脱敏错误。"""
    logger.exception("未捕获异常: %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "服务内部错误,请稍后再试"})


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key), Depends(rate_limit("chat", CHAT_CAPACITY, CHAT_REFILL))])
def chat(req: ChatRequest, rag: MeetingRAG = Depends(get_rag)) -> ChatResponse:
    """提问。路由用同步 def,FastAPI 会自动丢线程池,避免阻塞事件循环。"""
    answer = rag.chat(req.question, req.session_id)
    return ChatResponse(answer=answer)


@app.post("/build", response_model=BuildResponse, dependencies=[Depends(require_api_key), Depends(rate_limit("build", OPS_CAPACITY, OPS_REFILL))])
def build(rag: MeetingRAG = Depends(get_rag)) -> BuildResponse:
    """增量入库:扫描模板目录,导入新增 docx。"""
    added = rag.build_knowledge_base()
    return BuildResponse(added=added)


@app.post("/delete", dependencies=[Depends(require_api_key), Depends(rate_limit("delete", OPS_CAPACITY, OPS_REFILL))])
def delete(req: DeleteRequest, rag: MeetingRAG = Depends(get_rag)):
    """删除某文件的向量分片,并同步移除入库记录。"""
    rag.delete_by_filename(req.filename)
    return {"status": "ok", "filename": req.filename}


@app.get("/health")
def health():
    """健康检查:确认 RAG 已初始化且向量库可访问(不触发上游计费调用)。"""
    rag = getattr(app.state, "rag", None)
    if rag is None:
        logger.warning("健康检查失败:RAG 未初始化")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "detail": "RAG 未初始化"})
    try:
        # 轻量探活:读一次向量库,确认 Chroma 可访问(不调用 embedding、不触发计费)
        rag.db._collection.count()
    except Exception as e:
        logger.warning("健康检查失败:向量库不可访问 %s", e)
        return JSONResponse(status_code=503, content={"status": "unhealthy", "detail": "向量库不可访问"})
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
