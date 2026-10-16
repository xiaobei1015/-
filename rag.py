"""会议纪要 RAG 核心模块。

从 main.ipynb 抽离出来的业务逻辑:千问向量化适配、docx 解析、
增量入库、检索 + 重排 + 生成。供 FastAPI 服务与 notebook 共用。
"""

import functools
import json
import logging
import os
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import List, Optional

logger = logging.getLogger(__name__)

import dashscope
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
  

class TransientError(Exception):
    """暂时性错误(网络抖动、限流、超时),适合重试。"""


def _run_with_timeout(fn, timeout, *args, **kwargs):
    """用线程池强制超时;超时抛 TransientError。"""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            raise TransientError(f"调用超时({timeout}s)") from None


def retry_transient(retries: int = 3, base_delay: float = 1.0, max_delay: float = 8.0, timeout: Optional[float] = None):
    """装饰器:对 TransientError 做指数退避重试;非 TransientError 直接抛。

    timeout 给出时,每次调用用线程池强制超时,超时视为 TransientError。
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(retries + 1):
                try:
                    if timeout is not None:
                        return _run_with_timeout(fn, timeout, *args, **kwargs)
                    return fn(*args, **kwargs)
                except TransientError as e:
                    last_exc = e
                    if attempt < retries:
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.3), max_delay)
                        logger.warning("%s 第 %d 次重试(原因:%s),%.1fs 后重试", fn.__name__, attempt + 1, e, delay)
                        time.sleep(delay)
            logger.error("%s 重试 %d 次后仍失败", fn.__name__, retries, exc_info=last_exc)
            raise last_exc
        return wrapper
    return decorator


class Qwen3VLEmbeddings(Embeddings):
    """自定义封装类,阿里云千问向量模型不被 langchain 支持,需要自定义类适配 Chroma。"""

    def __init__(self, api_key: str, model_name: str = "qwen3-vl-embedding", dimension: int = 1024, batch_size: int = 20):
        self.api_key = api_key
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size

    @retry_transient(retries=3, timeout=30)
    def _embed_one(self, text: str) -> List[float]:
        try:
            resp = dashscope.MultiModalEmbedding.call(
                api_key=self.api_key,
                model=self.model_name,
                input=[{"text": text}],
                enable_fusion=True,
                parameters={"dimension": self.dimension},
            )
        except Exception as e:
            raise TransientError(f"向量化网络异常:{e}") from e
        if resp.status_code == HTTPStatus.OK:
            return resp.output["embeddings"][0]["embedding"]
        if resp.status_code in (429, 500, 502, 503, 504):
            raise TransientError(f"向量化限流/服务错误 HTTP {resp.status_code}: {getattr(resp, 'message', '')}")
        raise Exception(f"文档向量化失败:{resp.message}")

    @retry_transient(retries=3, timeout=120)
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """一次请求向量化一批文本(利用 input 列表),减少请求次数。"""
        try:
            resp = dashscope.MultiModalEmbedding.call(
                api_key=self.api_key,
                model=self.model_name,
                input=[{"text": t} for t in texts],
                enable_fusion=True,
                parameters={"dimension": self.dimension},
            )
        except Exception as e:
            raise TransientError(f"向量化网络异常:{e}") from e
        if resp.status_code == HTTPStatus.OK:
            return [item["embedding"] for item in resp.output["embeddings"]]
        if resp.status_code in (429, 500, 502, 503, 504):
            raise TransientError(f"向量化限流/服务错误 HTTP {resp.status_code}: {getattr(resp, 'message', '')}")
        raise Exception(f"文档向量化失败:{resp.message}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = []
        for i in range(0, len(texts), self.batch_size):
            result.extend(self._embed_batch(texts[i:i + self.batch_size]))
        return result

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


def extract_by_python_docx(file_path: str) -> str:
    """用 python-docx 解析 docx,保留文档结构。

    输出规则:
      - 标题段落 -> "## 标题文本"
      - 普通段落 -> 原文
      - 表格数据行 -> "[列头] 值 | [列头] 值 ..."
      - 嵌套表格 -> 缩进后递归展开
    """
    doc = Document(file_path)
    lines = []
    _render_body(doc, lines)
    return "\n".join(lines)


def _render_body(doc, lines):
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            _render_paragraph(Paragraph(child, doc), lines)
        elif child.tag == qn("w:tbl"):
            _render_table(Table(child, doc), lines, indent=0)


def _render_paragraph(paragraph, lines):
    text = paragraph.text.strip()
    if not text:
        return
    style_name = ""
    if paragraph.style is not None and paragraph.style.name:
        style_name = paragraph.style.name
    if "heading" in style_name.lower() or "标题" in style_name:
        lines.append(f"## {text}")
    else:
        lines.append(text)


def _render_table(table, lines, indent=0):
    prefix = "  " * indent
    rows = table.rows
    if not rows:
        return
    headers = [cell.text.replace("\n", " ").strip() for cell in rows[0].cells]
    data_rows = rows[1:] if len(rows) > 1 else rows
    seen = set()
    for row in data_rows:
        row_parts = []
        for col_idx, cell in enumerate(row.cells):
            key = id(cell._tc)
            if key in seen:
                continue
            seen.add(key)
            header = _column_header(headers, col_idx)
            cell_text = cell.text.replace("\n", " ").strip()
            if cell.tables:
                if cell_text:
                    lines.append(f"{prefix}[{header}] {cell_text}")
                for nested in cell.tables:
                    _render_table(nested, lines, indent + 1)
            elif cell_text:
                row_parts.append(f"[{header}] {cell_text}")
        if row_parts:
            lines.append(prefix + " | ".join(row_parts))


def _column_header(headers, col_idx):
    if col_idx < len(headers) and headers[col_idx]:
        return headers[col_idx]
    return f"列{col_idx + 1}"


def delete_old_chunks_by_filename(db: Chroma, filename: str):
    """根据文件名删除向量库中该文件的所有切片。"""
    result = db.get(where={"source": filename})
    target_ids = result["ids"]
    if len(target_ids) > 0:
        db.delete(ids=target_ids)


class MeetingRAG:
    """会议纪要 RAG 封装类:入库 + 检索 + 重排 + 生成全流程。

    进程内单实例持有 db/retriever/llm,避免每次请求重复开库;
    会话历史按 session_id 隔离;入库/删除用进程内锁串行化。
    """

    def __init__(
        self,
        template_folder: str = "./meeting_project/meeting_templates",
        chroma_path: str = "./chroma_rag_db",
        record_file: str = "./imported_files.json",
        history_db_path: str = "./chat_history.db",
        history_max_messages: int = 40,
        history_ttl_days: int = 30,
        api_key: Optional[str] = None,
        embed_model: str = "qwen3-vl-embedding",
        embed_dim: int = 1024,
        embed_batch_size: int = 20,
        rerank_model: str = "qwen3-rerank",
        llm_model: str = "qwen3.8-omni-flash",
        llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature: float = 0.5,
        search_k: int = 10,
        rerank_top_n: int = 5,
        rerank_threshold: float = 0.4,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[list] = None,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未设置 DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key

        self.template_folder = template_folder
        self.chroma_path = chroma_path
        self.record_file = record_file

        # 向量化 + 切分(separators 补中文标点)
        self.embedding = Qwen3VLEmbeddings(self.api_key, embed_model, embed_dim, embed_batch_size)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n", ",", "。", "；"],
        )

        # 向量库 + 检索器:进程内只建一次
        self.db = Chroma(embedding_function=self.embedding, persist_directory=self.chroma_path)
        self.retriever = self.db.as_retriever(search_type="mmr", search_kwargs={"k": search_k})

        # 对话模型
        self.llm = init_chat_model(
            model=llm_model,
            base_url=llm_base_url,
            model_provider="openai",
            api_key=self.api_key,
            temperature=temperature,
        )
        self.rerank_model = rerank_model
        self.rerank_top_n = rerank_top_n
        self.rerank_threshold = rerank_threshold

        # 会话历史:SQLite 持久化,按 session_id 隔离;两层清理:数量截断 + 死会话 TTL
        self.history_db_path = history_db_path
        self.history_max_messages = history_max_messages
        self.history_ttl_days = history_ttl_days
        self._init_history_db()

        # 入库/删除都是「JSON 记录 + Chroma」的复合操作,加进程内锁串行化
        self._lock = threading.Lock()

    # ---------- 入库 ----------
    def _load_imported(self):
        if os.path.exists(self.record_file):
            with open(self.record_file, "r", encoding="utf-8") as f:
                return set(json.load(f))
        with open(self.record_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        return set()

    def _save_imported(self, imported):
        with open(self.record_file, "w", encoding="utf-8") as f:
            json.dump(list(imported), f, ensure_ascii=False, indent=2)

    # ---------- 会话历史(SQLite)----------
    def _init_history_db(self) -> None:
        """建表 + WAL 模式。WAL 支持并发读、串行写,贴合本场景。"""
        conn = sqlite3.connect(self.history_db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)"
            )
            conn.commit()
        finally:
            conn.close()

    def _load_history(self, session_id: str, limit: int = 20):
        """读取最近 limit 条消息,返回按时间正序的 [(role, content), ...]。"""
        conn = sqlite3.connect(self.history_db_path)
        try:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return rows[::-1]  # 反转为正序
        finally:
            conn.close()

    def _append_message(self, session_id: str, role: str, content: str) -> None:
        conn = sqlite3.connect(self.history_db_path)
        try:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            # 第一层:该 session 只保留最近 history_max_messages 条
            conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND id NOT IN ("
                "SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?)",
                (session_id, session_id, self.history_max_messages),
            )
            # 第二层:惰性清理死 session(超过 history_ttl_days 天无活动)
            conn.execute(
                "DELETE FROM messages WHERE session_id IN ("
                "SELECT session_id FROM messages GROUP BY session_id "
                "HAVING MAX(created_at) < datetime('now', 'localtime', ?))",
                (f"-{self.history_ttl_days} days",),
            )
            conn.commit()
        finally:
            conn.close()

    def build_knowledge_base(self) -> int:
        with self._lock:
            imported = self._load_imported()
            texts, metas = [], []
            for fname in os.listdir(self.template_folder):
                if not fname.endswith(".docx") or fname in imported:
                    continue
                doc_txt = extract_by_python_docx(os.path.join(self.template_folder, fname))
                texts.append(doc_txt)
                metas.append({"source": fname})
            if not texts:
                logger.info("没有找到新增会议纪要文档,知识库无需更新")
                return 0

            chunks = self.text_splitter.create_documents(texts, metas)
            # 先写向量库,记录新增分片 id 以便失败回滚
            added_ids = self.db.add_documents(chunks)
            try:
                for m in metas:
                    imported.add(m["source"])
                self._save_imported(imported)
            except Exception:
                # 记录写失败 -> 回滚刚写入的分片,回到「未入库」的一致状态
                try:
                    self.db.delete(ids=added_ids)
                except Exception:
                    logger.error("回滚分片失败,请手动清理重复分片", exc_info=True)
                raise
            logger.info("知识库增量导入完成,本次新增 %d 份会议纪要", len(texts))
            return len(texts)

    def delete_by_filename(self, filename: str) -> None:
        """按文件名删除向量库分片,并同步从记录文件移除(同一把锁内串行化)。

        顺序:先删分片(幂等)再删记录。若删记录失败,分片已删但记录还在,
        重试本方法即可收敛(删分片是幂等 no-op),不会造成静默数据缺失。
        """
        with self._lock:
            delete_old_chunks_by_filename(self.db, filename)
            imported = self._load_imported()
            imported.discard(filename)
            self._save_imported(imported)
            logger.info("已删除 %s 的分片,并从入库记录移除", filename)

    # ---------- 问答 ----------
    @retry_transient(retries=2, timeout=30)
    def _rerank(self, query: str, documents: List[str]):
        try:
            resp = dashscope.TextReRank.call(
                model=self.rerank_model,
                query=query,
                documents=documents,
                top_n=self.rerank_top_n,
                return_documents=True,
            )
        except Exception as e:
            raise TransientError(f"重排网络异常:{e}") from e
        if resp.status_code == HTTPStatus.OK:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504):
            raise TransientError(f"重排限流/服务错误 HTTP {resp.status_code}: {getattr(resp, 'message', '')}")
        return resp  # 其他非 OK,返回由 chat 降级处理

    @retry_transient(retries=2, timeout=120)
    def _generate(self, sys_prompt: str, user_prompt: str) -> str:
        try:
            return self.llm.invoke(
                [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
            ).content
        except Exception as e:
            raise TransientError(f"LLM 调用失败:{e}") from e

    def chat(self, question: str, session_id: str = "default") -> str:
        logger.info("收到提问 session=%s question=%.100s", session_id, question)
        raw_docs = self.retriever.invoke(question)
        doc_text_list = [d.page_content for d in raw_docs]
        logger.info("初次检索命中 %d 条", len(raw_docs))

        # 重排(带重试);失败降级使用初次检索结果
        try:
            resp = self._rerank(question, doc_text_list)
            rerank_ok = resp.status_code == HTTPStatus.OK
        except TransientError:
            rerank_ok = False
            resp = None

        if rerank_ok:
            results = resp.output["results"]
            # 过滤掉相关性低于阈值的文档
            filtered = [r for r in results if r.get("relevance_score", 0) >= self.rerank_threshold]
            top_docs = [raw_docs[r["index"]] for r in filtered]
        else:
            logger.warning("重排失败,降级使用初次检索结果")
            top_docs = raw_docs[: self.rerank_top_n]
        logger.info("重排后保留 %d 条,来源=%s", len(top_docs), [d.metadata["source"] for d in top_docs])

        # 过滤后无相关内容:直接返回固定话术,不调 LLM、不标来源
        if not top_docs:
            answer = "知识库未查询到相关信息"
            logger.info("无相关文档,直接返回固定话术 session=%s", session_id)
            self._append_message(session_id, "user", question)
            self._append_message(session_id, "ai", answer)
            return answer

        context = "".join(
            f"\n===参考文档{i + 1},来源:{d.metadata['source']}===\n{d.page_content}"
            for i, d in enumerate(top_docs)
        )
        # 从 SQLite 读取最近 20 条(约 10 轮)历史,避免无限膨胀
        history_rows = self._load_history(session_id, limit=20)
        history_text = "".join(
            f"{'用户' if role == 'user' else '助手'}:{content}\n"
            for role, content in history_rows
        )

        sys_prompt = """
        你是一个专业问答助手,只能根据给到的参考文档回答用户
        规则:
        1.如果参考文档中没有相关信息直接回复:知识库未查询到相关信息
        2.回答结尾标注信息来源于哪一份纪要文档
        3.禁止编造文档不存在内容,可以参考历史对话理解上下文
    """
        user_prompt = f"""
    历史对话:{history_text},
    会议纪要参考资料:{context},
    用户现在的问题:{question}
"""
        answer = self._generate(sys_prompt, user_prompt)
        logger.info("回答完成 session=%s 长度=%d", session_id, len(answer))
        # 本轮问答落库持久化
        self._append_message(session_id, "user", question)
        self._append_message(session_id, "ai", answer)
        return answer

    # ---------- 生命周期 ----------
    def clear_history(self, session_id: Optional[str] = None) -> None:
        conn = sqlite3.connect(self.history_db_path)
        try:
            if session_id is None:
                conn.execute("DELETE FROM messages")
            else:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Chroma 无显式 close,这里预留资源释放/状态收尾
        return False
