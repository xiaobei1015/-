# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

一个基于 RAG（检索增强生成）的中文建设工程会议纪要问答系统。它将 `.docx` 会议纪要文件导入 Chroma 向量库，然后通过「检索相关片段 → 重排 → 生成」回答用户问题，模型均来自阿里云百炼（DashScope）。

整个应用都在单个 Jupyter notebook `main.ipynb` 中（从上到下顺序执行）。没有 Python 包、没有 CLI 入口、没有测试套件，也没有配置 linter。

## 常用命令

- **安装 / 同步依赖**（使用 `uv`，仓库内有 `uv.lock` 与 `pyproject.toml`）：
  ```
  uv sync
  ```
- **运行应用**：在 Jupyter 中打开 `main.ipynb` 并运行所有单元格（例如 `jupyter notebook main.ipynb`，或通过 IDE 运行）。最后一个单元格（`if __name__ == "__main__":`）会依次调用 `build_rag_knowledge_base()` 和 `rag_chat(...)`。
- **运行前设置必需的 API Key**（向量化、重排、对话都要用到）：
  ```powershell
  $env:DASHSCOPE_API_KEY = "<你的key>"
  ```

本项目没有定义 build、lint、test 相关命令。

## 架构

整个流水线是线性的，全部写在 notebook 单元格里。核心流程：**导入 docx → 切分 → 向量化 → 存入 Chroma → 检索 → 重排 → 生成**。

- **向量化** — `Qwen3VLEmbeddings` 封装了 `dashscope.MultiModalEmbedding.call`，将其适配为 LangChain 的 `Embeddings` 接口（`embed_documents` / `embed_query`）。LangChain 没有原生支持千问向量模型，所以必须靠这个适配器才能对接 Chroma。模型：`qwen3-vl-embedding`，维度 1024，`enable_fusion=True`。
- **入库** — `build_rag_knowledge_base()` 扫描 `TEMPLATE_FOLDER`，用 `extract_by_python_docx()` 提取每个新增 `.docx` 的文本（python-docx，保结构解析：标题段落标记为 `## `，普通段落原文输出，表格按 `[列头] 值 | ...` 拼接并递归展开嵌套表格、去重合并单元格），用 `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n", ","])` 切分，再把分片以元数据 `{"source": <文件名>}` 写入 Chroma。
- **检索 + 生成** — `rag_chat(question)` 依次执行：MMR 检索（k=10）→ 千问重排（`qwen3-rerank`，top_n=5）→ 拼接带引用标注的上下文字符串 → 发送 `SystemMessage` + `HumanMessage` 给对话模型。重排接口调用失败时降级使用初次检索的前 5 条文档。
- **对话模型** — `qwen3.8-omni-flash`，通过 `init_chat_model` 走百炼的 OpenAI 兼容接口（`base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"`，`model_provider = "openai"`）。

## 关键文件与数据产物

- `main.ipynb` — 全部应用代码（import、自定义向量类、全局常量、入库/检索函数、入口）。
- `pyproject.toml` / `uv.lock` — 依赖清单（`dashscope`、`docx2python`、`python-docx`、`langchain`、`langchain-chroma`、`langchain-openai`、`langchain-text-splitters`、`ipykernel`）。Python >= 3.13。
- `meeting_project/meeting_templates/` — 待入库的 `.docx` 会议纪要源文件。
- `chroma_rag_db/` — 持久化的 Chroma 向量库（已提交到仓库）。
- `imported_files.json` — 记录已入库的文件名，用于后续运行时跳过重复文件。

## 约定与注意事项

- **所有路径都是相对路径**（`./meeting_project/...`、`./chroma_rag_db`、`./imported_files.json`），所以进程的工作目录必须是项目根目录。
- **增量入库只按文件名判断。** `build_rag_knowledge_base()` 会跳过 `imported_files.json` 中已存在的 `.docx`。修改已入库的 docx **不会**重新入库 —— 需要手动从 Chroma 删除旧分片（`delete_old_chunks_by_filename()` 可做这件事，按 `source` 元数据删除），并把文件名从 `imported_files.json` 中移除。
- `chat_history` 是内存中的全局列表（仅当前进程有效，不持久化）。
- 文档来源存在元数据键 `source` 下（是纯文件名，不是完整路径）。
- 领域内容（会议纪要与生成的回答）为简体中文；系统提示词和 docstring 也是中文写的。
