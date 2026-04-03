## Engram

[English](README.md) | **中文**

轻量级、框架无关的 AI Agent 记忆层。基于 SQLite + sqlite-vec + FTS5 构建，无需外部服务。

```bash
pip install neuragram
```

```python
from engram import AgentMemory

mem = AgentMemory(db_path="./memory.db")
mem.remember("用户喜欢简洁的代码解释", user_id="u1", type="preference")
results = mem.recall("用户偏好什么风格？", user_id="u1")
print(results[0].memory.content)
mem.close()
```

## 对比

| | Engram | Mem0 | Letta | Graphiti |
|---|---|---|---|---|
| 安装方式 | `pip install neuragram` | `pip install` | Docker + Server | pip + Neo4j |
| 外部依赖 | 无 | 向量数据库 + LLM | PG + Server + LLM | 图数据库 + LLM |
| 框架锁定 | 无 | 无 | Letta 运行时 | 无 |
| 记忆生命周期 | 内置 | 无 | Agent 自管理 | 部分 |
| 需要 LLM | 否 | 是 | 是 | 是 |

## 功能特性

### 存储与检索
- **混合搜索** — 向量相似度 + FTS5 关键词 + 时效性评分，通过 Reciprocal Rank Fusion 融合
- **可解释排序** — `explain()` 返回每条结果的完整评分分解（向量排名、关键词排名、RRF 贡献、时效性因子）
- **LRU 缓存** — 热点记忆从内存缓存直接返回
- **性能调优** — WAL 模式、64 MB 页缓存、mmap、批量操作单事务提交

### 记忆智能
- **智能记忆** — 自动分类记忆类型、重要性和置信度（基于规则或 LLM 增强）
- **对话提取** — 从聊天消息中提取结构化记忆（需要 LLM）
- **冲突检测与解决** — 检测矛盾记忆并自动解决
- **合并** — 将相似记忆合并为摘要，减少噪音

### 生命周期管理
- **TTL 过期** — 设置了 `expires_at` 的记忆自动过期
- **不活跃归档** — 超过 N 天未访问的记忆自动归档
- **GDPR 遗忘** — `forget(user_id="u1")` 删除用户所有数据
- **后台 Worker** — `create_worker()` 按计划执行维护任务
- **版本历史** — 每次更新自动版本化；通过 `history()` 查看完整历史

### 多租户与访问控制
- **隔离** — 所有操作支持 `namespace` + `user_id` + `agent_id` 作用域
- **基于角色的访问** — `AccessLevel.READ / WRITE / ADMIN`，支持命名空间和用户级别作用域
- **命名空间统计** — 通过 `namespace_stats()` 查看按命名空间的记忆分布

### 集成
- **Claude Code** — 支持通过插件市场或手动 MCP 配置接入 Claude Code
- **MCP Server** — `engram-mcp` CLI 将 Engram 暴露为 MCP 工具服务器（Claude Desktop、Cursor 等）
- **REST API** — `engram-api` CLI 启动 FastAPI HTTP 服务，提供 12 个端点
- **LangChain** — `EngramMemory` 实现 `BaseMemory`（save_context / load_memory_variables）
- **LlamaIndex** — `EngramChatMemory` 提供 put / get / get_all / reset

### 可观测性
- **OpenTelemetry** — 所有操作自动生成 traces、metrics 和 spans；未安装 OTel 时零开销

## 使用

### 嵌入配置

```python
# OpenAI
mem = AgentMemory(db_path="./memory.db", embedding="openai", embedding_model="text-embedding-3-small")

# 本地 (sentence-transformers)
mem = AgentMemory(db_path="./memory.db", embedding="local")

# 无嵌入（仅关键词检索）
mem = AgentMemory(db_path="./memory.db")
```

### 智能记忆

```python
from engram import AgentMemory, CallableLLMProvider

async def my_llm(prompt):
    return await call_my_llm(prompt)

mem = AgentMemory(db_path="./memory.db", llm=CallableLLMProvider(my_llm))
ids = mem.smart_remember("用户更喜欢 Python 而不是 JavaScript")
```

### Claude Code

```bash
# 方式一：从 Claude Code 插件市场安装
claude plugin marketplace add flowerbear97/neuragram
claude plugin install engram

# 方式二：手动 MCP 配置
pip install neuragram[mcp]
claude mcp add engram -- engram-mcp --db-path ./memory.db

# 方式三：启用 OpenAI 嵌入，获得混合搜索能力
claude mcp add engram -- engram-mcp --db-path ./memory.db --embedding openai
```

安装后，Claude Code 自动获得跨会话的持久化记忆能力，包含 6 个工具：
`engram_remember`、`engram_recall`、`engram_smart_remember`、`engram_forget`、`engram_list`、`engram_stats`。

### MCP Server

```bash
engram-mcp --db-path ./memory.db
engram-mcp --db-path ./memory.db --embedding openai
```

### REST API

```bash
engram-api --db-path ./memory.db --port 8080
```

### LangChain

```python
from engram.integrations.langchain import EngramMemory

memory = EngramMemory(db_path="./memory.db", user_id="u1")
memory.save_context({"input": "我喜欢简洁的回答"}, {"output": "好的！"})
result = memory.load_memory_variables({"input": "回答风格"})
```

### LlamaIndex

```python
from engram.integrations.llamaindex import EngramChatMemory

memory = EngramChatMemory(db_path="./memory.db", user_id="u1")
memory.put("用户是 Python 开发者", memory_type="fact")
results = memory.get("编程语言")
```

### 可解释检索

```python
for exp in mem.explain("用户偏好", user_id="u1"):
    print(exp["summary"])
    # "vector rank #1 (+0.0082) | keyword rank #3 (+0.0048) | recency 0.95 (age 0.5d)"
```

### 访问控制

```python
from engram import AccessPolicy, AccessLevel

policy = AccessPolicy(enabled=True, default_level=AccessLevel.NONE)
policy.grant("reader_agent", AccessLevel.READ)
policy.grant("writer_agent", AccessLevel.WRITE, namespace="project_a")
policy.grant("admin_bot", AccessLevel.ADMIN)

mem = AgentMemory(db_path="./memory.db", access_policy=policy)
```

### 后台 Worker

```python
worker = mem.create_worker(interval_seconds=3600)
await worker.start()
# ...
await worker.stop()
```

## 安装

```bash
pip install neuragram              # 核心
pip install neuragram[openai]      # + OpenAI 嵌入
pip install neuragram[local]       # + sentence-transformers
pip install neuragram[mcp]         # + MCP 服务器
pip install neuragram[api]         # + REST API (FastAPI)
pip install neuragram[langchain]   # + LangChain 适配器
pip install neuragram[llamaindex]  # + LlamaIndex 适配器
pip install neuragram[telemetry]   # + OpenTelemetry
pip install neuragram[all]         # 全部
```

## 架构

```
AgentMemory (client.py)
├── 存储层              SQLite + sqlite-vec + FTS5
├── 检索引擎            RRF 融合、时效性加权、去重
├── 处理管线            提取 → 分类 → 冲突检测 → 合并
├── 生命周期            衰减、遗忘、后台 Worker
├── 访问控制            基于角色，命名空间作用域
├── 可观测性            OpenTelemetry traces + metrics
└── 集成                MCP Server, REST API, LangChain, LlamaIndex
```

## 许可证

Apache-2.0
