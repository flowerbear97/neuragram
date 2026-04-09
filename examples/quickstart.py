"""Engram quickstart — 30 seconds to get started."""
import os
from neuragram import AgentMemory

DB_PATH = "./quickstart.db"

# 1. Initialize (zero config)
mem = AgentMemory(db_path=DB_PATH)

# 2. Remember things
mem.remember(
    "用户是一名 Python 后端开发者",
    user_id="user_001",
    type="fact",
)
mem.remember(
    "用户偏好简洁的代码注释风格",
    user_id="user_001",
    type="preference",
    importance=0.8,
)
mem.remember(
    "2024-03-15 用户的 Redis 连接超时问题，最终通过切换到 IP 直连解决",
    user_id="user_001",
    type="episode",
    tags=["redis", "troubleshooting"],
)

# 3. Recall
results = mem.recall("用户的技术背景是什么？", user_id="user_001")
for r in results:
    print(f"[{r.score:.2f}] ({r.memory.memory_type.value}) {r.memory.content}")

# 4. Stats
print("\n--- Stats ---")
print(mem.stats())

# 5. Cleanup
mem.close()
for suffix in ("", "-wal", "-shm"):
    path = DB_PATH + suffix
    if os.path.exists(path):
        os.remove(path)
