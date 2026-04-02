"""Multi-user isolation and GDPR forgetting."""
import os
from engram import AgentMemory

DB_PATH = "./multi_user.db"
mem = AgentMemory(db_path=DB_PATH)

# Different users' memories are fully isolated
mem.remember("喜欢详细解释", user_id="alice", type="preference")
mem.remember("喜欢一句话总结", user_id="bob", type="preference")

# Each user only recalls their own memories
alice_results = mem.recall("回答风格偏好", user_id="alice")
bob_results = mem.recall("回答风格偏好", user_id="bob")

print("Alice:", alice_results[0].memory.content if alice_results else "无")
print("Bob:", bob_results[0].memory.content if bob_results else "无")

# GDPR: forget a user completely
deleted = mem.forget(user_id="bob", hard=True)
print(f"\n已删除 Bob 的 {deleted} 条记忆")

# Verify Bob's memories are gone
bob_after = mem.recall("回答风格偏好", user_id="bob")
print(f"Bob 剩余记忆: {len(bob_after)}")

mem.close()
for suffix in ("", "-wal", "-shm"):
    path = DB_PATH + suffix
    if os.path.exists(path):
        os.remove(path)
