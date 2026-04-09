"""Memory lifecycle: TTL, versioning, and decay."""
import os
from datetime import datetime, timedelta, timezone
from neuragram import AgentMemory

DB_PATH = "./lifecycle.db"
mem = AgentMemory(db_path=DB_PATH)

# 1. Memory with TTL (expires in 1 hour)
mem.remember(
    "用户当前正在调试支付接口",
    user_id="u1",
    type="plan_state",
    expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
)

# 2. Versioned updates
mid = mem.remember("用户的邮箱是 old@example.com", user_id="u1", type="fact")
print(f"Created memory: {mid}")

mem.update(mid, content="用户的邮箱是 new@example.com")
updated = mem.get(mid)
print(f"Updated content: {updated.content}")
print(f"Current version: {updated.version}")

# View version history
versions = mem.history(mid)
print(f"\nVersion history ({len(versions)} previous versions):")
for v in versions:
    print(f"  v{v.version}: {v.content}")

# 3. Run decay
result = mem.decay(max_age_days=7)
print(f"\nDecay result: expired={result['expired']}, archived={result['archived']}")

# 4. Stats
print(f"\nStore stats: {mem.stats()}")

mem.close()
for suffix in ("", "-wal", "-shm"):
    path = DB_PATH + suffix
    if os.path.exists(path):
        os.remove(path)
