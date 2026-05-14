from fastapi import FastAPI
import redis
import os

app = FastAPI()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

try:
    r = redis.from_url(REDIS_URL)
    r.ping()
    print(f"[INFO] Connected to Redis at {REDIS_URL}")
except (redis.RedisError, redis.ConnectionError) as e:
    print(f"[WARN] Redis not available: {e}")
    r = None

@app.get("/health")
def health():
    redis_status = "healthy"
    if r is not None:
        try:
            r.ping()
        except Exception:
            redis_status = "unhealthy"
    else:
        redis_status = "unavailable"
    return {"status": "ok" if redis_status == "healthy" else "degraded", "redis": redis_status}