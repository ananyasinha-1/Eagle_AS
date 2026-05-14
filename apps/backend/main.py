from fastapi import FastAPI
import redis
import os

app = FastAPI()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

try:
    r = redis.from_url(REDIS_URL)
    r.ping()
    print(f"[INFO] Connected to Redis at {REDIS_URL}")
except Exception as e:
    print(f"[WARN] Redis not available: {e}")
    r = None

@app.get("/health")
def health():
    return {"status": "ok"}