# apps/api/routes/health.py

from fastapi import APIRouter
import redis.asyncio as redis
import os

router = APIRouter()

@router.get("/health")
async def health_check():
    checks = {}

    try:
        r = redis.from_url(os.getenv("REDIS_URL"))
        await r.ping()
        checks["redis"] = "healthy"
        await r.aclose()
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"

    return {
        "status": "healthy",
        "service": "NexusAI",
        "version": "1.0.0",
        "checks": checks
    }