# infrastructure/cache/redis_cache.py

import os
import json
import hashlib
from typing import Optional, Any
from langfuse.decorators import observe


class RedisCache:
    """
    Redis-backed cache layer for NexusAI.
    Caches knowledge agent responses to avoid redundant LLM calls.
    Action responses are NEVER cached (order data changes).
    """

    def __init__(self):
        self.enabled = False
        self._client = None
        self.default_ttl = 3600  # 1 hour

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            import redis
            self._client = redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            self.enabled = True
            print("[RedisCache] Connected successfully")
        except ImportError:
            print("[RedisCache] redis package not installed. Caching disabled.")
        except Exception as e:
            print(f"[RedisCache] Connection failed: {e}. Caching disabled.")

    def _make_key(self, prefix: str, intent: str, query_hash: str) -> str:
        """Generate a deterministic cache key."""
        return f"nexusai:{prefix}:{intent}:{query_hash}"

    def _hash_query(self, query: str) -> str:
        """Create a short hash of the query text."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]

    @observe(as_type="span", name="cache_get")
    async def get(self, intent: str, query: str) -> Optional[str]:
        """
        Try to retrieve a cached response.
        Returns None on miss.
        """
        if not self.enabled:
            return None

        try:
            key = self._make_key("knowledge", intent, self._hash_query(query))
            cached = self._client.get(key)
            if cached:
                print(f"[RedisCache] HIT for intent={intent}")
                # Track hit count
                self._client.incr("nexusai:stats:cache_hits")
                return cached
            else:
                self._client.incr("nexusai:stats:cache_misses")
                return None
        except Exception as e:
            print(f"[RedisCache] Get error: {e}")
            return None

    @observe(as_type="span", name="cache_set")
    async def set(self, intent: str, query: str, response: str, ttl: int = None) -> bool:
        """
        Store a response in cache.
        """
        if not self.enabled:
            return False

        try:
            key = self._make_key("knowledge", intent, self._hash_query(query))
            self._client.setex(key, ttl or self.default_ttl, response)
            return True
        except Exception as e:
            print(f"[RedisCache] Set error: {e}")
            return False

    async def get_stats(self) -> dict:
        """Get cache hit/miss statistics."""
        if not self.enabled:
            return {"enabled": False}

        try:
            hits = int(self._client.get("nexusai:stats:cache_hits") or 0)
            misses = int(self._client.get("nexusai:stats:cache_misses") or 0)
            total = hits + misses
            return {
                "enabled": True,
                "hits": hits,
                "misses": misses,
                "total_requests": total,
                "hit_rate": round(hits / total, 3) if total > 0 else 0.0
            }
        except Exception:
            return {"enabled": True, "error": "stats unavailable"}

    async def flush(self) -> bool:
        """Clear all NexusAI cache entries."""
        if not self.enabled:
            return False
        try:
            keys = self._client.keys("nexusai:*")
            if keys:
                self._client.delete(*keys)
            return True
        except Exception:
            return False


# Singleton
redis_cache = RedisCache()
