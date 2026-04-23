# core/tools/implementations/user_profile.py

from sqlalchemy import select
from infrastructure.db.connection import AsyncSessionLocal
from infrastructure.db.models import User, Order
import uuid

async def get_user_profile(user_id: str) -> dict:
    """Fetch user profile and order count from database"""

    async with AsyncSessionLocal() as session:
        try:
            uid = uuid.UUID(user_id)
        except ValueError:
            return {
                "ok": False,
                "error": "Invalid user ID format",
                "data": None
            }

        result = await session.execute(select(User).where(User.id == uid))
        user = result.scalar_one_or_none()

        if not user:
            return {
                "ok": False,
                "error": "User not found",
                "data": None
            }

        # Count their total orders
        orders_result = await session.execute(select(Order).where(Order.customer_id == uid))
        orders = orders_result.scalars().all()

        return {
            "ok": True,
            "error": None,
            "data": {
                "user_id": str(user.id),
                "name": user.name,
                "email": user.email,
                "tier": user.tier,
                "preferred_channel": user.preferred_channel,
                "total_orders": len(orders),
                "is_vip": user.tier == "vip"
            }
        }