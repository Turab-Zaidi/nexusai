# core/tools/implementations/order_lookup.py

from sqlalchemy import select
from infrastructure.db.connection import AsyncSessionLocal
from infrastructure.db.models import Order
from langfuse.decorators import observe

@observe(as_type="span", name="lookup_order")
async def lookup_order(order_id: str) -> dict:
    """
    Look up order in PostgreSQL database.
    Returns order details or error.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Order).where(Order.id == order_id)
        )
        order = result.scalar_one_or_none()

        if not order:
            return {
                "ok": False,
                "error": f"Order {order_id} not found",
                "data": None
            }

        return {
            "ok": True,
            "error": None,
            "data": {
                "order_id": order.id,
                "product_name": order.product_name,
                "product_category": order.product_category,
                "amount": order.amount,
                "status": order.status,
                "ordered_at": str(order.ordered_at),
                "delivered_at": str(order.delivered_at) if order.delivered_at else None,
                "tracking_number": order.tracking_number,
                "refund_eligible": order.refund_eligible,
                "refund_window_days": order.refund_window_days
            }
        }