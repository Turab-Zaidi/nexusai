# core/tools/implementations/fintech_tools.py
"""
FinTech Action Tools — backed by SQLite via SQLAlchemy.
Each tool returns a standardised dict: {"ok": bool, "data": ..., "error": str|None}
"""

from sqlalchemy import select, update, func
from datetime import datetime, timedelta
from infrastructure.db.connection import AsyncSessionLocal
from infrastructure.db.models import User, Card, Transaction


async def get_user_profile(user_id: str) -> dict:
    """Fetch user profile including tier, fraud score, and card count."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            return {"ok": False, "data": None, "error": f"User {user_id} not found."}

        cards = await session.execute(
            select(Card).where(Card.user_id == user_id)
        )
        card_list = cards.scalars().all()

        return {
            "ok": True,
            "data": {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "tier": user.tier,
                "fraud_risk_score": user.fraud_risk_score,
                "cards": [
                    {
                        "id": c.id,
                        "last_4": c.last_4_digits,
                        "type": c.card_type,
                        "status": c.status,
                        "daily_limit": c.daily_limit
                    }
                    for c in card_list
                ]
            },
            "error": None
        }


async def get_recent_transactions(user_id: str, limit: int = 5) -> dict:
    """Fetch the most recent transactions for a user."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Transaction)
            .where(Transaction.user_id == user_id)
            .order_by(Transaction.timestamp.desc())
            .limit(limit)
        )
        txns = result.scalars().all()

        if not txns:
            return {"ok": True, "data": [], "error": None}

        return {
            "ok": True,
            "data": [
                {
                    "id": t.id,
                    "merchant": t.merchant_name,
                    "amount": t.amount,
                    "status": t.status,
                    "category": t.category,
                    "timestamp": t.timestamp.isoformat() if t.timestamp else None
                }
                for t in txns
            ],
            "error": None
        }


async def search_transactions(user_id: str, merchant_name: str = None, amount: float = None, limit: int = 10) -> dict:
    """Search for specific transactions for a user."""
    async with AsyncSessionLocal() as session:
        query = select(Transaction).where(Transaction.user_id == user_id)
        
        if merchant_name:
            query = query.where(Transaction.merchant_name.ilike(f"%{merchant_name}%"))
        if amount:
            # allow a small margin for float comparison
            query = query.where(Transaction.amount >= amount - 0.01, Transaction.amount <= amount + 0.01)
            
        query = query.order_by(Transaction.timestamp.desc()).limit(limit)
        result = await session.execute(query)
        txns = result.scalars().all()

        return {
            "ok": True,
            "data": [
                {
                    "id": t.id,
                    "merchant": t.merchant_name,
                    "amount": t.amount,
                    "status": t.status,
                    "category": t.category,
                    "timestamp": t.timestamp.isoformat() if t.timestamp else None
                }
                for t in txns
            ],
            "error": None
        }


async def freeze_card(card_id: str, user_id: str) -> dict:
    """Freeze a card. Validates ownership before executing."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Card).where(Card.id == card_id, Card.user_id == user_id)
        )
        card = result.scalar_one_or_none()

        if not card:
            return {"ok": False, "data": None, "error": f"Card {card_id} not found or does not belong to this user."}

        if card.status == "frozen":
            return {"ok": False, "data": None, "error": f"Card ending in {card.last_4_digits} is already frozen."}

        if card.status == "reported_stolen":
            return {"ok": False, "data": None, "error": "This card has already been reported stolen and cannot be modified."}

        await session.execute(
            update(Card).where(Card.id == card_id).values(status="frozen")
        )
        await session.commit()

        return {
            "ok": True,
            "data": {
                "card_id": card_id,
                "last_4": card.last_4_digits,
                "card_type": card.card_type,
                "new_status": "frozen"
            },
            "error": None
        }


async def unfreeze_card(card_id: str, user_id: str) -> dict:
    """Unfreeze a card. Validates ownership before executing."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Card).where(Card.id == card_id, Card.user_id == user_id)
        )
        card = result.scalar_one_or_none()

        if not card:
            return {"ok": False, "data": None, "error": f"Card {card_id} not found or does not belong to this user."}

        if card.status == "active":
            return {"ok": False, "data": None, "error": f"Card ending in {card.last_4_digits} is already active."}

        if card.status == "reported_stolen":
            return {"ok": False, "data": None, "error": "This card was reported stolen and cannot be unfrozen."}

        await session.execute(
            update(Card).where(Card.id == card_id).values(status="active")
        )
        await session.commit()

        return {
            "ok": True,
            "data": {
                "card_id": card_id,
                "last_4": card.last_4_digits,
                "card_type": card.card_type,
                "new_status": "active"
            },
            "error": None
        }


async def report_stolen_card(card_id: str, user_id: str) -> dict:
    """Permanently block a card by reporting it stolen."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Card).where(Card.id == card_id, Card.user_id == user_id)
        )
        card = result.scalar_one_or_none()

        if not card:
            return {"ok": False, "data": None, "error": f"Card {card_id} not found or does not belong to this user."}

        if card.status == "reported_stolen":
            return {"ok": False, "data": None, "error": "This card has already been reported stolen."}

        await session.execute(
            update(Card).where(Card.id == card_id).values(status="reported_stolen")
        )
        await session.commit()

        return {
            "ok": True,
            "data": {
                "card_id": card_id,
                "last_4": card.last_4_digits,
                "card_type": card.card_type,
                "new_status": "reported_stolen"
            },
            "error": None
        }


async def submit_dispute(transaction_id: str, user_id: str, reason: str) -> dict:
    """Submit a dispute on a transaction. Validates it belongs to the user."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Transaction).where(
                Transaction.id == transaction_id,
                Transaction.user_id == user_id
            )
        )
        txn = result.scalar_one_or_none()

        if not txn:
            return {"ok": False, "data": None, "error": f"Transaction {transaction_id} not found for this user."}

        if txn.status == "disputed":
            return {"ok": False, "data": None, "error": f"Transaction {transaction_id} is already under dispute."}

        if txn.status == "refunded":
            return {"ok": False, "data": None, "error": "This transaction has already been refunded."}

        if txn.category == "transfer":
            return {"ok": False, "data": None, "error": "Peer-to-peer transfers (like Zelle, Venmo, or wire transfers) cannot be disputed through the automated system due to immediate settlement. Please contact fraud support directly."}

        await session.execute(
            update(Transaction)
            .where(Transaction.id == transaction_id)
            .values(status="disputed")
        )
        await session.commit()

        return {
            "ok": True,
            "data": {
                "transaction_id": transaction_id,
                "merchant": txn.merchant_name,
                "amount": txn.amount,
                "new_status": "disputed",
                "reason": reason,
                "reference": f"DISP-{transaction_id[-6:]}"
            },
            "error": None
        }


async def waive_fee(transaction_id: str, user_id: str) -> dict:
    """
    Waive a fee transaction. Only valid for category='fee'.
    Returns the result so the Quality Judge can check against tier-based waiver limits.
    """
    async with AsyncSessionLocal() as session:
        # Fetch user tier first
        user_result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = user_result.scalar_one_or_none()

        if not user:
            return {"ok": False, "data": None, "error": "User not found."}

        # Fetch the transaction
        txn_result = await session.execute(
            select(Transaction).where(
                Transaction.id == transaction_id,
                Transaction.user_id == user_id
            )
        )
        txn = txn_result.scalar_one_or_none()

        if not txn:
            return {"ok": False, "data": None, "error": f"Transaction {transaction_id} not found for this user."}

        if txn.category != "fee":
            return {"ok": False, "data": None, "error": f"Transaction {transaction_id} is not a fee and cannot be waived."}

        if txn.status == "refunded":
            return {"ok": False, "data": None, "error": "This fee has already been waived/refunded."}

        await session.execute(
            update(Transaction)
            .where(Transaction.id == transaction_id)
            .values(status="refunded")
        )
        await session.commit()

        return {
            "ok": True,
            "data": {
                "transaction_id": transaction_id,
                "fee_amount": txn.amount,
                "merchant": txn.merchant_name,
                "new_status": "refunded",
                "user_tier": user.tier,
                "reference": f"WAIV-{transaction_id[-6:]}"
            },
            "error": None
        }


async def analyze_spending(user_id: str, days: int = 30, category: str = None) -> dict:
    """Analyze spending for a user over a time period, optionally filtered by category."""
    async with AsyncSessionLocal() as session:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = select(
            func.sum(Transaction.amount).label("total_spent"),
            func.count(Transaction.id).label("transaction_count")
        ).where(
            Transaction.user_id == user_id,
            Transaction.timestamp >= cutoff_date,
            Transaction.status != "refunded"
        )
        
        if category:
            clean_category = category.strip(", ")
            query = query.where(Transaction.category == clean_category)
            
        result = await session.execute(query)
        row = result.fetchone()
        
        total_spent = float(row.total_spent) if row and row.total_spent else 0.0
        txn_count = int(row.transaction_count) if row and row.transaction_count else 0
        
        # Also get current balance
        user_result = await session.execute(select(User.account_balance).where(User.id == user_id))
        account_balance = user_result.scalar_one_or_none() or 0.0

        return {
            "ok": True,
            "data": {
                "time_period_days": days,
                "category_filter": category or "all",
                "total_spent": total_spent,
                "transaction_count": txn_count,
                "current_account_balance": account_balance
            },
            "error": None
        }
