import asyncio
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker
from infrastructure.db.connection import AsyncSessionLocal
from infrastructure.db.models import User, Card, Transaction, SupportTicket

fake = Faker()

# Predefined realistic merchants
MERCHANTS = [
    ("Netflix", "subscription", 15.99),
    ("Spotify", "subscription", 10.99),
    ("Amazon", "retail", None), 
    ("Uber", "transport", None),
    ("Starbucks", "dining", None),
    ("Target", "retail", None),
    ("Nexus Bank Overdraft Fee", "fee", 35.00),
    ("Zelle Transfer", "transfer", None)
]

async def seed_data():
    async with AsyncSessionLocal() as session:
        print("Starting database seed...")
        
        # Generate 50 Users
        for _ in range(50):
            tier = random.choices(["standard", "premium", "private_wealth"], weights=[70, 25, 5])[0]
            user = User(
                id=str(uuid.uuid4()),
                name=fake.name(),
                email=fake.unique.email(),
                phone=fake.phone_number(),
                tier=tier,
                fraud_risk_score=random.randint(1, 40) if tier == 'standard' else random.randint(1, 15),
                account_balance=round(random.uniform(1500.0, 25000.0), 2),
                created_at=datetime.now() - timedelta(days=random.randint(90, 365))
            )
            session.add(user)
            await session.flush() 
            
            # 1-2 Cards per user
            cards = []
            for _ in range(random.randint(1, 2)):
                card = Card(
                    id=f"CARD-{fake.unique.random_int(min=1000, max=9999)}",
                    user_id=user.id,
                    last_4_digits=fake.credit_card_number()[-4:],
                    card_type=random.choice(["physical", "virtual"]),
                    status=random.choices(["active", "frozen"], weights=[90, 10])[0]
                )
                session.add(card)
                cards.append(card)
            
            await session.flush()

            # 150-250 Transactions per card (spanning 90 days)
            for card in cards:
                for _ in range(random.randint(150, 250)):
                    merchant, category, amt = random.choice(MERCHANTS)
                    amount = amt if amt else round(random.uniform(5.0, 150.0), 2)
                    
                    txn = Transaction(
                        id=f"TXN-{fake.unique.random_int(min=10000, max=999999)}",
                        user_id=user.id,
                        card_id=card.id,
                        merchant_name=merchant,
                        amount=amount,
                        category=category,
                        status=random.choices(["cleared", "pending", "disputed"], weights=[80, 15, 5])[0],
                        timestamp=datetime.now() - timedelta(days=random.randint(1, 90))
                    )
                    session.add(txn)
            
            # 2-3 Support Tickets per user
            for _ in range(random.randint(2, 3)):
                intent = random.choice(["fee_waiver", "dispute", "general_inquiry", "lost_card"])
                ticket = SupportTicket(
                    id=str(uuid.uuid4()),
                    user_id=user.id,
                    intent=intent,
                    summary={
                        "problem": f"User had an issue regarding {intent}.",
                        "solved_how": "Agent provided policy info or executed an action.",
                        "solution": "Resolved successfully."
                    },
                    status=random.choices(["resolved", "escalated"], weights=[90, 10])[0],
                    created_at=datetime.now() - timedelta(days=random.randint(1, 60))
                )
                session.add(ticket)
        
        # Commit all data to the database
        await session.commit()
        print("Successfully seeded 50 users, with their cards, transactions, and support tickets!")

if __name__ == "__main__":
    asyncio.run(seed_data())
