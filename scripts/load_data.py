
import asyncio
import json
import random
import uuid
from datetime import datetime, timedelta

import pandas as pd
from datasets import load_dataset
from faker import Faker
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from infrastructure.db.connection import (
    AsyncSessionLocal,
    create_tables
)
from infrastructure.db.models import (
    User, Order, KnowledgeEntry
)

fake = Faker()

from sqlalchemy import delete

async def clear_data(session: AsyncSession):
    print("\nClearing existing data (Users, Orders, Knowledge)...")
    
    # Use 'delete' for bulk deletion
    await session.execute(delete(Order))
    await session.execute(delete(User))
    await session.execute(delete(KnowledgeEntry))
    
    await session.commit()
    print("Data cleared successfully.")



async def load_knowledge_base(session: AsyncSession):
    print("Downloading Bitext dataset from HuggingFace...")

    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train"
    )

    df = dataset.to_pandas()

    print(df.head(1))

    print(f"Downloaded {len(df)} rows")
    print(f"Intents found: {df['intent'].nunique()}")
    print(f"Categories found: {df['category'].nunique()}")

    # Save raw data for reference
    df.to_csv("data/raw/bitext_raw.csv", index=False)
    print("Raw data saved to data/raw/bitext_raw.csv")

    df['response_length'] = df['response'].str.len()
    knowledge_df = (
        df.sort_values('response_length', ascending=False)
        .groupby('intent')
        .first()
        .reset_index()
    )

    print(f"\nLoading {len(knowledge_df)} knowledge entries...")
    print("Generating embeddings with sentence-transformers...")
    print("(This downloads a ~90MB model locally, no API calls needed)")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    entries_created = 0

    for _, row in knowledge_df.iterrows():
        embedding = model.encode(
            row['instruction']
        ).tolist()

        entry = KnowledgeEntry(
            id=uuid.uuid4(),
            question=row['instruction'],
            answer=row['response'],
            category=row['category'],
            intent=row['intent'],
            embedding=embedding
        )

        session.add(entry)
        entries_created += 1

    await session.commit()
    print(f"Knowledge base loaded: {entries_created} entries")

    df.to_csv("data/processed/bitext_processed.csv", index=False)
    test_df = df.sample(frac=0.2, random_state=42)
    test_df.to_csv("data/processed/test_set.csv", index=False)
    print(f"Test set saved: {len(test_df)} rows")


# ─────────────────────────────────────────────
# STEP 2: Generate fake users
# ─────────────────────────────────────────────

async def generate_users(
    session: AsyncSession,
    count: int = 500
) -> list:
    print(f"\nGenerating {count} fake users...")

    users = []
    tier_weights = {'standard': 60, 'silver': 25, 'gold': 12, 'vip': 3}
    tiers = list(tier_weights.keys())
    weights = list(tier_weights.values())

    for i in range(count):
        user = User(
            id=uuid.uuid4(),
            name=fake.name(),
            email=fake.unique.email(),
            phone=fake.phone_number(),
            tier=random.choices(tiers, weights=weights)[0],
            preferred_channel=random.choice(['web', 'email', 'web', 'web']),
            metadata_={
                "city": fake.city(),
                "country": fake.country(),
                "account_age_days": random.randint(30, 1500)
            }
        )
        session.add(user)
        users.append(user)

        if (i + 1) % 100 == 0:
            await session.commit()
            print(f"  Created {i + 1} users...")

    await session.commit()
    return users


# ─────────────────────────────────────────────
# STEP 3: Generate fake orders
# ─────────────────────────────────────────────

async def generate_orders(
    session: AsyncSession,
    users: list,
    count: int = 2000
):
    print(f"\nGenerating {count} fake orders...")

    products = [
        {"name": "iPhone 15 Pro", "category": "phones", "price": 999.99},
        {"name": "Samsung 4K TV 55inch", "category": "tv", "price": 799.99},
        {"name": "AirPods Pro 2nd Gen", "category": "audio", "price": 249.99},
        {"name": "MacBook Air M2", "category": "laptops", "price": 1299.99},
        {"name": "iPad Pro 12.9", "category": "tablets", "price": 1099.99},
        {"name": "Sony WH-1000XM5", "category": "audio", "price": 399.99},
        {"name": "Nintendo Switch OLED", "category": "gaming", "price": 349.99},
        {"name": "Kindle Paperwhite", "category": "ereaders", "price": 139.99},
        {"name": "Apple Watch Series 9", "category": "wearables", "price": 399.99},
        {"name": "GoPro Hero 12", "category": "cameras", "price": 399.99},
    ]

    statuses = [
        'delivered', 'delivered', 'delivered', 'delivered', 
        'delivered', 'shipped', 'processing', 'cancelled'
    ]

    for i in range(count):
        user = random.choice(users)
        product = random.choice(products)
        order_date = fake.date_between(start_date='-6m', end_date='today')
        status = random.choice(statuses)

        days_since_order = (datetime.now().date() - order_date).days
        delivered_date = None
        if status == 'delivered':
            delivered_date = order_date + timedelta(days=random.randint(2, 7))

        refund_eligible = days_since_order <= 30

        order = Order(
            id=f"ORD-{10000 + i}",
            customer_id=user.id,
            product_name=product['name'],
            product_category=product['category'],
            amount=product['price'],
            status=status,
            ordered_at=datetime.combine(order_date, datetime.min.time()),
            delivered_at=datetime.combine(delivered_date, datetime.min.time()) if delivered_date else None,
            tracking_number=fake.bothify(text='??########').upper(),
            refund_eligible=refund_eligible,
            refund_window_days=30
        )

        session.add(order)

        if (i + 1) % 500 == 0:
            await session.commit()
            print(f"  Created {i + 1} orders...")

    await session.commit()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def main():
    print("=" * 50)
    print("NexusAI Data Setup")
    print("=" * 50)

    # Ensure tables exist
    await create_tables()

    async with AsyncSessionLocal() as session:
        # Load Bitext knowledge base
        await clear_data(session)

        await load_knowledge_base(session)

        # Generate users
        users = await generate_users(session, count=500)

        # Generate orders for those users
        await generate_orders(session, users, count=2000)

    print("\n" + "=" * 50)
    print("Data setup complete!")
    print("Knowledge base: ~27 intent entries")
    print("Users: 500")
    print("Orders: 2000")
    print("Test set: data/processed/test_set.csv")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())