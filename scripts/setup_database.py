
import asyncio
import sys
import os

# Add the project root to the Python path so it can find the infrastructure module
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from infrastructure.db.connection import create_tables

async def main():
    print("Setting up database...")
    await create_tables()
    print("Database setup complete")

if __name__ == "__main__":
    asyncio.run(main())