import asyncio
from infrastructure.db.connection import create_tables

if __name__ == "__main__":
    asyncio.run(create_tables())
