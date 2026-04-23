# apps/api/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

load_dotenv()

from infrastructure.db.connection import create_tables
from apps.api.routes.chat import router as chat_router
from apps.api.routes.health import router as health_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    print("NexusAI starting up...")
    await create_tables()
    print("Database ready")
    yield
    # Shutdown
    print("NexusAI shutting down...")

app = FastAPI(
    title="NexusAI",
    description="Production Conversational AI Orchestrator",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router, prefix="/api")