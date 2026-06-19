from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    DateTime, Text, ForeignKey, JSON, SmallInteger
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func
import uuid

class Base(DeclarativeBase):
    pass

# ==========================================
# BUSINESS DOMAIN MODELS (FINTECH PIVOT)
# ==========================================

class User(Base):
    __tablename__ = "nexus_users"

    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    phone = Column(String(50))
    tier = Column(String(20), default="standard")  # standard, premium, private_wealth
    fraud_risk_score = Column(Integer, default=15) # 1-100
    account_balance = Column(Float, default=0.0)
    preferred_channel = Column(String(20), default="web")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Card(Base):
    __tablename__ = "cards"

    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(50), ForeignKey("nexus_users.id"))
    last_4_digits = Column(String(4))
    card_type = Column(String(20)) # physical, virtual
    status = Column(String(20), default="active") # active, frozen, reported_stolen
    daily_limit = Column(Float, default=2500.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(50), ForeignKey("nexus_users.id"))
    card_id = Column(String(50), ForeignKey("cards.id"), nullable=True)
    merchant_name = Column(String(200))
    amount = Column(Float)
    status = Column(String(50)) # pending, cleared, disputed, refunded
    category = Column(String(100)) # subscription, dining, fee, transfer
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class SupportTicket(Base):
    """Replaces Episodic Memory. Summaries of past interactions."""
    __tablename__ = "support_tickets"

    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(50), ForeignKey("nexus_users.id"))
    intent = Column(String(100)) # dispute, lost_card, general
    summary = Column(JSON) # {"problem": "...", "solved_how": "...", "solution": "..."}
    status = Column(String(50)) # resolved, escalated
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# ==========================================
# OBSERVABILITY & GRAPH MODELS
# ==========================================

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(50), ForeignKey("nexus_users.id"))
    channel = Column(String(50), default="web")
    current_state = Column(String(50))
    resolution_status = Column(String(50))
    escalated = Column(Boolean, default=False)
    escalation_reason = Column(String(200))
    total_turns = Column(Integer, default=0)
    total_cost_credits = Column(Float, default=0)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True))


class StateTransition(Base):
    __tablename__ = "state_transitions"

    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(50), ForeignKey("conversations.id"))
    from_state = Column(String(50))
    to_state = Column(String(50))
    agent_responsible = Column(String(100))
    triggered_at = Column(DateTime(timezone=True), server_default=func.now())
    metadata_ = Column("metadata", JSON, default={})


class LLMCall(Base):
    __tablename__ = "llm_calls"

    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(50), ForeignKey("conversations.id"))
    agent_name = Column(String(100))
    model = Column(String(100))
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    latency_ms = Column(Integer)
    called_at = Column(DateTime(timezone=True), server_default=func.now())


class ToolExecution(Base):
    __tablename__ = "tool_executions"

    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(50), ForeignKey("conversations.id"))
    tool_name = Column(String(100))
    input_data = Column(JSON)
    output_data = Column(JSON)
    success = Column(Boolean)
    error_message = Column(Text)
    duration_ms = Column(Integer)
    required_approval = Column(Boolean, default=False)
    executed_at = Column(DateTime(timezone=True), server_default=func.now())


class QualityEvaluation(Base):
    __tablename__ = "quality_evaluations"

    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(50), ForeignKey("conversations.id"))
    factual_accuracy = Column(SmallInteger)
    helpfulness = Column(SmallInteger)
    policy_compliance = Column(SmallInteger)
    tool_correctness = Column(SmallInteger)
    conversation_flow = Column(SmallInteger)
    overall_pass = Column(Boolean)
    revision_triggered = Column(Boolean, default=False)
    judge_reasoning = Column(Text)
    evaluated_at = Column(DateTime(timezone=True), server_default=func.now())


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_name = Column(String(100))
    version = Column(Integer)
    prompt_content = Column(Text)
    is_active = Column(Boolean, default=False)
    traffic_percentage = Column(Integer, default=100)
    avg_quality_score = Column(Float)
    total_uses = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())