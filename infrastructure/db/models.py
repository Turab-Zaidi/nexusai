
from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    DateTime, Text, ForeignKey, JSON, SmallInteger
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func
import uuid
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "nexus_users"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    name = Column(String(200), nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    phone = Column(String(50))
    tier = Column(
        String(20),
        default="standard"
    )  
    preferred_channel = Column(String(20), default="web")
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    metadata_ = Column("metadata", JSON, default={})


class Order(Base):
    __tablename__ = "orders"

    id = Column(String(50), primary_key=True)
    customer_id = Column(
        UUID(as_uuid=True),
        ForeignKey("nexus_users.id")
    )
    product_name = Column(String(200))
    product_category = Column(String(100))
    amount = Column(Float)
    status = Column(String(50))
    ordered_at = Column(DateTime(timezone=True))
    delivered_at = Column(DateTime(timezone=True))
    tracking_number = Column(String(100))
    refund_eligible = Column(Boolean, default=True)
    refund_window_days = Column(Integer, default=30)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("nexus_users.id")
    )
    channel = Column(String(50), default="web")
    current_state = Column(String(50))
    resolution_status = Column(String(50))
    escalated = Column(Boolean, default=False)
    escalation_reason = Column(String(200))
    total_turns = Column(Integer, default=0)
    total_cost_credits = Column(Float, default=0)
    started_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    ended_at = Column(DateTime(timezone=True))


class StateTransition(Base):
    __tablename__ = "state_transitions"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id")
    )
    from_state = Column(String(50))
    to_state = Column(String(50))
    agent_responsible = Column(String(100))
    triggered_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    metadata_ = Column("metadata", JSON, default={})


class LLMCall(Base):
    __tablename__ = "llm_calls"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id")
    )
    agent_name = Column(String(100))
    model = Column(String(100))
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    latency_ms = Column(Integer)
    called_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


class ToolExecution(Base):
    __tablename__ = "tool_executions"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id")
    )
    tool_name = Column(String(100))
    input_data = Column(JSON)
    output_data = Column(JSON)
    success = Column(Boolean)
    error_message = Column(Text)
    duration_ms = Column(Integer)
    required_approval = Column(Boolean, default=False)
    executed_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


class QualityEvaluation(Base):
    __tablename__ = "quality_evaluations"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id")
    )
    factual_accuracy = Column(SmallInteger)
    helpfulness = Column(SmallInteger)
    policy_compliance = Column(SmallInteger)
    tool_correctness = Column(SmallInteger)
    conversation_flow = Column(SmallInteger)
    overall_pass = Column(Boolean)
    revision_triggered = Column(Boolean, default=False)
    judge_reasoning = Column(Text)
    evaluated_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


class KnowledgeEntry(Base):
    __tablename__ = "knowledge_entries"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    category = Column(String(100))
    intent = Column(String(100))
    embedding = Column(Vector(384))  # stored as JSON string
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    agent_name = Column(String(100))
    version = Column(Integer)
    prompt_content = Column(Text)
    is_active = Column(Boolean, default=False)
    traffic_percentage = Column(Integer, default=100)
    avg_quality_score = Column(Float)
    total_uses = Column(Integer, default=0)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )