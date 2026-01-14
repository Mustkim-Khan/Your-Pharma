"""
Pydantic models for the Agentic AI Pharmacy System
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    CONFIRMED = "CONFIRMED"
    PREPARING = "PREPARING"
    PROCESSING = "PROCESSING"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class SafetyDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    CONDITIONAL = "CONDITIONAL"


class RefillAction(str, Enum):
    REMIND = "REMIND"
    AUTO_REFILL = "AUTO_REFILL"
    BLOCK = "BLOCK"


# Medicine Models
class Medicine(BaseModel):
    medicine_id: str
    medicine_name: str
    strength: str
    form: str
    stock_level: int
    prescription_required: bool
    category: str
    discontinued: bool
    max_quantity_per_order: int = 30
    controlled_substance: bool = False


# Patient Models
class Patient(BaseModel):
    patient_id: str
    patient_name: str
    patient_email: str
    patient_phone: str


# Extraction Agent Output
class ExtractedEntity(BaseModel):
    medicine: str = ""
    dosage: str = ""
    frequency: str = ""
    quantity: int = 0
    confidence: float = 0.0
    raw_text: str = ""


class ExtractionResult(BaseModel):
    entities: List[ExtractedEntity]
    needs_clarification: bool = False
    clarification_message: str = ""


# Safety Agent Output
class SafetyCheckResult(BaseModel):
    decision: SafetyDecision
    reasons: List[str] = []
    allowed_quantity: Optional[int] = None
    requires_followup: bool = False
    requires_prescription: bool = False
    blocked_items: List[str] = []


# Refill Agent Output
class RefillPrediction(BaseModel):
    patient_id: str
    patient_name: str
    medicine: str
    medicine_id: str
    days_remaining: int
    last_purchase_date: str
    action: RefillAction
    justification: str
    urgency: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "LOW"


# Order Models
class OrderItem(BaseModel):
    medicine_id: str
    medicine_name: str
    strength: str
    quantity: int
    unit_price: float = 0.0
    prescription_required: bool = False


class OrderPreview(BaseModel):
    preview_id: str
    patient_id: str
    patient_name: str
    items: List[OrderItem]
    total_amount: float
    safety_decision: SafetyDecision
    safety_reasons: List[str]
    requires_prescription: bool
    created_at: datetime


class Order(BaseModel):
    order_id: str
    patient_id: str
    patient_name: str
    patient_email: str
    patient_phone: str
    items: List[OrderItem]
    total_amount: float
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    prescription_id: Optional[str] = None
    trace_id: Optional[str] = None


class OrderStatusUpdate(BaseModel):
    order_id: str
    status: OrderStatus
    message: str
    timestamp: datetime


# Webhook Payload
class WarehouseWebhookPayload(BaseModel):
    order_id: str
    items: List[dict]
    delivery_type: Literal["HOME_DELIVERY", "PICKUP"] = "HOME_DELIVERY"
    patient_name: str
    patient_address: str = "123 Main St, City, State 12345"
    priority: Literal["NORMAL", "EXPRESS"] = "NORMAL"


# Chat Models
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[dict] = None


class ChatRequest(BaseModel):
    patient_id: str
    message: str
    session_id: Optional[str] = None
    conversation_history: List[dict] = []  # List of {"role": "user"|"assistant", "content": "..."}



class ChatResponse(BaseModel):
    message: str
    extracted_entities: Optional[ExtractionResult] = None
    safety_result: Optional[SafetyCheckResult] = None
    order_preview: Optional[OrderPreview] = None
    order: Optional[Order] = None
    refill_suggestions: List[RefillPrediction] = []
    trace_url: Optional[str] = None
    requires_confirmation: bool = False


# Agent Trace Models
class AgentTrace(BaseModel):
    trace_id: str
    agent_name: str
    input_data: dict
    output_data: dict
    model_used: str
    tokens_used: int = 0
    latency_ms: int = 0
    timestamp: datetime


# Voice Models
class VoiceRequest(BaseModel):
    audio_base64: str
    patient_id: str
    session_id: Optional[str] = None


class VoiceResponse(BaseModel):
    transcript: str
    chat_response: ChatResponse
    audio_response_base64: Optional[str] = None


# Structured Decision Schema
class AgentDecision(BaseModel):
    agent: str
    decision: Literal["APPROVED", "REJECTED", "NEEDS_INFO", "SCHEDULED"]
    reason: str
    message: Optional[str] = None
    evidence: List[str]
    next_agent: Optional[str] = None
