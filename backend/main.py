"""
Agentic AI Pharmacy System - FastAPI Backend
Main application entry point with all API endpoints
"""
import os
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# Load environment variables
load_dotenv(override=True)

# Import LangSmith utilities
from utils.tracing_utils import traceable, tracing_context

# Import services and agents
from services.data_service import data_service
from services.voice_service import voice_service
from agents.orchestrator_agent import orchestrator_agent
from agents.refill_agent import refill_agent
from agents.fulfillment_agent import fulfillment_agent
from models.schemas import (
    ChatRequest, ChatResponse, VoiceRequest, VoiceResponse,
    Medicine, Patient, Order, OrderStatus, RefillPrediction,
    WarehouseWebhookPayload
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    print("🚀 Starting Agentic AI Pharmacy System...")
    print(f"📊 LangSmith enabled: {bool(os.getenv('LANGCHAIN_API_KEY'))}")
    print(f"🤖 OpenAI enabled: {bool(os.getenv('OPENAI_API_KEY'))}")
    yield
    print("👋 Shutting down Agentic AI Pharmacy System...")


app = FastAPI(
    title="Agentic AI Pharmacy System",
    description="Autonomous pharmacy system with multi-agent architecture",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Health Check
# ============================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "system": "Agentic AI Pharmacy System",
        "version": "1.0.0",
        "agents": [
            "Conversational Extraction Agent (gpt-5-mini)",
            "Safety & Prescription Policy Agent (gpt-5.2)",
            "Predictive Refill Intelligence Agent (gpt-5.2)",
            "Inventory & Fulfillment Agent (gpt-5-mini)",
            "Orchestrator Agent (gpt-5.2)"
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "data_service": "ok",
            "langsmith": "ok" if os.getenv("LANGCHAIN_API_KEY") else "not configured",
            "openai": "ok" if os.getenv("OPENAI_API_KEY") else "not configured"
        }
    }


# ============================================
# Chat Endpoints
# ============================================

@app.post("/api/chat", response_model=ChatResponse)
@traceable(run_type="chain", name="API: /api/chat")
async def chat(request: ChatRequest):
    """
    Process a chat message through the agent pipeline
    """
    try:
        response = await orchestrator_agent.process_message(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Voice Endpoints
# ============================================

@app.post("/api/voice", response_model=VoiceResponse)
@traceable(run_type="chain", name="API: /api/voice")
async def voice_chat(request: VoiceRequest):
    """
    Process voice input through the agent pipeline
    """
    try:
        # Transcribe audio
        transcript, error = voice_service.process_voice_input(request.audio_base64)
        if error:
            return VoiceResponse(
                transcript="",
                chat_response=ChatResponse(message=error),
                audio_response_base64=voice_service.generate_voice_response(error)
            )
        
        # Process through chat pipeline
        chat_request = ChatRequest(
            patient_id=request.patient_id,
            message=transcript,
            session_id=request.session_id
        )
        chat_response = await orchestrator_agent.process_message(chat_request)
        
        # Generate voice response
        audio_response = voice_service.generate_voice_response(chat_response.message)
        
        return VoiceResponse(
            transcript=transcript,
            chat_response=chat_response,
            audio_response_base64=audio_response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Patient Endpoints
# ============================================

@app.get("/api/patients", response_model=List[Patient])
async def get_patients():
    """Get all patients"""
    return data_service.get_all_patients()


@app.get("/api/patients/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str):
    """Get patient by ID"""
    patient = data_service.get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


# ============================================
# Order Endpoints
# ============================================

@app.get("/api/orders")
async def get_orders(patient_id: Optional[str] = None):
    """Get all orders with timeline events, optionally filtered by patient"""
    orders = fulfillment_agent.get_all_orders_with_events()
    if patient_id:
        orders = [o for o in orders if o.get('patient_id') == patient_id]
    return orders


@app.get("/api/orders/{order_id}")
async def get_order(order_id: str):
    """Get order by ID"""
    order = fulfillment_agent.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@app.post("/api/orders/{order_id}/confirm")
@traceable(run_type="chain", name="API: /api/orders/confirm")
async def confirm_order(order_id: str, background_tasks: BackgroundTasks):
    """Confirm a pending order"""
    order = fulfillment_agent.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    if order.status != OrderStatus.PENDING:
        raise HTTPException(status_code=400, detail=f"Order is not pending. Current status: {order.status.value}")
    
    # Update status
    fulfillment_agent.update_order_status(order_id, OrderStatus.CONFIRMED, "Order confirmed")
    
    # Update inventory
    for item in order.items:
        data_service.update_stock(item.medicine_id, item.quantity)
    
    # Trigger webhook in background
    background_tasks.add_task(fulfillment_agent.trigger_warehouse_webhook, order)
    
    # Progress to preparing
    fulfillment_agent.update_order_status(order_id, OrderStatus.PREPARING, "Preparing order")
    
    return {"status": "confirmed", "order_id": order_id}


@app.post("/api/orders/{order_id}/cancel")
async def cancel_order(order_id: str):
    """Cancel an order"""
    order = fulfillment_agent.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    if order.status in [OrderStatus.COMPLETED, OrderStatus.CANCELLED]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel order. Current status: {order.status.value}")
    
    fulfillment_agent.update_order_status(order_id, OrderStatus.CANCELLED, "Order cancelled by user")
    return {"status": "cancelled", "order_id": order_id}


# ============================================
# Inventory Endpoints
# ============================================

@app.get("/api/inventory")
async def get_inventory():
    """Get all medicines in inventory"""
    return data_service.get_all_medicines()


@app.get("/api/inventory/stats")
async def get_inventory_stats():
    """Get inventory statistics for admin dashboard"""
    return data_service.get_inventory_stats()


@app.get("/api/inventory/search")
async def search_inventory(q: str):
    """Search medicines by name"""
    return data_service.search_medicine(q)


@app.get("/api/inventory/{medicine_id}", response_model=Medicine)
async def get_medicine(medicine_id: str):
    """Get medicine by ID"""
    medicine = data_service.get_medicine_by_id(medicine_id)
    if not medicine:
        raise HTTPException(status_code=404, detail="Medicine not found")
    return medicine


# ============================================
# Refill Endpoints
# ============================================

@app.get("/api/refills")
@traceable(run_type="chain", name="API: /api/refills")
async def get_refills():
    """Get all proactive refill alerts"""
    predictions = refill_agent.get_all_patient_refills(data_service, datetime.now())
    return predictions


@app.get("/api/refills/{patient_id}")
@traceable(run_type="chain", name="API: /api/refills/patient")
async def get_patient_refills(patient_id: str):
    """Get refill alerts for a specific patient"""
    patient = data_service.get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    medication_history = data_service.get_medicines_needing_refill(patient_id, datetime.now())
    if not medication_history:
        return []
    
    predictions = refill_agent.predict(
        patient_id,
        patient.patient_name,
        medication_history
    )
    return predictions


# ============================================
# Webhook Endpoints (Mock)
# ============================================

@app.post("/api/webhook/warehouse")
async def warehouse_webhook(payload: WarehouseWebhookPayload):
    """
    Mock warehouse fulfillment webhook
    Simulates receiving and processing a fulfillment request
    """
    print(f"📦 Warehouse received order: {payload.order_id}")
    print(f"   Items: {len(payload.items)}")
    print(f"   Delivery type: {payload.delivery_type}")
    print(f"   Patient: {payload.patient_name}")
    
    # Simulate processing
    order = fulfillment_agent.get_order(payload.order_id)
    if order:
        fulfillment_agent.update_order_status(
            payload.order_id, 
            OrderStatus.PROCESSING, 
            "Order received by warehouse, processing for shipment"
        )
    
    return {
        "status": "received",
        "order_id": payload.order_id,
        "warehouse_id": "WH-CENTRAL-001",
        "estimated_ship_date": (datetime.now()).strftime("%Y-%m-%d"),
        "tracking_number": f"TRK-{payload.order_id[-8:]}"
    }


# ============================================
# Observability Endpoints
# ============================================

@app.get("/api/traces/{order_id}")
async def get_trace_link(order_id: str):
    """Get LangSmith trace link for an order"""
    order = fulfillment_agent.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # Note: Retrieving the exact trace ID for a completed order might require storing it.
    # Currently, our agents update the run tree but we might not be storing the root trace ID on the order object persistently
    # unless we explicitly added that logic.
    # For now, we return a general link or the ID if available.
    
    # LangSmith doesn't have a simple public trace URL structure like Langfuse without sharing setup.
    # But we'll format it as a deep link to the LangSmith UI project if possible.
    
    # If using LangSmith, trace_id is usually the run ID.
    
    trace_url = None
    if hasattr(order, 'trace_id') and order.trace_id:
         trace_url = f"https://smith.langchain.com/trace/{order.trace_id}"
    
    return {
        "order_id": order_id,
        "trace_url": trace_url,
        "trace_id": getattr(order, 'trace_id', None)
    }


# ============================================
# Agent Status Endpoints
# ============================================

@app.get("/api/agents/status")
async def get_agent_status():
    """Get status of all agents"""
    return {
        "agents": [
            {
                "name": "Conversational Extraction Agent",
                "model": "gpt-5-mini",
                "status": "active",
                "purpose": "Parse natural language to extract medicine orders"
            },
            {
                "name": "Safety & Prescription Policy Agent",
                "model": "gpt-5.2",
                "status": "active",
                "purpose": "High-stakes safety reasoning and policy enforcement"
            },
            {
                "name": "Predictive Refill Intelligence Agent",
                "model": "gpt-5.2",
                "status": "active",
                "purpose": "Proactive refill predictions and reminders"
            },
            {
                "name": "Inventory & Fulfillment Agent",
                "model": "gpt-5-mini",
                "status": "active",
                "purpose": "Execute orders, update inventory, trigger webhooks"
            },
            {
                "name": "Orchestrator Agent",
                "model": "gpt-5.2",
                "status": "active",
                "purpose": "Coordinate all agents and maintain system state"
            }
        ],
        "voice": {
            "stt_model": "gpt-4o-mini-transcribe",
            "tts_model": "gpt-4o-mini-tts",
            "status": "active"
        },
        "observability": {
            "provider": "LangSmith",
            "status": "active" if os.getenv("LANGCHAIN_API_KEY") else "not configured"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)