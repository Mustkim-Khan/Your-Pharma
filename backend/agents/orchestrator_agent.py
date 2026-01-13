"""
Orchestrator Agent
Model: gpt-5.2
Purpose: Coordinate all agents, decide execution order, resolve conflicts, maintain system state
"""
import json
import uuid
from openai import OpenAI
from datetime import datetime, timedelta
import os

# Replaced Langfuse with LangSmith
from utils.tracing_utils import traceable, wrap_openai, tracing_context

from models.schemas import (
    ChatRequest, ChatResponse, OrderPreview, Order, OrderItem,
    ExtractionResult, SafetyCheckResult, SafetyDecision, RefillPrediction,
    OrderStatus
)
from services.data_service import data_service
from agents.extraction_agent import extraction_agent
from agents.safety_agent import safety_agent
from agents.refill_agent import refill_agent
from agents.fulfillment_agent import fulfillment_agent



ORCHESTRATOR_SYSTEM_PROMPT = """You are an intelligent, conversational AI pharmacist assistant for an autonomous pharmacy system.

## YOUR IDENTITY
You are a professional, helpful pharmacy AI assistant. You maintain a warm, conversational tone while being accurate and efficient.

## CRITICAL: PATIENT CONTEXT AWARENESS
You will receive the ACTUAL patient information from their medical records. This is the SOURCE OF TRUTH:
- The patient's REAL name is provided in the system context
- Their patient ID, order history, and medication records are provided
- If a user claims a DIFFERENT name than what's in the records, politely clarify: "I see you're logged in as [ACTUAL NAME] (Patient ID: [ID]). If this is incorrect, please contact our support team."
- NEVER accept a different identity - the selected patient context is authoritative

## YOUR ROLE
1. Have natural, helpful conversations about medication needs
2. Remember context from previous messages in the conversation
3. Coordinate medication orders, refills, and inquiries
4. Provide PERSONALIZED responses using the patient's actual name
6. ALWAYS reply in the SAME LANGUAGE/SCRIPT as the user.
   - If user speaks Spanish -> reply in Spanish.
   - If user speaks Hinglish (Hindi in English script) -> reply in Hinglish.
   - If user speaks English -> reply in English.

## TOOLS
You have access to tools to help the user:
- `extract_order_details`: Call this when the user indicates they want to buy, order, or get medication.
- `check_refill_status`: Call this when the user asks about refills, "do I need anything", or checks status.
- `check_order_status`: Call this when the user asks "where is my order" or about specific order status.
- `confirm_last_order`: Call this when the user explicitly confirms a pending order ("yes", "confirm", "ok").
- `cancel_order`: Call this when the user cancels ("no", "cancel", "stop").
- `place_order`: Call this if the user is explicit about placing an order now.

## GUIDELINES
- If the user just says "Hi" or asks a general question, just reply normally.
- If the user wants to do something, USE THE TOOL.
"""


class OrchestratorAgent:
    def __init__(self):
        # Wrap OpenAI client for tracing
        self.client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        self.model = "gpt-5.2"
        # Removed self.langfuse = Langfuse()
        self.pending_previews = {}  # Store pending order previews
        self.session_contexts = {}  # Store session contexts
        self.conversation_histories = {}  # Store conversation histories by session
    
    def _generate_preview_id(self) -> str:
        """Generate unique preview ID"""
        return f"PRV-{uuid.uuid4().hex[:8].upper()}"
    
    @traceable(run_type="chain", name="OrchestratorAgent.process_message")
    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """
        Process a user message through the agent pipeline
        """
        # Try to get trace_id using LangSmith helper
        trace_id = None
        try:
            trace = tracing_context.get_current_trace()
            trace_id = trace.id if trace else None
        except Exception:
            pass
        
        # Get patient context
        patient = data_service.get_patient_by_id(request.patient_id)
        if not patient:
            return ChatResponse(
                message="I couldn't find your patient record. Please select a valid patient.",
                trace_url=self._get_trace_url(trace_id)
            )
        
        patient_history = data_service.get_patient_order_history(request.patient_id)
        patient_context = {
            "patient_id": patient.patient_id,
            "patient_name": patient.patient_name,
            "recent_orders": patient_history.tail(5).to_dict('records') if not patient_history.empty else []
        }
        
        # Get conversation history
        session_id = request.session_id or request.patient_id
        if request.conversation_history:
            self.conversation_histories[session_id] = list(request.conversation_history)
        conversation_history = self.conversation_histories.get(session_id, [])
        
        # Build messages for LLM
        messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}]
        
        # Add patient context info
        patient_info = f"""Current Patient Information:
- Name: {patient_context.get('patient_name', 'Unknown')}
- ID: {patient_context.get('patient_id', 'N/A')}
- Recent Orders: {len(patient_context.get('recent_orders', []))} orders on file"""
        messages.append({"role": "system", "content": patient_info})

        # --- FIX: INJECT PENDING ORDER STATUS ---
        # Check if there's a pending order preview for this session
        preview_id = self.session_contexts.get(session_id)
        if preview_id and preview_id in self.pending_previews:
            preview = self.pending_previews[preview_id]
            items_summary = ", ".join([f"{item.medicine_name} {item.strength} x{item.quantity}" for item in preview.items])
            
            pending_status_msg = f"""
SYSTEM STATUS: 
The user currently has a PENDING ORDER (ID: {preview_id}) waiting for confirmation.
Items: {items_summary}
Total: ${preview.total_amount:.2f}

CRITICAL INSTRUCTION:
- If the user says "confirm", "yes", "place it", "ok", or similar affirmative, you MUST use the `confirm_last_order` tool.
- If the user says "cancel", "no", "stop", use the `cancel_order` tool.
- Do NOT use `extract_order_details` unless the user explicitly asks to CHANGE the order or add different items.
"""
            messages.append({"role": "system", "content": pending_status_msg})
        # ----------------------------------------

        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add current message
        messages.append({"role": "user", "content": request.message})
        
        # Call LLM with Tools
        from models.tools import ALL_TOOLS
        
        try:
            llm_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                tools=ALL_TOOLS,
                tool_choice="auto" 
            )
            
            message = llm_response.choices[0].message
            tool_calls = message.tool_calls
            
            response = None
            
            if tool_calls:
                # Handle all tool calls (though usually just one)
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    print(f"[Orchestrator] Selected Tool: {tool_name}")
                    tracing_context.update_current_observation(metadata={"tool_selected": tool_name})
                    
                    if tool_name in ["extract_order_details", "place_order"]:
                        tracing_context.update_current_observation(metadata={"reasoning": "User intent implies order placement. Proceeding to extraction and validation."})
                        response = await self._handle_order(request, patient, patient_context, trace_id)
                    elif tool_name == "check_refill_status":
                        response = await self._handle_refill_check(request, patient, trace_id)
                    elif tool_name == "confirm_last_order":
                        response = await self._handle_order_confirmation(request, patient, trace_id)
                    elif tool_name == "cancel_order":
                        response = await self._handle_order_cancellation(request, trace_id)
                    elif tool_name == "check_order_status":
                        response = await self._handle_status_check(request, trace_id)
            else:
                # Natural conversation
                tracing_context.update_current_observation(metadata={"response_type": "natural_conversation"})
                response = ChatResponse(
                    message=message.content
                )
                
            # Update History
            if session_id not in self.conversation_histories:
                self.conversation_histories[session_id] = []
            self.conversation_histories[session_id].append({"role": "user", "content": request.message})
            if response and response.message:
                self.conversation_histories[session_id].append({"role": "assistant", "content": response.message})
                
            return response
            
        except Exception as e:
            print(f"Orchestrator Error: {e}")
            return ChatResponse(
                message=f"I encountered an error processing your request. Error: {str(e)}"
            )
    
    @traceable(run_type="chain", name="Orchestrator._handle_order")
    async def _handle_order(self, request: ChatRequest, patient, patient_context: dict, trace_id: str) -> ChatResponse:
        """Handle medication order requests"""
        
        # Step 1: Extract entities with conversation history for context
        extraction_result = extraction_agent.extract(
            request.message, 
            patient_context,
            request.conversation_history  # Pass conversation history for context
        )
        tracing_context.update_current_observation(metadata={
            "step": "parse_medicine_request",
            "extracted": extraction_result.model_dump() if hasattr(extraction_result, 'model_dump') else str(extraction_result)
        })
        
        if extraction_result.needs_clarification:
            return ChatResponse(
                message=extraction_result.clarification_message,
                extracted_entities=extraction_result
            )
        
        if not extraction_result.entities:
            return ChatResponse(
                message="I couldn't identify any medications in your request. Could you please specify which medicine you need?",
                extracted_entities=extraction_result
            )
        
        # Step 2: Find matching medicines
        matched_medicines = []
        tracing_context.update_current_observation(metadata={"step": "medicine_search", "intent": "Checking inventory for extracted medicines"})
        for entity in extraction_result.entities:
            search_results = data_service.search_medicine(entity.medicine)
            if search_results:
                # Find best match considering dosage
                for med in search_results:
                    if not entity.dosage or entity.dosage.lower() in med.strength.lower():
                        matched_medicines.append(med)
                        break
                else:
                    matched_medicines.append(search_results[0])
        
        if not matched_medicines:
            return ChatResponse(
                message=f"I couldn't find '{extraction_result.entities[0].medicine}' in our inventory. Please check the spelling or try a different medication.",
                extracted_entities=extraction_result
            )
        
        # Step 3: Safety check with full context
        safety_result = safety_agent.evaluate(
            extraction_result.entities,
            matched_medicines,
            has_prescription=False,  # Will be updated based on patient records
            patient_context=patient_context,
            conversation_history=request.conversation_history
        )
        tracing_context.update_current_observation(metadata={
            "step": "check_safety_and_inventory",
            "result": {
                "decision": str(safety_result.decision),
                "reasons": safety_result.reasons,
                "requires_prescription": safety_result.requires_prescription
            }
        })
        
        # Create order preview
        if safety_result.decision == SafetyDecision.REJECT:
            return ChatResponse(
                message=f"I'm sorry, but I cannot process this order. {' '.join(safety_result.reasons)}",
                extracted_entities=extraction_result,
                safety_result=safety_result
            )
        
        # Step 4: Create order preview
        tracing_context.update_current_observation(metadata={"step": "execute_order_preview"})
        preview_id = self._generate_preview_id()
        items = []
        
        for i, entity in enumerate(extraction_result.entities):
            if i < len(matched_medicines):
                med = matched_medicines[i]
                quantity = entity.quantity if entity.quantity > 0 else 30  # Default quantity
                if safety_result.allowed_quantity:
                    quantity = min(quantity, safety_result.allowed_quantity)
                
                items.append(OrderItem(
                    medicine_id=med.medicine_id,
                    medicine_name=med.medicine_name,
                    strength=med.strength,
                    quantity=quantity,
                    prescription_required=med.prescription_required
                ))
        
        # Price map for calculating actual prices (same as fulfillment_agent)
        price_map = {
            "Paracetamol": 0.15,
            "Metformin": 0.20,
            "Atorvastatin": 0.85,
            "Lisinopril": 0.55,
            "Amlodipine": 0.65,
            "Omeprazole": 0.40,
            "Amoxicillin": 0.35,
            "Ibuprofen": 0.20,
            "Aspirin": 0.10,
        }
        
        # Set actual prices on items
        for item in items:
            item.unit_price = price_map.get(item.medicine_name, 0.50)
        
        subtotal = sum(item.unit_price * item.quantity for item in items)
        
        preview = OrderPreview(
            preview_id=preview_id,
            patient_id=patient.patient_id,
            patient_name=patient.patient_name,
            items=items,
            total_amount=round(subtotal, 2),
            safety_decision=safety_result.decision,
            safety_reasons=safety_result.reasons,
            requires_prescription=safety_result.requires_prescription,
            created_at=datetime.now()
        )
        
        # Store preview for confirmation
        self.pending_previews[preview_id] = preview
        self.session_contexts[request.session_id or request.patient_id] = preview_id
        
        # Build response message
        items_summary = ", ".join([f"{item.medicine_name} {item.strength} x{item.quantity}" for item in items])
        
        if safety_result.decision == SafetyDecision.CONDITIONAL:
            message = f"I can prepare your order for {items_summary}. However: {' '.join(safety_result.reasons)}\n\nWould you like to proceed? Reply 'confirm' to place the order or 'cancel' to cancel."
        else:
            message = f"Great! I've prepared your order for {items_summary}.\n\nEstimated total: ${preview.total_amount:.2f}\n\nPlease reply 'confirm' to place the order or 'cancel' to cancel."
        
        tracing_context.update_current_observation(metadata={
            "preview_id": preview_id,
            "total": preview.total_amount,
            "items": items_summary
        })
        
        return ChatResponse(
            message=message,
            extracted_entities=extraction_result,
            safety_result=safety_result,
            order_preview=preview,
            requires_confirmation=True
        )
    
    @traceable(run_type="chain", name="Orchestrator._handle_order_confirmation")
    async def _handle_order_confirmation(self, request: ChatRequest, patient, trace_id: str) -> ChatResponse:
        """Handle order confirmation"""
        session_id = request.session_id or request.patient_id
        preview_id = self.session_contexts.get(session_id)
        
        if not preview_id or preview_id not in self.pending_previews:
            return ChatResponse(
                message="I don't see any pending order to confirm. Would you like to place a new order?"
            )
        
        preview = self.pending_previews[preview_id]
        
        # Create the actual order
        order = fulfillment_agent.create_order(
            patient_id=patient.patient_id,
            patient_name=patient.patient_name,
            patient_email=patient.patient_email,
            patient_phone=patient.patient_phone,
            items=preview.items
        )
        
        # Step 1: Record safety validation and update to VALIDATED status
        fulfillment_agent.record_safety_validation(
            order.order_id, 
            preview.safety_decision.value if hasattr(preview.safety_decision, 'value') else preview.safety_decision, 
            preview.safety_reasons
        )
        fulfillment_agent.update_order_status(order.order_id, OrderStatus.VALIDATED, "Safety validation completed")
        
        # Step 2: Record order confirmation and update to CONFIRMED status
        fulfillment_agent.record_order_confirmed(order.order_id)
        fulfillment_agent.update_order_status(order.order_id, OrderStatus.CONFIRMED, "Order confirmed by patient")
        
        # Step 3: Update inventory and record event
        total_quantity = 0
        for item in order.items:
            data_service.update_stock(item.medicine_id, item.quantity)
            total_quantity += item.quantity
        
        fulfillment_agent.record_inventory_updated(order.order_id, total_quantity)
        
        # Step 4: Record fulfillment initiated and update to PROCESSING status  
        fulfillment_agent.record_fulfillment_initiated(order.order_id)
        fulfillment_agent.update_order_status(order.order_id, OrderStatus.PROCESSING, "Order is being processed for delivery")
        
        # Add to order history
        for item in order.items:
            data_service.add_order({
                "order_id": order.order_id,
                "patient_id": patient.patient_id,
                "patient_name": patient.patient_name,
                "patient_email": patient.patient_email,
                "patient_phone": patient.patient_phone,
                "medicine": item.medicine_name,
                "medicine_id": item.medicine_id,
                "dosage": item.strength,
                "quantity": item.quantity,
                "purchase_date": datetime.now().strftime("%Y-%m-%d"),
                "supply_days": 30,
                "prescription_id": order.prescription_id or "null",
                "order_status": "PROCESSING"
            })
        
        # Generate receipt
        receipt = fulfillment_agent.generate_receipt(order)
        
        # Clean up preview
        del self.pending_previews[preview_id]
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]
        
        items_summary = ", ".join([f"{item.medicine_name} {item.strength} x{item.quantity}" for item in order.items])
        
        # Calculate grand total (same as frontend: subtotal + 5% tax + $2 delivery)
        tax = order.total_amount * 0.05
        delivery = 2.00
        grand_total = order.total_amount + tax + delivery
        
        # Calculate Next Refill Date (assuming 30 days supply)
        # In a real system, this would vary by quantity/dosage
        next_refill_date = datetime.now() + timedelta(days=30)
        formatted_refill_date = next_refill_date.strftime("%B %d, %Y")
        
        # Check for OTHER refills that might be needed (Proactive AI)
        other_refills_msg = ""
        try:
            medication_history = data_service.get_medicines_needing_refill(patient.patient_id, datetime.now())
            # Filter out the meds we just ordered
            ordered_ids = [item.medicine_id for item in order.items]
            medication_history = [m for m in medication_history if m['medicine_id'] not in ordered_ids]
            
            if medication_history:
                predictions = refill_agent.predict(patient.patient_id, patient.patient_name, medication_history)
                urgent_refills = [p for p in predictions if p.action in ["REMIND", "BLOCK"]]
                
                if urgent_refills:
                    other_refills_msg = "\n\n**⚠️ Proactive Alert:**\nWhile reviewing your records, I noticed you are also running low on:\n"
                    for ref in urgent_refills:
                        other_refills_msg += f"- {ref.medicine} ({ref.days_remaining} days remaining)\n"
                    other_refills_msg += "\nWould you like to add these to a new order?"
        except Exception as e:
            print(f"Error checking other refills: {e}")
            
        message = f"""✅ **Order Confirmed!**

**Order ID:** {order.order_id}
**Items:** {items_summary}
**Subtotal:** ${order.total_amount:.2f}
**Tax (5%):** ${tax:.2f}
**Delivery:** ${delivery:.2f}
**Total:** ${grand_total:.2f}

**Receipt #:** {receipt.get('receipt_number', 'N/A')}

{receipt.get('thank_you_message', 'Thank you for your order!')}

---
**🤖 AI Planner Activated:**
I have updated your medication schedule. Based on this 30-day supply, I will proactively remind you to refill this prescription around **{formatted_refill_date}**. You don't need to track this yourself.
{other_refills_msg}"""
        
        return ChatResponse(
            message=message,
            order=order
        )
    
    @traceable(run_type="chain", name="Orchestrator._handle_order_cancellation")
    async def _handle_order_cancellation(self, request: ChatRequest, trace_id: str) -> ChatResponse:
        """Handle order cancellation"""
        session_id = request.session_id or request.patient_id
        preview_id = self.session_contexts.get(session_id)
        
        if preview_id and preview_id in self.pending_previews:
            del self.pending_previews[preview_id]
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]
        
        return ChatResponse(
            message="Your order has been cancelled. Is there anything else I can help you with?"
        )
    
    @traceable(run_type="chain", name="Orchestrator._handle_refill_check")
    async def _handle_refill_check(self, request: ChatRequest, patient, trace_id: str) -> ChatResponse:
        """Handle refill check requests"""
        medication_history = data_service.get_medicines_needing_refill(
            patient.patient_id,
            datetime.now()
        )
        
        if not medication_history:
            return ChatResponse(
                message=f"Hi {patient.patient_name}! I checked your medication history, and you don't have any refills due at the moment. All your medications should be well-stocked."
            )
        
        predictions = refill_agent.predict(
            patient.patient_id,
            patient.patient_name,
            medication_history
        )
        
        if not predictions:
            return ChatResponse(
                message=f"Hi {patient.patient_name}! Your medications are all looking good - no urgent refills needed right now."
            )
        
        # Build response
        refill_messages = []
        for pred in predictions:
            status = f"**{pred.medicine}**: {pred.days_remaining} days remaining"
            if pred.action == "REMIND":
                status += " ⚠️ (refill soon)"
            elif pred.action == "AUTO_REFILL":
                status += " 🔄 (auto-refill eligible)"
            elif pred.action == "BLOCK":
                status += " ❌ (action required)"
            refill_messages.append(status)
        
        message = f"Hi {patient.patient_name}! Here's your medication refill status:\n\n" + "\n".join(refill_messages)
        
        if any(p.action == "REMIND" for p in predictions):
            message += "\n\nWould you like me to prepare a refill order for any of these?"
        
        return ChatResponse(
            message=message,
            refill_suggestions=predictions
        )
    
    @traceable(run_type="chain", name="Orchestrator._handle_status_check")
    async def _handle_status_check(self, request: ChatRequest, trace_id: str) -> ChatResponse:
        """Handle order status check"""
        orders = fulfillment_agent.get_all_orders()
        patient_orders = [o for o in orders if o.patient_id == request.patient_id]
        
        if not patient_orders:
            return ChatResponse(
                message="You don't have any recent orders. Would you like to place a new order?"
            )
        
        latest = patient_orders[-1]
        items_summary = ", ".join([f"{item.medicine_name} x{item.quantity}" for item in latest.items])
        
        status_emoji = {
            "PENDING": "⏳",
            "CONFIRMED": "✅",
            "PREPARING": "📦",
            "PROCESSING": "🚚",
            "COMPLETED": "✔️",
            "CANCELLED": "❌"
        }
        
        message = f"""**Order Status: {latest.order_id}**

{status_emoji.get(latest.status.value, '📋')} Status: {latest.status.value}
📋 Items: {items_summary}
💰 Total: ${latest.total_amount:.2f}
📅 Ordered: {latest.created_at.strftime('%Y-%m-%d %H:%M')}
"""
        
        return ChatResponse(
            message=message,
            order=latest
        )
    
    def _get_trace_url(self, trace_id: str) -> str:
        """Generate LangSmith trace URL"""
        if not trace_id:
            return None
        # LangSmith URL format
        return f"https://smith.langchain.com/trace/{trace_id}"


# Singleton instance
orchestrator_agent = OrchestratorAgent()
