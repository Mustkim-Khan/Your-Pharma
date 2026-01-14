"""
Pharmacist Agent
Model: gpt-4o
Purpose: Interprets user intent, extracts order details, and routes to appropriate downstream agent.
"""
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from openai import OpenAI
from pydantic import BaseModel

from utils.tracing_utils import traceable, wrap_openai, tracing_context
from models.schemas import ChatRequest, ChatResponse, AgentDecision
from services.data_service import data_service

# Define the Tool Schema for OpenAI
RECORD_DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "record_agent_decision",
        "description": "Record the agent's decision and routing logic. MANDATORY for all responses.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Name of the agent (e.g., PharmacistAgent)"
                },
                "decision": {
                    "type": "string",
                    "enum": ["APPROVED", "REJECTED", "NEEDS_INFO", "SCHEDULED"],
                    "description": "The decision outcome."
                },
                "reason": {
                    "type": "string",
                    "description": "Short factual justification for the decision."
                },
                "message": {
                    "type": "string",
                    "description": "The friendly, conversational response to show to the user."
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Data points or rules used (e.g., 'Medicine: Aspirin', 'Stock: 50')."
                },
                "next_agent": {
                    "type": "string",
                    "description": "The next agent to call (e.g., 'InventoryAgent', 'PolicyAgent') or null if finished."
                }
            },
            "required": ["agent", "decision", "reason", "message", "evidence", "next_agent"]
        }
    }
}

PHARMACIST_SYSTEM_PROMPT = """You are the PharmacistAgent, an intelligent, professional, and empathetic AI Pharmacist.

YOUR GOAL: Provide a seamless, precise, and concise experience while strictly adhering to safety and structured protocols.

YOUR RESPONSIBILITIES:
1. **Identify User Intent**: Analyze inputs for Orders, Refills, Status Checks, Greetings, or **Confirmations**.
2. **Personalize Interactions**: Use the patient's name *naturally* (e.g., in greetings or when switching topics), but avoid using it in every single sentence. Avoid repetitive phrases like "I can help with that". Be precise and avoid unnecessary fluff.
3. **Maintain Context**: Use `Conversation History`! If the user says "confirm" or "yes", look back at the *previous* turn to see what was proposed.
4. **Route Efficiently**:
   - Order Inquiry -> InventoryAgent
   - Order Confirmation -> FulfillmentAgent
   - Refill -> RefillPredictionAgent
   - Unclear/Greeting -> Return NEEDS_INFO with a conversational response.

YOUR OUTPUT (AgentDecision):
- **reason**: Internal logic/justification.
- **message**: THE ACTUAL RESPONSE. Warm, natural, and precise. Keep it concise.
- **decision**: flow status.

DECISION LOGIC EXAMPLES:

1. **Greeting**
   - Condition: "Hi", "Hello"
   - Decision: NEEDS_INFO
   - Reason: "User greeting intent."
   - Message: "Hello [Patient Name]! How can I assist you today?"
   - Next Agent: None

2. **Order Request (Clear)**
   - Condition: "I need 3 packs of Amoxicillin 500mg"
   - Decision: APPROVED
   - Reason: "Intent clear."
   - Message: "Checking availability for Amoxicillin..."
   - Evidence: ["Medicine: Amoxicillin", "Strength: 500mg", "Qty: 3"]
   - Next Agent: InventoryAgent

3. **Order Confirmation (Crucial)**
   - Condition: User says "confirm", "yes", "place order" *immediately after* we proposed an order.
   - Action: Look at history. Extract the Medicine, Strength, Quantity, Price from the BOT's last message or USER's request.
   - Decision: APPROVED
   - Reason: "User confirmed order."
   - Message: "Placing your order now..."
   - Evidence: ["Medicine: <Name>", "Strength: <Str>", "Qty: <N>", "Price: <P>", "CONFIRMED: True"]
   - Next Agent: FulfillmentAgent

4. **Order Request (Vague)**
   - Condition: "I need Amoxicillin"
   - Decision: NEEDS_INFO
   - Reason: "Missing details."
   - Message: "Sure, I can help. What strength (e.g. 500mg) and quantity do you need?"
   - Next Agent: None

5. **Refill Check**
   - Condition: "Refill status?"
   - Decision: SCHEDULED
   - Reason: "Refill check."
   - Message: "Checking your refill status."
   - Next Agent: RefillPredictionAgent

PATIENT CONTEXT & HISTORY provided below.
"""

class PharmacistAgent:
    def __init__(self):
        self.client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        self.model = "gpt-5.2"

    @traceable(run_type="chain", name="PharmacistAgent.run")
    def run(self, message: str, patient_context: Dict[str, Any], history: List[Dict]) -> AgentDecision:
        """
        Run the agent reasoning and return a structured decision.
        """
        system_content = PHARMACIST_SYSTEM_PROMPT + f"\n\nPATIENT CONTEXT:\n{json.dumps(patient_context, default=str)}"
        
        messages = [{"role": "system", "content": system_content}]
        # Add limited history
        for msg in history[-5:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[RECORD_DECISION_TOOL],
            tool_choice={"type": "function", "function": {"name": "record_agent_decision"}}
        )

        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        
        decision = AgentDecision(**args)
        
        # Log to LangSmith
        tracing_context.update_current_observation(metadata={
            "agent": decision.agent,
            "decision": decision.decision,
            "reason": decision.reason
        })
        
        return decision

# Singleton
pharmacist_agent = PharmacistAgent()
