"""
Inventory Agent
Model: gpt-4o-mini
Purpose: Checks medicine availability and stock levels.
"""
import json
import os
from typing import Dict, Any, List
from openai import OpenAI
from utils.tracing_utils import traceable, wrap_openai, tracing_context
from models.schemas import AgentDecision
from services.data_service import data_service

RECORD_DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "record_agent_decision",
        "description": "Record the agent's decision.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent": {"type": "string"},
                "decision": {"type": "string", "enum": ["APPROVED", "REJECTED", "NEEDS_INFO", "SCHEDULED"]},
                "reason": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "next_agent": {"type": "string"}
            },
            "required": ["agent", "decision", "reason", "evidence", "next_agent"]
        }
    }
}

INVENTORY_SYSTEM_PROMPT = """You are the InventoryAgent.

YOUR RESPONSIBILITIES:
1. Check if the requested medication exists in the catalog.
2. Check if the requested quantity is available in stock.
3. Provide pricing information in evidence.

DECISION LOGIC:
- If medicine found AND stock >= requested_quantity:
  - Decision: APPROVED
  - Reason: "Sufficient stock available."
  - Evidence: ["Stock: <Available>", "Price: <Unit Price>", "Requested: <Qty>"]
  - Next Agent: PolicyAgent
- If medicine not found:
  - Decision: REJECTED
  - Reason: "Medication not found in inventory."
  - Next Agent: None
- If stock low:
  - Decision: REJECTED (or NEEDS_INFO if partial allowed, but stick to REJECTED for strictness)
  - Reason: "Insufficient stock."
  - Evidence: ["Stock: <Available>", "Requested: <Qty>"]
  - Next Agent: None

SYSTEM CONTEXT contains the 'current_request' with medicine details.
"""

class InventoryAgent:
    def __init__(self):
        self.client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        self.model = "gpt-5-mini"
    
    @traceable(run_type="chain", name="InventoryAgent.run")
    def run(self, medicine_name: str, quantity: int) -> AgentDecision:
        # Search DB
        medicines = data_service.search_medicine(medicine_name)
        
        request_context = {
            "requested_medicine": medicine_name,
            "requested_quantity": quantity,
            "search_results": [m.model_dump() for m in medicines] if medicines else []
        }
        
        messages = [
            {"role": "system", "content": INVENTORY_SYSTEM_PROMPT + f"\n\nCONTEXT:\n{json.dumps(request_context)}"},
            {"role": "user", "content": f"Check inventory for {quantity} units of {medicine_name}."}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[RECORD_DECISION_TOOL],
            tool_choice={"type": "function", "function": {"name": "record_agent_decision"}}
        )

        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        return AgentDecision(**args)

inventory_agent = InventoryAgent()
