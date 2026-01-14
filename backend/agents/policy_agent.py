"""
Policy Agent (formerly Safety Agent)
Model: gpt-5.2
Purpose: Enforces prescription rules, quantity limits, and safety warnings.
"""
import json
import os
from openai import OpenAI
from utils.tracing_utils import traceable, wrap_openai, tracing_context
from models.schemas import AgentDecision

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

POLICY_SYSTEM_PROMPT = """You are the PolicyAgent.

YOUR RESPONSIBILITIES:
1. Enforce Prescription Rules: Does the medicine require a prescription? Does the patient have one?
2. Enforce Quantity Limits: Is the requested quantity safe (max 30 usually)?
3. Check Drug Interactions (mock logic via rules): Warn if dangerous.

DECISION LOGIC:
- If Prescription Required AND No Prescription:
  - Decision: REJECTED
  - Reason: "Prescription required but not provided."
  - Evidence: ["Medicine: <Name>", "Requires Prescription: Yes"]
  - Next Agent: None
- If Quantity > Max:
  - Decision: REJECTED (or APPROVED with warning/cap, but let's say REJECTED for strictness or NEEDS_INFO)
  - Reason: "Quantity exceeds safe limit."
  - Evidence: ["Requested: <Qty>", "Max: 30"]
  - Next Agent: None
- If Safe & Valid:
  - Decision: APPROVED
  - Reason: "Safety checks passed."
  - Evidence: ["Prescription: <ID/None>", "Quantity: Safe"]
  - Next Agent: FulfillmentAgent

System Context provides the medicine details and patient details.
"""

class PolicyAgent:
    def __init__(self):
        self.client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        self.model = "gpt-5.2"

    @traceable(run_type="chain", name="PolicyAgent.run")
    def run(self, medicine_name: str, quantity: int, prescription_required: bool, patient_context: dict) -> AgentDecision:
        context = {
            "medicine": medicine_name,
            "quantity": quantity,
            "prescription_required": prescription_required,
            "patient_has_prescription": True, # Mocking true for demo unless specified
            "max_allowed": 30
        }
        
        messages = [
            {"role": "system", "content": POLICY_SYSTEM_PROMPT + f"\n\nCONTEXT:\n{json.dumps(context)}"},
            {"role": "user", "content": f"Validate order for {quantity} {medicine_name}."}
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

policy_agent = PolicyAgent()
