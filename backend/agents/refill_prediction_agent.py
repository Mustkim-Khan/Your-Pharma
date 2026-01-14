"""
Refill Prediction Agent
Model: gpt-5-mini
Purpose: Predicts next refill date based on purchase history.
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

REFILL_SYSTEM_PROMPT = """You are the RefillPredictionAgent.

YOUR RESPONSIBILITIES:
1. Analyze patient purchase history.
2. Predict when the next refill is needed (standard supply is 30 days).
3. Recommend action.

DECISION LOGIC:
- If days_since_last_purchase > 25 (Assuming 30 supply):
  - Decision: SCHEDULED
  - Reason: "Refill due soon."
  - Evidence: ["Last Purchase: <Date>", "Days Elapsed: <N>", "Supply: 30 days"]
  - Next Agent: None (Just informs user)
- If days_since_last_purchase < 20:
  - Decision: REJECTED (for auto-refill) - Actually, strictly user asked for refill CHECK, so:
  - Decision: REJECTED
  - Reason: "Refill not due yet."
  - Evidence: ["Last Purchase: <Date>", "Days Remaining: <N>"]
  - Next Agent: None

"""

class RefillPredictionAgent:
    def __init__(self):
        self.client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        self.model = "gpt-5.2"

    @traceable(run_type="chain", name="RefillPredictionAgent.run")
    def run(self, patient_history: list) -> AgentDecision:
        # Mock logic inside LLM via prompt
        messages = [
            {"role": "system", "content": REFILL_SYSTEM_PROMPT + f"\n\nPATIENT HISTORY:\n{json.dumps(patient_history, default=str)}"},
            {"role": "user", "content": "Check refill eligibility."}
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

refill_prediction_agent = RefillPredictionAgent()
