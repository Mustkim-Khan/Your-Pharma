"""
Safety & Prescription Policy Agent
Model: gpt-4o-mini
Purpose: High-stakes safety reasoning, prescription validation, policy enforcement with conversation context
"""
from typing import List
from openai import OpenAI
import os
import json

from models.schemas import ExtractedEntity, Medicine, SafetyCheckResult, SafetyDecision
from utils.tracing_utils import traceable, wrap_openai, tracing_context


SAFETY_SYSTEM_PROMPT = """You are a pharmaceutical safety AI agent responsible for evaluating medication orders.

## YOUR ROLE
Evaluate medication orders for safety, prescription requirements, and policy compliance.

## RULES TO ENFORCE
1. **Prescription Validation**: Check if medicines require a valid prescription
2. **Controlled Substances**: Flag controlled substances for special handling
3. **Quantity Limits**: Enforce maximum quantity per order limits
4. **Stock Availability**: Check if requested quantities are in stock
5. **Discontinued Medicines**: Block orders for discontinued medicines
6. **Drug Interactions**: Flag potential issues (based on patient history if available)

## OUTPUT FORMAT (JSON only)
{
  "decision": "APPROVE" | "CONDITIONAL" | "REJECT",
  "reasons": ["List of reasons for the decision"],
  "allowed_quantity": null or number (if quantity was adjusted),
  "requires_followup": true/false,
  "requires_prescription": true/false,
  "blocked_items": ["List of medicine names that cannot be fulfilled"]
}

## DECISION GUIDELINES
- APPROVE: All checks passed, safe to proceed
- CONDITIONAL: Can proceed with conditions (e.g., pending prescription, reduced quantity)
- REJECT: Cannot fulfill order (e.g., all items discontinued, out of stock, safety concern)

Be thorough but efficient. Patient safety is paramount."""


class SafetyAgent:
    def __init__(self):
        self.client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        self.model = "gpt-4o-mini"
    
    @traceable(run_type="chain", name="SafetyAgent.evaluate")
    def evaluate(
        self, 
        entities: List[ExtractedEntity], 
        matched_medicines: List[Medicine],
        has_prescription: bool = False,
        patient_context: dict = None,
        conversation_history: List[dict] = None
    ) -> SafetyCheckResult:
        """
        Evaluate safety and prescription requirements with explicit decision tracing
        """
        # Log input context
        tracing_context.update_current_observation(
            metadata={
                "input_entities_count": len(entities),
                "has_prescription": has_prescription,
                "patient_name": patient_context.get("patient_name") if patient_context else None
            }
        )
        
        # Perform evaluation
        result = self._perform_safety_checks(entities, matched_medicines, has_prescription)
        
        # Explicitly trace the decision outcome as a distinct span for visibility
        self._trace_decision(result)
        
        return result

    def _perform_safety_checks(self, entities, matched_medicines, has_prescription) -> SafetyCheckResult:
        """Internal logic for safety checks"""
        reasons = []
        blocked_items = []
        requires_prescription = False
        allowed_quantity = None
        requires_followup = False
        
        for i, medicine in enumerate(matched_medicines):
            entity = entities[i] if i < len(entities) else None
            requested_qty = entity.quantity if entity and entity.quantity > 0 else 30
            
            # Check if medicine is discontinued
            if medicine.discontinued:
                blocked_items.append(medicine.medicine_name)
                reasons.append(f"{medicine.medicine_name} has been discontinued and is no longer available.")
                continue
            
            # Check if prescription is required
            if medicine.prescription_required:
                requires_prescription = True
                if not has_prescription:
                    reasons.append(f"{medicine.medicine_name} requires a valid prescription.")
            
            # Check if controlled substance
            if medicine.controlled_substance:
                reasons.append(f"{medicine.medicine_name} is a controlled substance. Special handling required.")
                requires_followup = True
            
            # Check stock availability
            if medicine.stock_level == 0:
                blocked_items.append(medicine.medicine_name)
                reasons.append(f"{medicine.medicine_name} is currently out of stock.")
                continue
            
            if medicine.stock_level < requested_qty:
                allowed_quantity = min(medicine.stock_level, medicine.max_quantity_per_order)
                reasons.append(f"Limited stock available for {medicine.medicine_name}. Maximum quantity: {allowed_quantity}")
            
            # Check max quantity limits
            if requested_qty > medicine.max_quantity_per_order:
                if not allowed_quantity:
                    allowed_quantity = medicine.max_quantity_per_order
                reasons.append(f"Maximum quantity per order for {medicine.medicine_name} is {medicine.max_quantity_per_order}")
        
        # Determine decision
        if blocked_items and len(blocked_items) == len(matched_medicines):
            decision = SafetyDecision.REJECT
        elif blocked_items or (requires_prescription and not has_prescription):
            decision = SafetyDecision.CONDITIONAL
        elif requires_followup or allowed_quantity:
            decision = SafetyDecision.CONDITIONAL
        else:
            decision = SafetyDecision.APPROVE
            if not reasons:
                reasons.append("All safety checks passed.")
        
        return SafetyCheckResult(
            decision=decision,
            reasons=reasons,
            allowed_quantity=allowed_quantity,
            requires_followup=requires_followup,
            requires_prescription=requires_prescription,
            blocked_items=blocked_items
        )

    @traceable(run_type="chain", name="Safety: Make Decision")
    def _trace_decision(self, result: SafetyCheckResult):
        """
        Helper method to create a explicit span for the final decision.
        This makes the 'reasoning' clearly visible in the trace tree.
        """
        tracing_context.update_current_observation(metadata={
            "outcome": str(result.decision),
            "reasons": result.reasons,
            "blocked_items": result.blocked_items
        })
        # If rejected, we want to make it very obvious in the trace name or metadata
        if result.decision == SafetyDecision.REJECT:
             tracing_context.update_current_observation(metadata={"critical_alert": "ORDER REJECTED"})


# Singleton instance
safety_agent = SafetyAgent()
