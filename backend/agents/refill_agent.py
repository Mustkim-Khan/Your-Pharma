"""
Predictive Refill Intelligence Agent
Model: gpt-5.2
Purpose: Proactive refill management, stock prediction, and adherence monitoring
"""
from typing import List
from openai import OpenAI
import os
import json
from datetime import datetime

from models.schemas import RefillPrediction, Medicine
from utils.tracing_utils import traceable, wrap_openai, tracing_context


REFILL_SYSTEM_PROMPT = """You are an AI Refill Intelligence Agent.
Your goal is to predict refill needs and adherence issues based on patient medication history.

## ANALYSIS RULES
1. Calculate days remaining for each medication
2. If remaining days <= 7: Suggest REMIND (Refill Soon)
3. If remaining days <= 0: Suggest REMIND (Overdue)
4. If adherence < 80%: Suggest REMIND + educational note
5. If medication is 'PRN' (as needed): Don't predict refills unless frequency is high

## OUTPUT FORMAT
Return a JSON array of predictions.
"""


class RefillAgent:
    def __init__(self):
        self.client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        self.model = "gpt-5.2"
    
    @traceable(run_type="chain", name="RefillAgent.predict")
    def predict(
        self,
        patient_id: str,
        patient_name: str,
        medication_history: List[dict],
        conversation_history: List[dict] = None
    ) -> List[RefillPrediction]:
        """
        Predict refill needs for a patient
        """
        tracing_context.update_current_observation(metadata={
            "patient_info": f"{patient_name} ({patient_id})",
            "med_count": len(medication_history)
        })
        
        # Prepare context for LLM
        history_text = json.dumps(medication_history, indent=2, default=str)
        
        messages = [
            {"role": "system", "content": REFILL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze refill needs for patient {patient_name} based on this history:\n{history_text}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2, # Low temp for analytical task
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            # Parse response
            try:
                data = json.loads(content)
                predictions_data = data.get("predictions", [])
                
                predictions = []
                for p in predictions_data:
                    predictions.append(RefillPrediction(
                        medicine=p.get("medicine"),
                        days_remaining=p.get("days_remaining", 0),
                        action=p.get("action", "NONE"),
                        confidence=p.get("confidence", 0.0),
                        reason=p.get("reason", "")
                    ))
                
                # If LLM fails to structure, do fallback calculation
                if not predictions:
                    predictions = self._fallback_calculation(medication_history)
                
                tracing_context.update_current_observation(metadata={
                    "predictions_count": len(predictions)
                })
                
                return predictions
                
            except json.JSONDecodeError:
                return self._fallback_calculation(medication_history)
                
        except Exception as e:
            tracing_context.update_current_observation(metadata={"error": str(e)})
            return self._fallback_calculation(medication_history)
    
    def _fallback_calculation(self, history: List[dict]) -> List[RefillPrediction]:
        """Simple deterministic fallback if AI fails"""
        predictions = []
        for med in history:
            # Mock calculation logic
            days_supply = med.get('supply_days', 30)
            purchase_date = datetime.strptime(med.get('purchase_date'), "%Y-%m-%d") if isinstance(med.get('purchase_date'), str) else med.get('purchase_date')
            
            if purchase_date:
                days_since = (datetime.now() - purchase_date).days
                days_remaining = max(0, days_supply - days_since)
                
                action = "NONE"
                if days_remaining <= 5:
                    action = "REMIND"
                
                predictions.append(RefillPrediction(
                    medicine=med.get('medicine_name', 'Unknown'),
                    days_remaining=days_remaining,
                    action=action,
                    confidence=1.0,
                    reason="Deterministic calculation"
                ))
        return predictions

    @traceable(run_type="chain", name="RefillAgent.get_all_patient_refills")
    def get_all_patient_refills(self, data_service, current_date: datetime) -> List[RefillPrediction]:
        """
        Get proactive refills for all patients (admin view)
        """
        # In a real system this would key off database
        # For demo, just check a few key patients
        patients = data_service.get_all_patients()
        all_predictions = []
        
        for patient in patients:
            meds = data_service.get_medicines_needing_refill(patient.patient_id, current_date)
            if meds:
                preds = self.predict(patient.patient_id, patient.patient_name, meds)
                all_predictions.extend([p for p in preds if p.action != "NONE"])
        
        tracing_context.update_current_observation(metadata={"total_predictions": len(all_predictions)})
        return all_predictions


# Singleton instance
refill_agent = RefillAgent()
