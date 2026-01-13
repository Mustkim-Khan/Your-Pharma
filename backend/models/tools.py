"""
OpenAI Tool Definitions for Agentic AI Pharmacy
"""

EXTRACT_ORDER_DETAILS_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_order_details",
        "description": "Extract medicine, dosage, quantity, and frequency from a user's request. Used when the user expresses an intent to order or buy medication.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "medicine": {
                                "type": "string",
                                "description": "The name of the medicine (e.g., 'Metformin', 'Paracetamol')"
                            },
                            "dosage": {
                                "type": "string",
                                "description": "The strength or dosage (e.g., '500mg', '10mg'). Leave empty if not specified."
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "The number of units requested (e.g., 30, 60). Set to 0 if not specified."
                            },
                            "frequency": {
                                "type": "string",
                                "description": "How often the user takes it (e.g., 'once daily'). Leave empty if not specified."
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score between 0.0 and 1.0"
                            },
                            "raw_text": {
                                "type": "string",
                                "description": "The specific text segment referring to this medicine"
                            }
                        },
                        "required": ["medicine", "quantity"]
                    }
                },
                "needs_clarification": {
                    "type": "boolean",
                    "description": "True if critical information (like medicine name) is missing or ambiguous."
                },
                "clarification_message": {
                    "type": "string",
                    "description": "A polite question to ask the user if clarification is needed."
                }
            },
            "required": ["entities", "needs_clarification"]
        }
    }
}

CHECK_REFILL_STATUS_TOOL = {
    "type": "function",
    "function": {
        "name": "check_refill_status",
        "description": "Check if valid refills are available for the patient. Used when user asks about refills, checking status, or generic 'do I need anything'.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

PLACE_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "place_order",
        "description": "Proceed to place an order for the extracted medicines. Used after entities are successfully extracted and confirmed by context.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

CONFIRM_LAST_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "confirm_last_order",
        "description": "Confirm the pending order. Used when user says 'yes', 'confirm', 'go ahead', etc.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

CANCEL_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "cancel_order",
        "description": "Cancel the pending order. Used when user says 'no', 'cancel', 'stop'.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

CHECK_ORDER_STATUS_TOOL = {
    "type": "function",
    "function": {
        "name": "check_order_status",
        "description": "Check the status of recent orders. Used when user asks 'where is my order' or 'status'.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

ALL_TOOLS = [
    EXTRACT_ORDER_DETAILS_TOOL,
    CHECK_REFILL_STATUS_TOOL,
    PLACE_ORDER_TOOL,
    CONFIRM_LAST_ORDER_TOOL,
    CANCEL_ORDER_TOOL,
    CHECK_ORDER_STATUS_TOOL
]