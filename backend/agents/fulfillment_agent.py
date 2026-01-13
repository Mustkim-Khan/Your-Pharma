"""
Inventory & Fulfillment Agent
Model: gpt-4o-mini
Purpose: Execute orders, update inventory, trigger warehouse webhooks, send notifications, generate receipts
"""
import uuid
import httpx
from datetime import datetime
from typing import List, Optional, Dict
import os

from models.schemas import Order, OrderItem, OrderStatus, OrderStatusUpdate
from utils.tracing_utils import traceable, tracing_context


class AgentEvent:
    """Represents a single agent action in the order timeline"""
    def __init__(self, agent_name: str, action: str, description: str, status: str = "completed"):
        self.agent_name = agent_name
        self.action = action
        self.description = description
        self.status = status  # completed, current, pending, blocked
        self.timestamp = datetime.now()
    
    def to_dict(self):
        return {
            "agent_name": self.agent_name,
            "action": self.action,
            "description": self.description,
            "status": self.status,
            "timestamp": self.timestamp.isoformat()
        }


class FulfillmentAgent:
    def __init__(self):
        self.orders = {}  # In-memory order storage
        self.order_history = {}  # Order status history
        self.order_events = {}  # Store agent events per order
    
    def add_event(self, order_id: str, agent_name: str, action: str, description: str, status: str = "completed"):
        """Add an agent event to the order timeline"""
        if order_id not in self.order_events:
            self.order_events[order_id] = []
        
        event = AgentEvent(agent_name, action, description, status)
        self.order_events[order_id].append(event)
        return event
    
    def get_events(self, order_id: str) -> List[Dict]:
        """Get all events for an order"""
        if order_id not in self.order_events:
            return []
        return [event.to_dict() for event in self.order_events[order_id]]
    
    @traceable(run_type="chain", name="FulfillmentAgent.create_order")
    def create_order(
        self,
        patient_id: str,
        patient_name: str,
        patient_email: str,
        patient_phone: str,
        items: List[OrderItem],
        conversation_history: List[dict] = None
    ) -> Order:
        """
        Create a new order with proper pricing and tracking
        """
        order_id = f"ORD-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        
        # Calculate total with realistic prices based on medicine type
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
        
        for item in items:
            if item.unit_price == 0:
                # Look up price or use default
                base_price = price_map.get(item.medicine_name, 0.50)
                item.unit_price = base_price
        
        total_amount = sum(item.unit_price * item.quantity for item in items)
        
        order = Order(
            order_id=order_id,
            patient_id=patient_id,
            patient_name=patient_name,
            patient_email=patient_email,
            patient_phone=patient_phone,
            items=items,
            total_amount=round(total_amount, 2),
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.orders[order_id] = order
        self.order_history[order_id] = [
            OrderStatusUpdate(
                order_id=order_id,
                status=OrderStatus.PENDING,
                message=f"Order created for {patient_name}",
                timestamp=datetime.now()
            )
        ]
        
        # Add initial event
        self.add_event(
            order_id, 
            "Conversational Ordering Agent", 
            "Order Requested", 
            "Order initiated via conversation",
            "completed"
        )
        
        tracing_context.update_current_observation(
            metadata={
                "order_id": order_id, 
                "total_amount": total_amount,
                "patient_name": patient_name,
                "items_count": len(items)
            }
        )
        
        return order
    
    def record_safety_validation(self, order_id: str, decision: str, reasons: List[str] = None):
        """Record safety agent validation event"""
        if decision == "APPROVE":
            self.add_event(
                order_id,
                "Safety & Policy Agent",
                "AI Safety Validation",
                "Prescription verified and approved",
                "completed"
            )
        elif decision == "CONDITIONAL":
            self.add_event(
                order_id,
                "Safety & Policy Agent", 
                "AI Safety Validation",
                f"Conditional approval: {reasons[0] if reasons else 'Conditions apply'}",
                "completed"
            )
        else:  # REJECT/BLOCKED
            self.add_event(
                order_id,
                "Safety & Policy Agent",
                "Blocked by AI",
                f"Blocked: {reasons[0] if reasons else 'Safety check failed'}",
                "blocked"
            )
    
    def record_order_confirmed(self, order_id: str):
        """Record order confirmation event"""
        self.add_event(
            order_id,
            "Safety & Policy Agent",
            "AI Order Confirmed",
            "AI validated and confirmed order",
            "completed"
        )
    
    def record_inventory_updated(self, order_id: str, quantity: int):
        """Record inventory update event"""
        self.add_event(
            order_id,
            "Inventory & Fulfillment Agent",
            "Inventory Updated",
            f"Stock reduced by {quantity} units",
            "completed"
        )
    
    def record_fulfillment_initiated(self, order_id: str):
        """Record fulfillment initiation event"""
        self.add_event(
            order_id,
            "Inventory & Fulfillment Agent",
            "Fulfillment Initiated",
            "Warehouse notified for fulfillment",
            "completed"
        )
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_all_orders(self) -> List[Order]:
        """Get all orders"""
        return list(self.orders.values())
    
    def get_order_with_events(self, order_id: str) -> Optional[Dict]:
        """Get order with its timeline events"""
        order = self.get_order(order_id)
        if not order:
            return None
        
        order_dict = order.model_dump() if hasattr(order, 'model_dump') else order.__dict__
        order_dict['timeline'] = self.get_events(order_id)
        return order_dict
    
    def get_all_orders_with_events(self) -> List[Dict]:
        """Get all orders with their timeline events"""
        result = []
        for order in self.orders.values():
            order_dict = order.model_dump() if hasattr(order, 'model_dump') else {
                "order_id": order.order_id,
                "patient_id": order.patient_id,
                "patient_name": order.patient_name,
                "patient_email": order.patient_email,
                "patient_phone": order.patient_phone,
                "items": [{"medicine_id": i.medicine_id, "medicine_name": i.medicine_name, 
                           "strength": i.strength, "quantity": i.quantity, "unit_price": i.unit_price} 
                          for i in order.items],
                "total_amount": order.total_amount,
                "status": order.status.value if hasattr(order.status, 'value') else order.status,
                "created_at": order.created_at.isoformat(),
                "updated_at": order.updated_at.isoformat()
            }
            order_dict['timeline'] = self.get_events(order.order_id)
            result.append(order_dict)
        return result
    
    @traceable(run_type="chain", name="FulfillmentAgent.update_order_status")
    def update_order_status(self, order_id: str, status: OrderStatus, message: str) -> Optional[Order]:
        """Update order status with tracking"""
        order = self.orders.get(order_id)
        if not order:
            return None
        
        order.status = status
        order.updated_at = datetime.now()
        
        # Add to history
        if order_id not in self.order_history:
            self.order_history[order_id] = []
            
        self.order_history[order_id].append(OrderStatusUpdate(
            order_id=order_id,
            status=status,
            message=message,
            timestamp=datetime.now()
        ))
        
        tracing_context.update_current_observation(
            metadata={"order_id": order_id, "new_status": status.value}
        )
        
        return order
    
    async def trigger_warehouse_webhook(self, order: Order):
        """Trigger warehouse fulfillment webhook"""
        webhook_url = os.getenv("WAREHOUSE_WEBHOOK_URL", "http://localhost:8000/api/webhook/warehouse")
        
        payload = {
            "order_id": order.order_id,
            "items": [
                {
                    "medicine_id": item.medicine_id,
                    "medicine_name": item.medicine_name,
                    "strength": item.strength,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price
                }
                for item in order.items
            ],
            "delivery_type": "HOME_DELIVERY",
            "patient_name": order.patient_name,
            "patient_email": order.patient_email,
            "patient_phone": order.patient_phone,
            "patient_address": "123 Main St, City, State 12345",
            "priority": "NORMAL"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=payload, timeout=10.0)
                if response.status_code == 200:
                    print(f"✅ Warehouse webhook triggered successfully for order {order.order_id}")
                    self.update_order_status(
                        order.order_id, 
                        OrderStatus.PROCESSING, 
                        "Order sent to warehouse for fulfillment"
                    )
                    # Add dispatched event
                    self.add_event(
                        order.order_id,
                        "System",
                        "Dispatched",
                        "Package dispatched from warehouse",
                        "completed"
                    )
                else:
                    print(f"⚠️ Warehouse webhook returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Failed to trigger warehouse webhook: {str(e)}")
    
    def generate_receipt(self, order: Order) -> dict:
        """Generate detailed receipt for an order"""
        receipt_number = f"RCP-{order.order_id[-6:]}"
        
        items_detail = []
        for item in order.items:
            item_total = round(item.unit_price * item.quantity, 2)
            items_detail.append({
                "medicine": item.medicine_name,
                "strength": item.strength,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "total": item_total
            })
        
        subtotal = order.total_amount
        tax = round(subtotal * 0.05, 2)
        delivery_fee = 2.00
        grand_total = round(subtotal + tax + delivery_fee, 2)
        
        return {
            "receipt_number": receipt_number,
            "order_id": order.order_id,
            "order_date": order.created_at.strftime("%Y-%m-%d %H:%M"),
            "patient_name": order.patient_name,
            "patient_id": order.patient_id,
            "patient_email": order.patient_email,
            "patient_phone": order.patient_phone,
            "items": items_detail,
            "subtotal": subtotal,
            "tax": tax,
            "delivery_fee": delivery_fee,
            "grand_total": grand_total,
            "payment_status": "Paid",
            "delivery_status": "Preparing",
            "estimated_delivery": "1-2 business days",
            "issued_at": datetime.now().isoformat(),
            "thank_you_message": f"Thank you, {order.patient_name}! Your order is being prepared. You'll receive updates via email."
        }
    
    def get_order_history(self, order_id: str) -> List[OrderStatusUpdate]:
        """Get status history for an order"""
        return self.order_history.get(order_id, [])
    
    def get_order_summary(self, order: Order) -> str:
        """Generate a human-readable order summary"""
        items_list = "\n".join([
            f"  • {item.medicine_name} {item.strength} x{item.quantity} - ${item.unit_price * item.quantity:.2f}"
            for item in order.items
        ])
        
        summary = f"""
📋 **Order Summary**
━━━━━━━━━━━━━━━━━━━━━━
**Order ID:** {order.order_id}
**Patient:** {order.patient_name}
**Date:** {order.created_at.strftime("%Y-%m-%d %H:%M")}

**Items:**
{items_list}

**Subtotal:** ${order.total_amount:.2f}
**Tax (5%):** ${order.total_amount * 0.05:.2f}
**Delivery:** $2.00
━━━━━━━━━━━━━━━━━━━━━━
**Total:** ${order.total_amount * 1.05 + 2:.2f}

**Status:** {order.status.value}
"""
        return summary


# Singleton instance
fulfillment_agent = FulfillmentAgent()
