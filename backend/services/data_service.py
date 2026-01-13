"""
Data service for accessing medicine and order data from CSV files
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import os

from models.schemas import Medicine, Patient, Order, OrderItem, OrderStatus


class DataService:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self._medicines_df = None
        self._orders_df = None
        self._load_data()
    
    def _load_data(self):
        """Load CSV data into DataFrames"""
        medicine_path = self.data_dir / "medicine_master.csv"
        order_path = self.data_dir / "order_history.csv"
        
        if medicine_path.exists():
            self._medicines_df = pd.read_csv(medicine_path)
            # Convert boolean columns
            bool_cols = ['prescription_required', 'discontinued', 'controlled_substance']
            for col in bool_cols:
                if col in self._medicines_df.columns:
                    self._medicines_df[col] = self._medicines_df[col].astype(str).str.lower() == 'true'
        
        if order_path.exists():
            self._orders_df = pd.read_csv(order_path)
    
    def get_all_medicines(self) -> List[Medicine]:
        """Get all medicines from the master data"""
        if self._medicines_df is None:
            return []
        
        medicines = []
        for _, row in self._medicines_df.iterrows():
            medicines.append(Medicine(
                medicine_id=row['medicine_id'],
                medicine_name=row['medicine_name'],
                strength=row['strength'],
                form=row['form'],
                stock_level=int(row['stock_level']),
                prescription_required=bool(row['prescription_required']),
                category=row['category'],
                discontinued=bool(row['discontinued']),
                max_quantity_per_order=int(row.get('max_quantity_per_order', 30)),
                controlled_substance=bool(row.get('controlled_substance', False))
            ))
        return medicines
    
    def get_medicine_by_id(self, medicine_id: str) -> Optional[Medicine]:
        """Get a specific medicine by ID"""
        if self._medicines_df is None:
            return None
        
        row = self._medicines_df[self._medicines_df['medicine_id'] == medicine_id]
        if row.empty:
            return None
        
        row = row.iloc[0]
        return Medicine(
            medicine_id=row['medicine_id'],
            medicine_name=row['medicine_name'],
            strength=row['strength'],
            form=row['form'],
            stock_level=int(row['stock_level']),
            prescription_required=bool(row['prescription_required']),
            category=row['category'],
            discontinued=bool(row['discontinued']),
            max_quantity_per_order=int(row.get('max_quantity_per_order', 30)),
            controlled_substance=bool(row.get('controlled_substance', False))
        )
    
    def search_medicine(self, query: str) -> List[Medicine]:
        """Search medicines by name (fuzzy match)"""
        if self._medicines_df is None:
            return []
        
        query_lower = query.lower()
        matches = self._medicines_df[
            self._medicines_df['medicine_name'].str.lower().str.contains(query_lower, na=False)
        ]
        
        medicines = []
        for _, row in matches.iterrows():
            medicines.append(Medicine(
                medicine_id=row['medicine_id'],
                medicine_name=row['medicine_name'],
                strength=row['strength'],
                form=row['form'],
                stock_level=int(row['stock_level']),
                prescription_required=bool(row['prescription_required']),
                category=row['category'],
                discontinued=bool(row['discontinued']),
                max_quantity_per_order=int(row.get('max_quantity_per_order', 30)),
                controlled_substance=bool(row.get('controlled_substance', False))
            ))
        return medicines
    
    def get_all_patients(self) -> List[Patient]:
        """Get unique patients from order history"""
        if self._orders_df is None:
            return []
        
        unique_patients = self._orders_df.drop_duplicates(subset=['patient_id'])
        patients = []
        for _, row in unique_patients.iterrows():
            patients.append(Patient(
                patient_id=row['patient_id'],
                patient_name=row['patient_name'],
                patient_email=row['patient_email'],
                patient_phone=row['patient_phone']
            ))
        return patients
    
    def get_patient_by_id(self, patient_id: str) -> Optional[Patient]:
        """Get a specific patient by ID"""
        if self._orders_df is None:
            return None
        
        row = self._orders_df[self._orders_df['patient_id'] == patient_id]
        if row.empty:
            return None
        
        row = row.iloc[0]
        return Patient(
            patient_id=row['patient_id'],
            patient_name=row['patient_name'],
            patient_email=row['patient_email'],
            patient_phone=row['patient_phone']
        )
    
    def get_patient_order_history(self, patient_id: str) -> pd.DataFrame:
        """Get order history for a specific patient"""
        if self._orders_df is None:
            return pd.DataFrame()
        
        return self._orders_df[self._orders_df['patient_id'] == patient_id].copy()
    
    def update_stock(self, medicine_id: str, quantity_sold: int, auto_reset_quantity: int = 100) -> bool:
        """Update stock level after a sale. Auto-resets to default when stock hits zero."""
        if self._medicines_df is None:
            return False
        
        idx = self._medicines_df[self._medicines_df['medicine_id'] == medicine_id].index
        if len(idx) == 0:
            return False
        
        current_stock = self._medicines_df.loc[idx[0], 'stock_level']
        new_stock = max(0, current_stock - quantity_sold)
        
        # Auto-reset when stock becomes zero
        if new_stock == 0:
            new_stock = auto_reset_quantity
            print(f"[Inventory Agent] Stock for {medicine_id} depleted. Auto-resetting to {auto_reset_quantity} units.")
        
        self._medicines_df.loc[idx[0], 'stock_level'] = new_stock
        
        # Persist to CSV
        self._medicines_df.to_csv(self.data_dir / "medicine_master.csv", index=False)
        return True
    
    def add_order(self, order_data: dict) -> bool:
        """Add a new order to the history"""
        if self._orders_df is None:
            self._orders_df = pd.DataFrame()
        
        new_row = pd.DataFrame([order_data])
        self._orders_df = pd.concat([self._orders_df, new_row], ignore_index=True)
        
        # Persist to CSV
        self._orders_df.to_csv(self.data_dir / "order_history.csv", index=False)
        return True
    
    def get_inventory_stats(self) -> dict:
        """Get inventory statistics for admin dashboard"""
        if self._medicines_df is None:
            return {}
        
        return {
            "total_skus": len(self._medicines_df),
            "unique_medicines": self._medicines_df['medicine_name'].nunique(),
            "out_of_stock": len(self._medicines_df[self._medicines_df['stock_level'] == 0]),
            "low_stock": len(self._medicines_df[
                (self._medicines_df['stock_level'] > 0) & 
                (self._medicines_df['stock_level'] <= 50)
            ]),
            "prescription_required": len(self._medicines_df[self._medicines_df['prescription_required'] == True]),
            "discontinued": len(self._medicines_df[self._medicines_df['discontinued'] == True])
        }
    
    def get_medicines_needing_refill(self, patient_id: str, current_date: datetime) -> List[dict]:
        """Calculate which medicines need refill for a patient"""
        history = self.get_patient_order_history(patient_id)
        if history.empty:
            return []
        
        refill_candidates = []
        
        # Group by medicine and get the latest order for each
        for medicine_id in history['medicine_id'].unique():
            med_orders = history[history['medicine_id'] == medicine_id].copy()
            med_orders['purchase_date'] = pd.to_datetime(med_orders['purchase_date'])
            latest_order = med_orders.sort_values('purchase_date', ascending=False).iloc[0]
            
            # Calculate days remaining
            purchase_date = latest_order['purchase_date']
            supply_days = int(latest_order['supply_days'])
            end_date = purchase_date + pd.Timedelta(days=supply_days)
            days_remaining = (end_date - pd.Timestamp(current_date)).days
            
            # Get medicine details
            medicine = self.get_medicine_by_id(medicine_id)
            if medicine:
                refill_candidates.append({
                    "medicine_id": medicine_id,
                    "medicine_name": latest_order['medicine'],
                    "dosage": latest_order['dosage'],
                    "last_quantity": int(latest_order['quantity']),
                    "last_purchase_date": purchase_date.strftime("%Y-%m-%d"),
                    "supply_days": supply_days,
                    "days_remaining": days_remaining,
                    "prescription_required": medicine.prescription_required,
                    "discontinued": medicine.discontinued,
                    "stock_available": medicine.stock_level > 0
                })
        
        return refill_candidates


# Singleton instance
data_service = DataService()