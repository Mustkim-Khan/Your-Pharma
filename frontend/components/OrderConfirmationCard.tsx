'use client';

import { Pill, X, Edit3, Truck, Clock } from 'lucide-react';

interface OrderItem {
    medicine_id: string;
    medicine_name: string;
    strength: string;
    quantity: number;
    unit_price?: number;
    prescription_required?: boolean;
    supply_days?: number; // Estimated supply duration
}

interface OrderPreview {
    preview_id: string;
    patient_id: string;
    patient_name: string;
    items: OrderItem[];
    total_amount: number;
    safety_decision: string;
    safety_reasons: string[];
    requires_prescription: boolean;
    created_at: string;
    estimated_delivery?: string; // e.g., "Tomorrow by 9:00 PM"
}

interface OrderConfirmationCardProps {
    orderPreview: OrderPreview;
    onConfirm: () => void;
    onEdit?: () => void;
    onCancel?: () => void;
    isLoading?: boolean;
}

export default function OrderConfirmationCard({
    orderPreview,
    onConfirm,
    onEdit,
    onCancel,
    isLoading = false
}: OrderConfirmationCardProps) {
    const subtotal = orderPreview.total_amount;
    const tax = subtotal * 0.05;
    const total = subtotal + tax;

    // Calculate average unit price
    const totalQuantity = orderPreview.items.reduce((sum, item) => sum + item.quantity, 0);
    const pricePerUnit = totalQuantity > 0 ? subtotal / totalQuantity : 0;

    // Default delivery estimate
    const deliveryEstimate = orderPreview.estimated_delivery || 'Tomorrow by 9:00 PM';

    return (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border-l-4 border-l-indigo-600 border border-gray-200 dark:border-gray-700 p-5 max-w-md my-4 transition-colors duration-300">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Confirm Your Order</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Review details before confirming home delivery</p>
                </div>
                {onCancel && (
                    <button
                        onClick={onCancel}
                        className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors"
                    >
                        <X className="w-5 h-5 text-gray-400" />
                    </button>
                )}
            </div>

            {/* Medicine Label */}
            <p className="text-xs text-gray-400 uppercase tracking-wider mb-3 font-medium">Medicine Details</p>

            {/* Items */}
            <div className="space-y-3 mb-5">
                {orderPreview.items.map((item, index) => {
                    const itemTotal = item.unit_price ? item.unit_price * item.quantity : 0;
                    const supplyDays = item.supply_days || Math.round(item.quantity * 0.5); // Estimate ~45 days for 90 tablets

                    return (
                        <div key={index} className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                            <div className="w-10 h-10 rounded-full bg-indigo-100 dark:bg-indigo-900/50 flex items-center justify-center flex-shrink-0">
                                <Pill className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                            </div>
                            <div className="flex-1 min-w-0">
                                <div className="flex items-center justify-between gap-2">
                                    <p className="font-medium text-gray-900 dark:text-gray-100">
                                        {item.medicine_name}
                                    </p>
                                    {item.unit_price && (
                                        <p className="text-sm font-semibold text-gray-700 dark:text-gray-200 whitespace-nowrap">
                                            ${itemTotal.toFixed(2)}
                                        </p>
                                    )}
                                </div>
                                <p className="text-sm text-gray-500 dark:text-gray-400">
                                    {item.strength} • {item.quantity} units
                                </p>
                                <p className="text-xs text-cyan-600 dark:text-cyan-400 mt-1">
                                    Supply: ~{supplyDays} days
                                </p>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Prescription Notice */}
            {orderPreview.requires_prescription && (
                <div className="mb-4 p-3 bg-amber-50 dark:bg-amber-900/30 border border-amber-200 dark:border-amber-700 rounded-lg">
                    <p className="text-sm text-amber-700 dark:text-amber-400">
                        ⚠️ This order requires a valid prescription
                    </p>
                </div>
            )}

            {/* Safety Notes */}
            {orderPreview.safety_reasons.length > 0 && orderPreview.safety_decision !== 'APPROVE' && (
                <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 rounded-lg">
                    <p className="text-sm text-blue-700 dark:text-blue-300 font-medium mb-1">Note:</p>
                    <ul className="text-sm text-blue-600 dark:text-blue-400 list-disc list-inside">
                        {orderPreview.safety_reasons.map((reason, idx) => (
                            <li key={idx}>{reason}</li>
                        ))}
                    </ul>
                </div>
            )}

            {/* Price Summary */}
            <div className="border-t border-gray-200 dark:border-gray-600 pt-4 mb-4">
                <div className="flex justify-between text-sm text-gray-500 dark:text-gray-400 mb-2">
                    <span>Price per unit</span>
                    <span>${pricePerUnit.toFixed(2)}/unit</span>
                </div>
                <div className="flex justify-between items-baseline">
                    <div>
                        <p className="font-semibold text-gray-900 dark:text-white">Total Amount</p>
                        <p className="text-xs text-gray-400 dark:text-gray-500">Inclusive of all taxes</p>
                    </div>
                    <span className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                        ${total.toFixed(2)}
                    </span>
                </div>
            </div>

            {/* Home Delivery Section */}
            <div className="mb-5 p-3 bg-gray-50 dark:bg-gray-700/40 rounded-lg flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-indigo-100 dark:bg-indigo-900/50 flex items-center justify-center flex-shrink-0">
                    <Truck className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                </div>
                <div>
                    <p className="font-medium text-gray-900 dark:text-white">Home Delivery</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        Est. {deliveryEstimate}
                    </p>
                </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3">
                {onCancel && (
                    <button
                        onClick={onCancel}
                        className="flex-1 py-3 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 font-medium rounded-lg transition-colors"
                    >
                        Cancel
                    </button>
                )}
                <button
                    onClick={onConfirm}
                    disabled={isLoading}
                    className="flex-1 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {isLoading ? (
                        <span className="flex items-center justify-center gap-2">
                            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                            Processing...
                        </span>
                    ) : (
                        'Confirm Order'
                    )}
                </button>
            </div>

            {/* Patient Info Footer */}
            <p className="text-xs text-gray-400 text-center mt-4">
                Order for {orderPreview.patient_name} • Preview ID: {orderPreview.preview_id}
            </p>
        </div>
    );
}
