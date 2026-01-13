'use client';

import { Pill, X, Edit3 } from 'lucide-react';

interface OrderItem {
    medicine_id: string;
    medicine_name: string;
    strength: string;
    quantity: number;
    unit_price?: number;
    prescription_required?: boolean;
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
    const deliveryFee = 2.00;
    const total = subtotal + tax + deliveryFee;

    return (
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-5 max-w-md mx-auto my-4">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="font-semibold text-gray-900 text-lg">Confirm Your Order</h3>
                    <p className="text-sm text-gray-500">Review details before confirming home delivery</p>
                </div>
                {onCancel && (
                    <button
                        onClick={onCancel}
                        className="p-1 hover:bg-gray-100 rounded-full transition-colors"
                    >
                        <X className="w-5 h-5 text-gray-400" />
                    </button>
                )}
            </div>

            {/* Medicine Label */}
            <p className="text-xs text-gray-400 uppercase tracking-wide mb-3">Medicine</p>

            {/* Items */}
            <div className="space-y-3 mb-5">
                {orderPreview.items.map((item, index) => (
                    <div key={index} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                        <div className="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center flex-shrink-0">
                            <Pill className="w-5 h-5 text-indigo-600" />
                        </div>
                        <div className="flex-1">
                            <p className="font-medium text-gray-900">
                                {item.medicine_name} {item.strength}
                            </p>
                            <p className="text-sm text-gray-500">
                                Quantity: {item.quantity} tablets
                            </p>
                            {item.unit_price && (
                                <p className="text-sm text-gray-500">
                                    Price: ${(item.unit_price * item.quantity).toFixed(2)}
                                </p>
                            )}
                        </div>
                    </div>
                ))}
            </div>

            {/* Prescription Notice */}
            {orderPreview.requires_prescription && (
                <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                    <p className="text-sm text-amber-700">
                        ⚠️ This order requires a valid prescription
                    </p>
                </div>
            )}

            {/* Safety Notes */}
            {orderPreview.safety_reasons.length > 0 && orderPreview.safety_decision !== 'APPROVE' && (
                <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-sm text-blue-700 font-medium mb-1">Note:</p>
                    <ul className="text-sm text-blue-600 list-disc list-inside">
                        {orderPreview.safety_reasons.map((reason, idx) => (
                            <li key={idx}>{reason}</li>
                        ))}
                    </ul>
                </div>
            )}

            {/* Price Summary */}
            <div className="border-t border-gray-200 pt-4 mb-5">
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                    <span>Subtotal</span>
                    <span>${subtotal.toFixed(2)}</span>
                </div>
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                    <span>Tax (5%)</span>
                    <span>${tax.toFixed(2)}</span>
                </div>
                <div className="flex justify-between text-sm text-gray-600 mb-2">
                    <span>Delivery</span>
                    <span>${deliveryFee.toFixed(2)}</span>
                </div>
                <div className="flex justify-between font-semibold text-gray-900">
                    <span>Total</span>
                    <span>${total.toFixed(2)}</span>
                </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3">
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
                {onEdit && (
                    <button
                        onClick={onEdit}
                        className="px-5 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded-lg transition-colors flex items-center gap-2"
                    >
                        <Edit3 className="w-4 h-4" />
                        Edit
                    </button>
                )}
            </div>

            {/* Patient Info Footer */}
            <p className="text-xs text-gray-400 text-center mt-4">
                Order for {orderPreview.patient_name} • Preview ID: {orderPreview.preview_id}
            </p>
        </div>
    );
}
