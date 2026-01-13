'use client';

import { ShoppingCart, Package, DollarSign, Check, X, Pill } from 'lucide-react';

interface OrderPreviewCardProps {
    preview: {
        preview_id: string;
        patient_id: string;
        patient_name: string;
        items: Array<{
            medicine_id: string;
            medicine_name: string;
            strength: string;
            quantity: number;
            prescription_required: boolean;
        }>;
        total_amount: number;
        safety_decision: string;
        safety_reasons: string[];
        requires_prescription: boolean;
    };
    onConfirm: () => void;
    onCancel: () => void;
}

export default function OrderPreviewCard({ preview, onConfirm, onCancel }: OrderPreviewCardProps) {
    return (
        <div className="bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-primary-500 to-purple-600 px-4 py-3">
                <div className="flex items-center gap-2 text-white">
                    <ShoppingCart className="w-5 h-5" />
                    <h3 className="font-semibold">Order Preview</h3>
                </div>
                <p className="text-primary-100 text-xs mt-1">Review and confirm your order</p>
            </div>

            {/* Items */}
            <div className="p-4">
                <div className="space-y-2 mb-4">
                    {preview.items.map((item, index) => (
                        <div
                            key={index}
                            className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                        >
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-lg bg-primary-100 flex items-center justify-center">
                                    <Pill className="w-5 h-5 text-primary-600" />
                                </div>
                                <div>
                                    <p className="font-medium text-gray-900">{item.medicine_name}</p>
                                    <p className="text-xs text-gray-500">{item.strength}</p>
                                </div>
                            </div>
                            <div className="text-right">
                                <p className="font-semibold text-gray-900">x{item.quantity}</p>
                                {item.prescription_required && (
                                    <span className="text-xs text-purple-600">Rx</span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Total */}
                <div className="flex items-center justify-between py-3 border-t border-gray-100">
                    <span className="text-gray-600">Estimated Total</span>
                    <span className="text-xl font-bold text-gray-900 flex items-center gap-1">
                        <DollarSign className="w-5 h-5" />
                        {preview.total_amount.toFixed(2)}
                    </span>
                </div>

                {/* Prescription Notice */}
                {preview.requires_prescription && (
                    <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 mb-4">
                        <p className="text-sm text-purple-800">
                            <strong>Note:</strong> This order contains prescription medications.
                            Your prescription will be verified.
                        </p>
                    </div>
                )}

                {/* Actions */}
                <div className="flex gap-2 pt-2">
                    <button
                        onClick={onCancel}
                        className="btn-secondary flex-1 flex items-center justify-center gap-2"
                    >
                        <X className="w-4 h-4" />
                        Cancel
                    </button>
                    <button
                        onClick={onConfirm}
                        className="btn-success flex-1 flex items-center justify-center gap-2"
                    >
                        <Check className="w-4 h-4" />
                        Confirm Order
                    </button>
                </div>
            </div>
        </div>
    );
}
