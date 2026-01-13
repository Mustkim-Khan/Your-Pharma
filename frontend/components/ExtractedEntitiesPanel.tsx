'use client';

import { Package, Pill, Hash, Percent } from 'lucide-react';

interface ExtractedEntitiesPanelProps {
    entities: {
        entities: Array<{
            medicine: string;
            dosage: string;
            frequency: string;
            quantity: number;
            confidence: number;
            raw_text: string;
        }>;
        needs_clarification: boolean;
        clarification_message: string;
    };
}

export default function ExtractedEntitiesPanel({ entities }: ExtractedEntitiesPanelProps) {
    if (!entities.entities || entities.entities.length === 0) return null;

    return (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 px-4 py-3 border-b border-gray-100">
                <div className="flex items-center gap-2">
                    <Package className="w-4 h-4 text-primary-600" />
                    <h3 className="font-semibold text-gray-900">Extracted Entities</h3>
                </div>
                <p className="text-xs text-gray-500 mt-1">AI-parsed from your message</p>
            </div>

            <div className="p-4 space-y-3">
                {entities.entities.map((entity, index) => (
                    <div
                        key={index}
                        className="bg-gray-50 rounded-lg p-3 border border-gray-100"
                    >
                        {/* Medicine Name */}
                        <div className="flex items-center gap-2 mb-2">
                            <Pill className="w-4 h-4 text-primary-500" />
                            <span className="font-medium text-gray-900">
                                {entity.medicine || 'Unknown Medicine'}
                            </span>
                        </div>

                        {/* Details Grid */}
                        <div className="grid grid-cols-2 gap-2 text-sm">
                            {entity.dosage && (
                                <div className="flex items-center gap-1">
                                    <span className="text-gray-500">Dosage:</span>
                                    <span className="font-medium text-gray-700">{entity.dosage}</span>
                                </div>
                            )}
                            {entity.quantity > 0 && (
                                <div className="flex items-center gap-1">
                                    <Hash className="w-3 h-3 text-gray-400" />
                                    <span className="text-gray-500">Qty:</span>
                                    <span className="font-medium text-gray-700">{entity.quantity}</span>
                                </div>
                            )}
                            {entity.frequency && (
                                <div className="col-span-2 flex items-center gap-1">
                                    <span className="text-gray-500">Frequency:</span>
                                    <span className="font-medium text-gray-700">{entity.frequency}</span>
                                </div>
                            )}
                        </div>

                        {/* Confidence Bar */}
                        <div className="mt-2 pt-2 border-t border-gray-200">
                            <div className="flex items-center justify-between text-xs mb-1">
                                <span className="text-gray-500 flex items-center gap-1">
                                    <Percent className="w-3 h-3" />
                                    Confidence
                                </span>
                                <span className={`font-medium ${entity.confidence >= 0.8 ? 'text-green-600' :
                                        entity.confidence >= 0.5 ? 'text-yellow-600' :
                                            'text-red-600'
                                    }`}>
                                    {Math.round(entity.confidence * 100)}%
                                </span>
                            </div>
                            <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                                <div
                                    className={`h-full rounded-full transition-all ${entity.confidence >= 0.8 ? 'bg-green-500' :
                                            entity.confidence >= 0.5 ? 'bg-yellow-500' :
                                                'bg-red-500'
                                        }`}
                                    style={{ width: `${entity.confidence * 100}%` }}
                                />
                            </div>
                        </div>
                    </div>
                ))}

                {/* Clarification Notice */}
                {entities.needs_clarification && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                        <p className="text-sm text-yellow-800">
                            <strong>Note:</strong> Some information needs clarification
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}
