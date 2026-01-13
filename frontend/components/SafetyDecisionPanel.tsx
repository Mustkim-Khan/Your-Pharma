'use client';

import { ShieldCheck, ShieldX, ShieldAlert, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

interface SafetyDecisionPanelProps {
    safety: {
        decision: 'APPROVE' | 'REJECT' | 'CONDITIONAL';
        reasons: string[];
        allowed_quantity: number | null;
        requires_followup: boolean;
        requires_prescription: boolean;
        blocked_items: string[];
    };
}

export default function SafetyDecisionPanel({ safety }: SafetyDecisionPanelProps) {
    const getDecisionConfig = () => {
        switch (safety.decision) {
            case 'APPROVE':
                return {
                    icon: ShieldCheck,
                    bgColor: 'from-green-50 to-emerald-50',
                    borderColor: 'border-green-200',
                    textColor: 'text-green-700',
                    iconColor: 'text-green-500',
                    badge: 'badge-success',
                    label: 'Approved',
                };
            case 'REJECT':
                return {
                    icon: ShieldX,
                    bgColor: 'from-red-50 to-rose-50',
                    borderColor: 'border-red-200',
                    textColor: 'text-red-700',
                    iconColor: 'text-red-500',
                    badge: 'badge-danger',
                    label: 'Rejected',
                };
            case 'CONDITIONAL':
                return {
                    icon: ShieldAlert,
                    bgColor: 'from-yellow-50 to-amber-50',
                    borderColor: 'border-yellow-200',
                    textColor: 'text-yellow-700',
                    iconColor: 'text-yellow-500',
                    badge: 'badge-warning',
                    label: 'Conditional',
                };
            default:
                return {
                    icon: ShieldCheck,
                    bgColor: 'from-gray-50 to-slate-50',
                    borderColor: 'border-gray-200',
                    textColor: 'text-gray-700',
                    iconColor: 'text-gray-500',
                    badge: 'badge-primary',
                    label: 'Unknown',
                };
        }
    };

    const config = getDecisionConfig();
    const Icon = config.icon;

    return (
        <div className={`bg-gradient-to-r ${config.bgColor} rounded-xl border ${config.borderColor} overflow-hidden`}>
            <div className="px-4 py-3 border-b border-gray-100/50">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Icon className={`w-5 h-5 ${config.iconColor}`} />
                        <h3 className="font-semibold text-gray-900">Safety Decision</h3>
                    </div>
                    <span className={`badge ${config.badge}`}>
                        {config.label}
                    </span>
                </div>
                <p className="text-xs text-gray-500 mt-1">AI-powered safety & policy check</p>
            </div>

            <div className="p-4 space-y-3">
                {/* Reasons */}
                {safety.reasons && safety.reasons.length > 0 && (
                    <div className="space-y-2">
                        <p className="text-xs font-medium text-gray-600 uppercase tracking-wide">Reasons</p>
                        {safety.reasons.map((reason, index) => (
                            <div key={index} className="flex items-start gap-2">
                                {safety.decision === 'APPROVE' ? (
                                    <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
                                ) : safety.decision === 'REJECT' ? (
                                    <XCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
                                ) : (
                                    <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0 mt-0.5" />
                                )}
                                <p className={`text-sm ${config.textColor}`}>{reason}</p>
                            </div>
                        ))}
                    </div>
                )}

                {/* Blocked Items */}
                {safety.blocked_items && safety.blocked_items.length > 0 && (
                    <div className="bg-red-100/50 rounded-lg p-3">
                        <p className="text-xs font-medium text-red-700 mb-1">Blocked Items</p>
                        <div className="flex flex-wrap gap-1">
                            {safety.blocked_items.map((item, index) => (
                                <span key={index} className="text-xs bg-red-200 text-red-800 px-2 py-0.5 rounded">
                                    {item}
                                </span>
                            ))}
                        </div>
                    </div>
                )}

                {/* Additional Flags */}
                <div className="flex flex-wrap gap-2 pt-2 border-t border-gray-200/50">
                    {safety.requires_prescription && (
                        <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3" />
                            Prescription Required
                        </span>
                    )}
                    {safety.requires_followup && (
                        <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3" />
                            Follow-up Needed
                        </span>
                    )}
                    {safety.allowed_quantity && (
                        <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">
                            Max Qty: {safety.allowed_quantity}
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
}
