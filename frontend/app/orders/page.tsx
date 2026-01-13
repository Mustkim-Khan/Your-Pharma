'use client';

import { useState, useEffect } from 'react';
import { CheckCircle, Clock, Package, Truck, XCircle, ChevronDown, ChevronUp, ExternalLink, AlertTriangle, ShieldCheck, Box, Mail } from 'lucide-react';

interface OrderItem {
    medicine_id: string;
    medicine_name: string;
    strength: string;
    quantity: number;
    unit_price: number;
}

interface OrderTimeline {
    step: string;
    title: string;
    description: string;
    agent: string;
    timestamp: string;
    status: 'completed' | 'current' | 'pending' | 'blocked';
}

interface Order {
    order_id: string;
    patient_id: string;
    patient_name: string;
    patient_email: string;
    patient_phone?: string;
    items: OrderItem[];
    total_amount: number;
    status: string;
    created_at: string;
    updated_at: string;
    safety_decision?: string;
    timeline?: OrderTimeline[];
}

const STATUS_STEPS = ['Ordered', 'Validated', 'Confirmed', 'Processing', 'Shipped', 'Delivered'];

const getStatusIndex = (status: string): number => {
    const statusMap: Record<string, number> = {
        'PENDING': 0,
        'VALIDATED': 1,
        'CONFIRMED': 2,
        'PROCESSING': 3,
        'SHIPPED': 4,
        'DELIVERED': 5,
        'BLOCKED': 1, // Stops at validation
        'CANCELLED': 0,
    };
    return statusMap[status?.toUpperCase()] ?? 0;
};

const getStatusBadge = (status: string) => {
    const badges: Record<string, { label: string; color: string; bgColor: string }> = {
        'PENDING': { label: 'Pending', color: 'text-yellow-700', bgColor: 'bg-yellow-100' },
        'CONFIRMED': { label: 'Confirmed', color: 'text-green-700', bgColor: 'bg-green-100' },
        'PROCESSING': { label: 'Processing', color: 'text-blue-700', bgColor: 'bg-blue-100' },
        'SHIPPED': { label: 'Dispatched', color: 'text-teal-700', bgColor: 'bg-teal-100' },
        'DELIVERED': { label: 'Delivered', color: 'text-green-700', bgColor: 'bg-green-100' },
        'BLOCKED': { label: 'Blocked by AI', color: 'text-red-700', bgColor: 'bg-red-100' },
        'CANCELLED': { label: 'Cancelled', color: 'text-gray-700', bgColor: 'bg-gray-100' },
    };
    return badges[status?.toUpperCase()] || badges['PENDING'];
};

const generateTimeline = (order: Order): OrderTimeline[] => {
    const createdDate = new Date(order.created_at);
    const timeline: OrderTimeline[] = [];
    const statusIndex = getStatusIndex(order.status);
    const isBlocked = order.status?.toUpperCase() === 'BLOCKED';

    // Order Requested
    timeline.push({
        step: 'ordered',
        title: 'Order Requested',
        description: 'Order initiated via conversation',
        agent: 'Conversational Ordering Agent',
        timestamp: createdDate.toLocaleString(),
        status: 'completed'
    });

    // AI Safety Validation
    if (isBlocked) {
        timeline.push({
            step: 'validated',
            title: 'Blocked by AI',
            description: 'Blocked: Valid prescription required',
            agent: 'Safety & Policy Agent',
            timestamp: createdDate.toLocaleString(),
            status: 'blocked'
        });
        return timeline;
    }

    timeline.push({
        step: 'validated',
        title: 'AI Safety Validation',
        description: statusIndex >= 1 ? 'Prescription verified and approved' : 'Pending safety verification',
        agent: 'Safety & Policy Agent',
        timestamp: statusIndex >= 1 ? new Date(createdDate.getTime() + 60000).toLocaleString() : '',
        status: statusIndex >= 1 ? 'completed' : 'pending'
    });

    // AI Order Confirmed
    timeline.push({
        step: 'confirmed',
        title: 'AI Order Confirmed',
        description: statusIndex >= 2 ? 'AI validated and confirmed order' : 'Awaiting confirmation',
        agent: 'Safety & Policy Agent',
        timestamp: statusIndex >= 2 ? new Date(createdDate.getTime() + 120000).toLocaleString() : '',
        status: statusIndex >= 2 ? 'completed' : statusIndex === 1 ? 'current' : 'pending'
    });

    // Inventory Updated - INTERNAL ONLY (not shown to customers)
    // The inventory update still happens but is not displayed in customer timeline

    // Fulfillment Initiated
    timeline.push({
        step: 'fulfillment',
        title: 'Fulfillment Initiated',
        description: statusIndex >= 3 ? 'Warehouse notified for fulfillment' : 'Awaiting fulfillment',
        agent: 'Inventory & Fulfillment Agent',
        timestamp: statusIndex >= 3 ? new Date(createdDate.getTime() + 240000).toLocaleString() : '',
        status: statusIndex >= 3 ? 'completed' : 'pending'
    });

    // Dispatched
    timeline.push({
        step: 'shipped',
        title: 'Dispatched',
        description: statusIndex >= 4 ? 'Package dispatched from warehouse' : 'Awaiting dispatch',
        agent: 'System',
        timestamp: statusIndex >= 4 ? new Date(createdDate.getTime() + 3600000).toLocaleString() : '',
        status: statusIndex >= 4 ? 'completed' : statusIndex === 3 ? 'current' : 'pending'
    });

    // Delivered
    timeline.push({
        step: 'delivered',
        title: 'Delivered',
        description: statusIndex >= 5 ? 'Delivered successfully' : 'Awaiting delivery',
        agent: 'System',
        timestamp: statusIndex >= 5 ? new Date(createdDate.getTime() + 86400000).toLocaleString() : '',
        status: statusIndex >= 5 ? 'completed' : statusIndex === 4 ? 'current' : 'pending'
    });

    return timeline;
};

export default function OrdersPage() {
    const [orders, setOrders] = useState<Order[]>([]);
    const [loading, setLoading] = useState(true);
    const [expandedOrderId, setExpandedOrderId] = useState<string | null>(null);
    const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);

    useEffect(() => {
        // Load selected patient context
        const savedPid = localStorage.getItem('selected_patient_id');
        setSelectedPatientId(savedPid);

        fetchOrders();
        // Poll for new orders every 10 seconds
        const interval = setInterval(fetchOrders, 10000);
        return () => clearInterval(interval);
    }, []);

    const fetchOrders = async () => {
        try {
            const res = await fetch('/api/orders');
            const data = await res.json();
            setOrders(data);
        } catch (error) {
            console.error('Failed to fetch orders:', error);
        } finally {
            setLoading(false);
        }
    };

    const toggleExpand = (orderId: string) => {
        setExpandedOrderId(expandedOrderId === orderId ? null : orderId);
    };

    const getStepIcon = (status: string) => {
        switch (status) {
            case 'completed': return <CheckCircle className="w-4 h-4 text-white" />;
            case 'current': return <Clock className="w-4 h-4 text-white" />;
            case 'blocked': return <XCircle className="w-4 h-4 text-white" />;
            default: return null;
        }
    };

    const getStepColor = (status: string) => {
        switch (status) {
            case 'completed': return 'bg-teal-500';
            case 'current': return 'bg-blue-500';
            case 'blocked': return 'bg-red-500';
            default: return 'bg-gray-300';
        }
    };

    // Filter orders by selected patient if one exists
    const filteredOrders = selectedPatientId
        ? orders.filter(o => o.patient_id === selectedPatientId)
        : orders;

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="w-8 h-8 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
        );
    }

    return (
        <div className="p-6 bg-gray-50 dark:bg-gray-900 min-h-screen transition-colors duration-300">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="mb-6 flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Order Confirmation / Details</h1>
                        <p className="text-sm text-gray-500 dark:text-gray-400">Track your orders and view AI-driven fulfillment timeline</p>
                    </div>
                </div>

                {selectedPatientId && filteredOrders.length > 0 && (
                    <div className="mb-6 px-4 py-2 bg-indigo-50 dark:bg-indigo-900/30 border border-indigo-100 dark:border-indigo-800 rounded-lg flex items-center gap-2">
                        <Package className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                        <span className="text-sm text-indigo-700 dark:text-indigo-300 font-medium">
                            Showing orders for active patient context
                        </span>
                    </div>
                )}

                {/* Empty State */}
                {filteredOrders.length === 0 && (
                    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-12 text-center transition-colors">
                        <Package className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
                        <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-2">No Orders Found</h3>
                        <p className="text-gray-500 dark:text-gray-400 mb-4">
                            {selectedPatientId
                                ? "This patient hasn't placed any orders yet."
                                : "No orders found in the system."
                            }
                        </p>
                    </div>
                )}

                {/* Orders List */}
                <div className="space-y-4">
                    {filteredOrders.map((order) => {
                        const statusBadge = getStatusBadge(order.status);
                        const statusIndex = getStatusIndex(order.status);
                        const isBlocked = order.status?.toUpperCase() === 'BLOCKED';
                        const isExpanded = expandedOrderId === order.order_id;
                        // Use real timeline from API if available, otherwise generate client-side
                        // Filter out internal events that customers shouldn't see
                        const internalEvents = ['inventory_updated', 'Inventory Updated'];
                        const timeline = order.timeline && order.timeline.length > 0
                            ? order.timeline
                                .filter((t: any) => !internalEvents.includes(t.action))
                                .map((t: any) => ({
                                    step: t.action?.toLowerCase().replace(/\s+/g, '_') || 'unknown',
                                    title: t.action || 'Unknown',
                                    description: t.description || '',
                                    agent: t.agent_name || 'System',
                                    timestamp: t.timestamp ? new Date(t.timestamp).toLocaleString() : '',
                                    status: t.status || 'completed'
                                }))
                            : generateTimeline(order);
                        const mainItem = order.items[0];

                        return (
                            <div key={order.order_id} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 overflow-hidden transition-colors">
                                {/* Order Header */}
                                <div
                                    className="p-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                                    onClick={() => toggleExpand(order.order_id)}
                                >
                                    <div className="flex items-start gap-4">
                                        {/* Status Icon */}
                                        <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${isBlocked ? 'bg-red-100 dark:bg-red-900/30' :
                                            statusIndex >= 5 ? 'bg-green-100 dark:bg-green-900/30' :
                                                statusIndex >= 4 ? 'bg-teal-100 dark:bg-teal-900/30' : 'bg-teal-100 dark:bg-teal-900/30'
                                            }`}>
                                            {isBlocked ? (
                                                <XCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
                                            ) : statusIndex >= 5 ? (
                                                <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                                            ) : (
                                                <Package className="w-5 h-5 text-teal-600 dark:text-teal-400" />
                                            )}
                                        </div>

                                        {/* Order Info */}
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2 mb-1">
                                                <h3 className="font-semibold text-gray-900 dark:text-white">
                                                    {mainItem?.medicine_name} {mainItem?.strength}
                                                </h3>
                                                <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${statusBadge.bgColor} ${statusBadge.color}`}>
                                                    {statusBadge.label}
                                                </span>
                                            </div>
                                            <p className="text-sm text-gray-500 dark:text-gray-400">
                                                Order #{order.order_id} • {order.patient_name} • Qty: {order.items.reduce((sum, i) => sum + i.quantity, 0)}
                                            </p>
                                        </div>

                                        {/* Price and Date */}
                                        <div className="text-right flex-shrink-0">
                                            <p className="font-semibold text-gray-900 dark:text-white">
                                                ${(order.total_amount * 1.05 + 2).toFixed(2)}
                                            </p>
                                            <p className="text-sm text-gray-500 dark:text-gray-400">
                                                {new Date(order.created_at).toLocaleDateString()}
                                            </p>
                                        </div>

                                        {/* Expand Toggle */}
                                        <div className="flex-shrink-0">
                                            {isExpanded ? (
                                                <ChevronUp className="w-5 h-5 text-gray-400 dark:text-gray-500" />
                                            ) : (
                                                <ChevronDown className="w-5 h-5 text-gray-400 dark:text-gray-500" />
                                            )}
                                        </div>
                                    </div>

                                    {/* Status Timeline Bar */}
                                    <div className="mt-4 flex items-center gap-1">
                                        {STATUS_STEPS.map((step, index) => {
                                            const isCompleted = index <= statusIndex && !isBlocked;
                                            const isCurrent = index === statusIndex && !isBlocked;
                                            const isBlockedStep = isBlocked && index === 1;

                                            return (
                                                <div key={step} className="flex-1 flex flex-col items-center">
                                                    <div className="flex items-center w-full">
                                                        <div className={`w-3 h-3 rounded-full ${isBlockedStep ? 'bg-red-500 dark:bg-red-600' :
                                                            isCompleted ? 'bg-teal-500 dark:bg-teal-500' : 'bg-gray-300 dark:bg-gray-600'
                                                            }`} />
                                                        {index < STATUS_STEPS.length - 1 && (
                                                            <div className={`flex-1 h-1 ${isBlockedStep ? 'bg-red-300 dark:bg-red-800' :
                                                                isCompleted && index < statusIndex ? 'bg-teal-500 dark:bg-teal-500' : 'bg-gray-200 dark:bg-gray-700'
                                                                }`} />
                                                        )}
                                                    </div>
                                                    <p className={`text-xs mt-1 ${isBlockedStep ? 'text-red-600 dark:text-red-400 font-medium' :
                                                        isCompleted ? 'text-teal-600 dark:text-teal-400' : 'text-gray-400 dark:text-gray-500'
                                                        }`}>
                                                        {step}
                                                    </p>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>

                                {/* Expanded Details */}
                                {isExpanded && (
                                    <div className="border-t border-gray-100 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-900/50">
                                        <div className="flex items-center gap-2 mb-4">
                                            <Clock className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                                            <h4 className="font-semibold text-gray-900 dark:text-white">AI-Driven Order Timeline</h4>
                                        </div>

                                        <div className="space-y-4">
                                            {timeline.map((event, index) => (
                                                <div key={index} className="flex gap-4">
                                                    {/* Timeline Dot */}
                                                    <div className="flex flex-col items-center">
                                                        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${getStepColor(event.status)}`}>
                                                            {getStepIcon(event.status) || <span className="w-2 h-2 bg-white rounded-full" />}
                                                        </div>
                                                        {index < timeline.length - 1 && (
                                                            <div className={`w-0.5 flex-1 min-h-8 ${event.status === 'completed' || event.status === 'blocked' ?
                                                                (event.status === 'blocked' ? 'bg-red-300 dark:bg-red-800' : 'bg-teal-300 dark:bg-teal-800') :
                                                                'bg-gray-200 dark:bg-gray-700'
                                                                }`} />
                                                        )}
                                                    </div>

                                                    {/* Event Content */}
                                                    <div className="flex-1 pb-4">
                                                        <div className="flex items-start justify-between">
                                                            <div>
                                                                <p className={`font-medium ${event.status === 'blocked' ? 'text-red-700 dark:text-red-400' : 'text-gray-900 dark:text-white'
                                                                    }`}>
                                                                    {event.title}
                                                                </p>
                                                                <p className={`text-sm ${event.status === 'blocked' ? 'text-red-600 dark:text-red-300' : 'text-gray-500 dark:text-gray-400'
                                                                    }`}>
                                                                    {event.description}
                                                                </p>
                                                                <div className="flex items-center gap-1 mt-1 text-xs text-gray-400 dark:text-gray-500">
                                                                    <ShieldCheck className="w-3 h-3" />
                                                                    <span>{event.agent}</span>
                                                                </div>
                                                            </div>
                                                            <div className="text-right">
                                                                {event.timestamp && (
                                                                    <p className="text-xs text-gray-400 dark:text-gray-500">{event.timestamp}</p>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>

                                        {/* Order Items Summary */}
                                        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                                            <h5 className="font-medium text-gray-700 dark:text-gray-300 mb-2">Order Items</h5>
                                            <div className="space-y-2">
                                                {order.items.map((item, idx) => (
                                                    <div key={idx} className="flex justify-between text-sm">
                                                        <span className="text-gray-600 dark:text-gray-400">
                                                            {item.medicine_name} {item.strength} x{item.quantity}
                                                        </span>
                                                        <span className="text-gray-900 dark:text-white font-medium">
                                                            ${(item.unit_price * item.quantity).toFixed(2)}
                                                        </span>
                                                    </div>
                                                ))}
                                                <div className="flex justify-between text-sm pt-2 border-t border-gray-100 dark:border-gray-700">
                                                    <span className="text-gray-600 dark:text-gray-400">Subtotal</span>
                                                    <span className="text-gray-900 dark:text-white">${order.total_amount.toFixed(2)}</span>
                                                </div>
                                                <div className="flex justify-between text-sm">
                                                    <span className="text-gray-600 dark:text-gray-400">Tax (5%)</span>
                                                    <span className="text-gray-900 dark:text-white">${(order.total_amount * 0.05).toFixed(2)}</span>
                                                </div>
                                                <div className="flex justify-between text-sm">
                                                    <span className="text-gray-600 dark:text-gray-400">Delivery</span>
                                                    <span className="text-gray-900 dark:text-white">$2.00</span>
                                                </div>
                                                <div className="flex justify-between font-semibold text-gray-900 dark:text-white pt-2 border-t border-gray-200 dark:border-gray-700">
                                                    <span>Total</span>
                                                    <span>${(order.total_amount * 1.05 + 2).toFixed(2)}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}
