'use client';

import { useState, useEffect } from 'react';
import {
    RefreshCw, Clock, AlertTriangle, CheckCircle,
    Search, ChevronDown, ChevronLeft, ChevronRight,
    TrendingUp, Shield, Package
} from 'lucide-react';

interface RefillPrediction {
    patient_id: string;
    patient_name: string;
    medicine: string;
    medicine_id: string;
    dosage: string;
    days_remaining: number;
    last_purchase_date: string;
    action: 'REMIND' | 'AUTO_REFILL' | 'BLOCK';
    justification: string;
    urgency: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}

export default function RefillsPage() {
    const [refills, setRefills] = useState<RefillPrediction[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');
    const [medicineFilter, setMedicineFilter] = useState('all');
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 5;

    useEffect(() => {
        fetchRefills();
    }, []);

    const fetchRefills = async () => {
        try {
            const res = await fetch('/api/refills');
            const data = await res.json();
            setRefills(data);
        } catch (error) {
            console.error('Failed to fetch refills:', error);
        } finally {
            setLoading(false);
        }
    };

    const filteredRefills = refills.filter(refill => {
        const matchesSearch = refill.patient_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            refill.medicine.toLowerCase().includes(searchQuery.toLowerCase());

        if (!matchesSearch) return false;

        if (statusFilter !== 'all') {
            if (statusFilter === 'due_soon' && refill.days_remaining > 3) return false;
            if (statusFilter === 'overdue' && refill.days_remaining >= 0) return false;
            if (statusFilter === 'scheduled' && refill.action !== 'AUTO_REFILL') return false;
        }

        return true;
    });

    const paginatedRefills = filteredRefills.slice(
        (currentPage - 1) * itemsPerPage,
        currentPage * itemsPerPage
    );

    const totalPages = Math.ceil(filteredRefills.length / itemsPerPage);

    const getRefillStatus = (days: number, action: string) => {
        if (action === 'AUTO_REFILL') {
            return { label: 'Scheduled', color: 'bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 border-indigo-200 dark:border-indigo-800' };
        }
        if (days < 0) {
            return { label: 'Overdue', color: 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 border-red-200 dark:border-red-800' };
        }
        if (days <= 3) {
            return { label: 'Due Soon', color: 'bg-yellow-50 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300 border-yellow-200 dark:border-yellow-800' };
        }
        return { label: 'Scheduled', color: 'bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 border-indigo-200 dark:border-indigo-800' };
    };

    const getAIActionStatus = (action: string, days: number) => {
        if (action === 'AUTO_REFILL') {
            return { label: 'Auto-refill scheduled', color: 'bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400' };
        }
        if (action === 'BLOCK') {
            return { label: 'Refill blocked – prescription required', color: 'bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400' };
        }
        if (days < 0) {
            return { label: 'Refill blocked – prescription required', color: 'bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400' };
        }
        if (action === 'REMIND') {
            return { label: 'AI reminder sent', color: 'bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400' };
        }
        return { label: 'Awaiting prescription', color: 'bg-yellow-50 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400' };
    };

    return (
        <div className="p-6 bg-gray-50 dark:bg-gray-900 min-h-screen transition-colors duration-300">
            {/* Header */}
            <div className="mb-6 flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Proactive Refill Alerts</h1>
                </div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Admin role: read-only — no manual execution, approvals, or order creation</p>
            </div>

            {/* Agent Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-100 dark:border-gray-700 flex items-center justify-between transition-colors">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-indigo-100 dark:bg-indigo-900/50 flex items-center justify-center">
                            <TrendingUp className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                        </div>
                        <span className="font-medium text-gray-900 dark:text-white">Predictive Refill Agent</span>
                    </div>
                    <span className="px-3 py-1 bg-green-50 dark:bg-green-900/30 text-green-600 dark:text-green-400 text-xs font-medium rounded-full border border-green-200 dark:border-green-800 flex items-center gap-1">
                        <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span>
                        Active
                    </span>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-100 dark:border-gray-700 flex items-center justify-between transition-colors">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-purple-100 dark:bg-purple-900/50 flex items-center justify-center">
                            <Shield className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                        </div>
                        <span className="font-medium text-gray-900 dark:text-white">Safety & Prescription Policy Agent</span>
                    </div>
                    <span className="px-3 py-1 bg-green-50 dark:bg-green-900/30 text-green-600 dark:text-green-400 text-xs font-medium rounded-full border border-green-200 dark:border-green-800 flex items-center gap-1">
                        <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span>
                        Active
                    </span>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-100 dark:border-gray-700 flex items-center justify-between transition-colors">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-amber-100 dark:bg-amber-900/50 flex items-center justify-center">
                            <Package className="w-5 h-5 text-amber-600 dark:text-amber-400" />
                        </div>
                        <span className="font-medium text-gray-900 dark:text-white">Inventory & Fulfillment Agent</span>
                    </div>
                    <span className="px-3 py-1 bg-green-50 dark:bg-green-900/30 text-green-600 dark:text-green-400 text-xs font-medium rounded-full border border-green-200 dark:border-green-800 flex items-center gap-1">
                        <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span>
                        Active
                    </span>
                </div>
            </div>

            {/* Filters Row */}
            <div className="flex gap-4 mb-6">
                <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <input
                        type="text"
                        placeholder="Search patients or medicine"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-10 pr-4 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                    />
                </div>
                <div className="relative">
                    <select
                        value={statusFilter}
                        onChange={(e) => setStatusFilter(e.target.value)}
                        className="appearance-none px-4 py-2 pr-10 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm text-gray-900 dark:text-white"
                    >
                        <option value="all">Filter by Status</option>
                        <option value="due_soon">Due Soon</option>
                        <option value="overdue">Overdue</option>
                        <option value="scheduled">Scheduled</option>
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                </div>
                <div className="relative">
                    <select
                        value={medicineFilter}
                        onChange={(e) => setMedicineFilter(e.target.value)}
                        className="appearance-none px-4 py-2 pr-10 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm text-gray-900 dark:text-white"
                    >
                        <option value="all">Filter by Medicine</option>
                        {Array.from(new Set(refills.map(r => r.medicine))).map(med => (
                            <option key={med} value={med}>{med}</option>
                        ))}
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                </div>
            </div>

            {/* Refill Alerts Section */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 overflow-hidden transition-colors">
                <div className="p-4 border-b border-gray-100 dark:border-gray-700">
                    <h2 className="font-semibold text-gray-900 dark:text-white">Refill Alerts</h2>
                    <p className="text-sm text-gray-500 dark:text-gray-400">All refills and reminders are autonomously executed by AI agents.</p>
                </div>

                {/* Table */}
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-gray-50 dark:bg-gray-700/50">
                            <tr className="text-left text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider border-b border-gray-100 dark:border-gray-700">
                                <th className="px-6 py-3 font-medium">Patient Name</th>
                                <th className="px-6 py-3 font-medium">Medicine</th>
                                <th className="px-6 py-3 font-medium">Dosage</th>
                                <th className="px-6 py-3 font-medium">Days Remaining</th>
                                <th className="px-6 py-3 font-medium">Refill Status</th>
                                <th className="px-6 py-3 font-medium">AI Action Status</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                            {loading ? (
                                <tr>
                                    <td colSpan={6} className="text-center py-8">
                                        <RefreshCw className="w-6 h-6 text-gray-400 mx-auto animate-spin" />
                                        <p className="text-gray-500 dark:text-gray-400 mt-2">Loading...</p>
                                    </td>
                                </tr>
                            ) : paginatedRefills.length === 0 ? (
                                <tr>
                                    <td colSpan={6} className="text-center py-8">
                                        <CheckCircle className="w-8 h-8 text-green-400 mx-auto mb-2" />
                                        <p className="text-gray-600 dark:text-gray-400 font-medium">All caught up!</p>
                                    </td>
                                </tr>
                            ) : (
                                paginatedRefills.map((refill, index) => {
                                    const status = getRefillStatus(refill.days_remaining, refill.action);
                                    const aiAction = getAIActionStatus(refill.action, refill.days_remaining);

                                    return (
                                        <tr key={index} className="border-b border-gray-50 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                                            <td className="px-6 py-4 text-sm font-medium text-gray-900 dark:text-white">
                                                {refill.patient_name}
                                            </td>
                                            <td className="px-6 py-4 text-sm text-gray-600 dark:text-gray-300">
                                                {refill.medicine}
                                            </td>
                                            <td className="px-6 py-4 text-sm text-gray-600 dark:text-gray-300">
                                                {refill.dosage || '1 tablet/day'}
                                            </td>
                                            <td className="px-6 py-4 text-sm">
                                                <span className={refill.days_remaining < 0 ? 'text-red-600 dark:text-red-400' : 'text-gray-600 dark:text-gray-300'}>
                                                    {refill.days_remaining < 0
                                                        ? `${Math.abs(refill.days_remaining)} days overdue`
                                                        : `${refill.days_remaining} days`
                                                    }
                                                </span>
                                            </td>
                                            <td className="px-6 py-4">
                                                <span className={`px-3 py-1 text-xs font-medium rounded-full border ${status.color}`}>
                                                    {status.label}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4">
                                                <span className={`px-3 py-1 text-xs rounded ${aiAction.color}`}>
                                                    {aiAction.label}
                                                </span>
                                            </td>
                                        </tr>
                                    );
                                })
                            )}
                        </tbody>
                    </table>
                </div>

                {/* Pagination */}
                <div className="px-6 py-4 flex items-center justify-between border-t border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 transition-colors">
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                        Showing {((currentPage - 1) * itemsPerPage) + 1}-{Math.min(currentPage * itemsPerPage, filteredRefills.length)} of {filteredRefills.length} results
                    </p>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                            disabled={currentPage === 1}
                            className="px-3 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
                        >
                            <ChevronLeft className="w-4 h-4" />
                            Previous
                        </button>
                        {[...Array(totalPages)].map((_, i) => (
                            <button
                                key={i}
                                onClick={() => setCurrentPage(i + 1)}
                                className={`w-8 h-8 rounded text-sm font-medium ${currentPage === i + 1
                                    ? 'bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900'
                                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                                    }`}
                            >
                                {i + 1}
                            </button>
                        ))}
                        <button
                            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                            disabled={currentPage === totalPages}
                            className="px-3 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
                        >
                            Next
                            <ChevronRight className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
