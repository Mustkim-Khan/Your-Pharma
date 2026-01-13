'use client';

import { useState, useEffect } from 'react';
import {
    Search, AlertCircle, ExternalLink
} from 'lucide-react';

interface Medicine {
    medicine_id: string;
    medicine_name: string;
    strength: string;
    form: string;
    stock_level: number;
    prescription_required: boolean;
    category: string;
    discontinued: boolean;
    controlled_substance: boolean;
    last_updated?: string;
}

interface InventoryStats {
    total_skus: number;
    unique_medicines: number;
    out_of_stock: number;
    low_stock: number;
    prescription_required: number;
    discontinued: number;
}

export default function AdminPage() {
    const [medicines, setMedicines] = useState<Medicine[]>([]);
    const [stats, setStats] = useState<InventoryStats | null>(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        try {
            const [medicinesRes, statsRes] = await Promise.all([
                fetch('/api/inventory'),
                fetch('/api/inventory/stats'),
            ]);

            const medicinesData = await medicinesRes.json();
            const statsData = await statsRes.json();

            setMedicines(medicinesData);
            setStats(statsData);
        } catch (error) {
            console.error('Failed to fetch inventory:', error);
        } finally {
            setLoading(false);
        }
    };

    const filteredMedicines = medicines.filter(med =>
        med.medicine_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        med.medicine_id.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const getStockLabel = (level: number) => {
        if (level === 0) {
            return { text: `0 Out of Stock`, color: 'text-red-600 dark:text-red-400' };
        } else if (level <= 100) {
            return { text: `${level} Low Stock`, color: 'text-orange-600 dark:text-orange-400' };
        }
        return { text: `${level} High Stock`, color: 'text-gray-900 dark:text-white' };
    };

    return (
        <div className="p-6 bg-gray-50 dark:bg-gray-900 min-h-screen transition-colors duration-300">
            {/* Header */}
            <div className="mb-6 flex items-center gap-3">
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Admin Inventory Dashboard</h1>
                <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                    <AlertCircle className="w-4 h-4" />
                    Inventory is managed autonomously by AI. Admin view is read-only.
                </div>
            </div>

            {/* Search */}
            <div className="relative max-w-md mb-6">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                    type="text"
                    placeholder="Search medicine by name..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                />
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-sm border border-gray-100 dark:border-gray-700 transition-colors">
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Total SKUs</p>
                    <p className="text-3xl font-bold text-indigo-600 dark:text-indigo-400">{stats?.total_skus || 150}</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Different types of products in stock.</p>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-sm border border-gray-100 dark:border-gray-700 transition-colors">
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Unique Items in Inventory</p>
                    <p className="text-3xl font-bold text-gray-900 dark:text-white">{stats?.unique_medicines || 75}</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Distinct medicines available.</p>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-xl p-5 shadow-sm border border-gray-100 dark:border-gray-700 transition-colors">
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Out of Stock</p>
                    <p className="text-3xl font-bold text-red-600 dark:text-red-400">{stats?.out_of_stock || 3}</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Items with zero quantity</p>
                </div>
            </div>

            {/* Medicine Inventory Table */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden transition-colors">
                <div className="p-4 border-b border-gray-100 dark:border-gray-700">
                    <h2 className="font-semibold text-gray-900 dark:text-white">Medicine Inventory</h2>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="text-left text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider border-b border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-700/50">
                                <th className="px-6 py-3 font-medium">Medicine Name</th>
                                <th className="px-6 py-3 font-medium">Strength</th>
                                <th className="px-6 py-3 font-medium">Form</th>
                                <th className="px-6 py-3 font-medium">Stock Level</th>
                                <th className="px-6 py-3 font-medium">Prescription Required</th>
                                <th className="px-6 py-3 font-medium">Category</th>
                                <th className="px-6 py-3 font-medium">Discontinued</th>
                                <th className="px-6 py-3 font-medium">Last Updated</th>
                                <th className="px-6 py-3 font-medium">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                            {filteredMedicines.map((med) => {
                                const stockInfo = getStockLabel(med.stock_level);

                                return (
                                    <tr key={med.medicine_id} className="border-b border-gray-50 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                                        <td className="px-6 py-4 text-sm font-medium text-gray-900 dark:text-white">
                                            {med.medicine_name}
                                        </td>
                                        <td className="px-6 py-4 text-sm text-gray-600 dark:text-gray-300">
                                            {med.strength}
                                        </td>
                                        <td className="px-6 py-4 text-sm text-gray-600 dark:text-gray-300">
                                            {med.form}
                                        </td>
                                        <td className="px-6 py-4 text-sm">
                                            <span className={stockInfo.color}>
                                                {stockInfo.text}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            {med.prescription_required ? (
                                                <span className="px-2 py-1 bg-indigo-50 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 text-xs font-medium rounded">
                                                    Rx Required
                                                </span>
                                            ) : (
                                                <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 text-xs font-medium rounded">
                                                    OTC
                                                </span>
                                            )}
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className="px-2 py-1 bg-purple-50 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300 text-xs font-medium rounded">
                                                {med.category}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            {med.discontinued ? (
                                                <span className="px-2 py-1 bg-red-50 dark:bg-red-900/40 text-red-700 dark:text-red-300 text-xs font-medium rounded">
                                                    Discontinued
                                                </span>
                                            ) : (
                                                <span className="px-2 py-1 bg-green-50 dark:bg-green-900/40 text-green-700 dark:text-green-300 text-xs font-medium rounded">
                                                    Active
                                                </span>
                                            )}
                                        </td>
                                        <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
                                            {med.last_updated || '2025-08-26'}
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex flex-col gap-1">
                                                <span className="text-xs text-gray-500 dark:text-gray-400">Managed by Ai</span>
                                            </div>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
