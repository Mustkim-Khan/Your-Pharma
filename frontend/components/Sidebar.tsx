'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Bot, Moon, Sun } from 'lucide-react';
import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

const navItems = [
    { href: '/', label: 'Conversational Chat' },
    { href: '/admin', label: 'Admin Inventory Dashboard' },
    { href: '/refills', label: 'Proactive Refill Alerts' },
    { href: '/orders', label: 'Order Confirmation / Details' },
];

export default function Sidebar() {
    const pathname = usePathname();
    const { theme, setTheme } = useTheme();
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    const toggleTheme = () => {
        setTheme(theme === 'dark' ? 'light' : 'dark');
    };

    return (
        <header className="fixed left-0 top-0 right-0 h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 flex items-center px-6 z-50 transition-colors duration-300">
            {/* Logo */}
            <div className="flex items-center gap-2 mr-8">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                    <Bot className="w-5 h-5 text-white" />
                </div>
                <h1 className="font-semibold text-gray-900 dark:text-white transition-colors">Your Pharma Ai</h1>
            </div>

            {/* Navigation */}
            <nav className="flex items-center gap-1 mx-auto">
                {navItems.map((item) => {
                    const isActive = pathname === item.href;

                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${isActive
                                ? 'bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-800'
                                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-800'
                                }`}
                        >
                            {item.label}
                        </Link>
                    );
                })}
            </nav>

            {/* Theme Toggle */}
            {mounted && (
                <button
                    onClick={toggleTheme}
                    className="ml-4 p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800 transition-colors"
                    aria-label="Toggle Dark Mode"
                >
                    {theme === 'dark' ? (
                        <Sun className="w-5 h-5" />
                    ) : (
                        <Moon className="w-5 h-5" />
                    )}
                </button>
            )}
        </header>
    );
}
