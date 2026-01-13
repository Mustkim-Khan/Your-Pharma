import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Sidebar from '@/components/Sidebar';
import Providers from '@/components/Providers';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
    title: 'Your Pharma AI - Agentic Pharmacy System',
    description: 'Autonomous AI-powered pharmacy with multi-agent architecture',
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <body className={`${inter.className} min-h-screen flex flex-col bg-gray-50 dark:bg-gray-900 transition-colors duration-300`}>
                <Providers>
                    <Sidebar />
                    <main className="pt-16 flex-1 flex flex-col">
                        {children}
                    </main>
                    {/* Footer */}
                    <footer className="text-center py-4 text-sm text-gray-500 border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 transition-colors duration-300">
                        © 2026 Your Pharma AI. All rights reserved.
                    </footer>
                </Providers>
            </body>
        </html>
    );
}
