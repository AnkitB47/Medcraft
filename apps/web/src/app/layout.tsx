import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
    title: 'MedCraft | Multimodal Medical AI',
    description: 'Production-grade multimodal medical AI platform',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en">
            <body className={inter.className}>
                <div className="flex min-h-screen flex-col">
                    <header className="sticky top-0 z-50 glass border-b border-white/10">
                        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center font-bold text-white">M</div>
                                <span className="text-xl font-bold text-gradient">MedCraft</span>
                            </div>
                            <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-white/70">
                                <a href="#" className="hover:text-white transition-colors">Dashboard</a>
                                <a href="#" className="hover:text-white transition-colors">Models</a>
                                <a href="#" className="hover:text-white transition-colors">Datasets</a>
                                <a href="#" className="hover:text-white transition-colors">Settings</a>
                            </nav>
                            <div className="flex items-center gap-4">
                                <button className="px-4 py-2 rounded-full bg-primary hover:bg-primary-dark text-white text-sm font-medium transition-all">
                                    Connect Wallet
                                </button>
                            </div>
                        </div>
                    </header>
                    <main className="flex-1">
                        {children}
                    </main>
                    <footer className="py-8 border-t border-white/10 text-center text-white/40 text-sm">
                        &copy; 2025 MedCraft AI. All rights reserved.
                    </footer>
                </div>
            </body>
        </html>
    )
}
