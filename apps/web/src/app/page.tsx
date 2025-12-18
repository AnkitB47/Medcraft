'use client'

import { useState } from 'react'
import { Upload, Activity, FileText, Eye, MessageSquare, Shield, Zap } from 'lucide-react'

export default function Dashboard() {
    const [activeModule, setActiveModule] = useState('parkinson')

    const modules = [
        { id: 'parkinson', name: 'Parkinson Screening', icon: Activity, color: 'text-blue-400' },
        { id: 'cxr', name: 'Chest X-Ray QA', icon: FileText, color: 'text-green-400' },
        { id: 'retina', name: 'Retina Tracking', icon: Eye, color: 'text-purple-400' },
        { id: 'pathology', name: 'Pathology WSI', icon: Zap, color: 'text-yellow-400' },
        { id: 'assistant', name: 'Clinical Assistant', icon: MessageSquare, color: 'text-pink-400' },
    ]

    return (
        <div className="container mx-auto px-4 py-8">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                {/* Sidebar */}
                <div className="lg:col-span-1 space-y-4">
                    <h2 className="text-lg font-semibold mb-4 text-white/80">Medical Modules</h2>
                    {modules.map((mod) => (
                        <button
                            key={mod.id}
                            onClick={() => setActiveModule(mod.id)}
                            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${activeModule === mod.id
                                    ? 'glass-card text-white'
                                    : 'text-white/50 hover:text-white hover:bg-white/5'
                                }`}
                        >
                            <mod.icon className={`w-5 h-5 ${mod.color}`} />
                            <span className="font-medium">{mod.name}</span>
                        </button>
                    ))}
                </div>

                {/* Main Content */}
                <div className="lg:col-span-3 space-y-8">
                    <div className="glass-card rounded-3xl p-8">
                        <div className="flex items-center justify-between mb-8">
                            <div>
                                <h1 className="text-3xl font-bold text-white mb-2">
                                    {modules.find(m => m.id === activeModule)?.name}
                                </h1>
                                <p className="text-white/50">Upload patient data for multimodal analysis.</p>
                            </div>
                            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-green-500/10 text-green-400 text-xs font-semibold border border-green-500/20">
                                <Shield className="w-3 h-3" />
                                Production Ready
                            </div>
                        </div>

                        {/* Upload Area */}
                        <div className="border-2 border-dashed border-white/10 rounded-2xl p-12 flex flex-col items-center justify-center gap-4 hover:border-primary/50 transition-colors cursor-pointer group">
                            <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center group-hover:scale-110 transition-transform">
                                <Upload className="w-8 h-8 text-primary" />
                            </div>
                            <div className="text-center">
                                <p className="text-lg font-medium text-white">Drop files here or click to upload</p>
                                <p className="text-sm text-white/40">Support for DICOM, JPG, PNG, WAV, and WSI pointers</p>
                            </div>
                        </div>

                        {/* Results Preview (Mock) */}
                        <div className="mt-12 space-y-6">
                            <h3 className="text-xl font-semibold text-white">Recent Analysis</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {[1, 2].map((i) => (
                                    <div key={i} className="glass rounded-2xl p-4 flex items-center gap-4 border border-white/5">
                                        <div className="w-12 h-12 rounded-lg bg-white/5 flex items-center justify-center">
                                            <Activity className="w-6 h-6 text-blue-400" />
                                        </div>
                                        <div>
                                            <p className="font-medium text-white">Patient_ID_00{i}</p>
                                            <p className="text-xs text-white/40">Analyzed 2 hours ago • Confidence: 94%</p>
                                        </div>
                                        <button className="ml-auto text-primary text-sm font-medium hover:underline">View Report</button>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
