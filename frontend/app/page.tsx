'use client';

import { useState, useEffect, useRef } from 'react';
import { Send, Mic, MicOff, Loader2, Paperclip, CheckCircle, ExternalLink, Settings, ChevronRight, Clock, ChevronDown, User, Volume2, Pill, Upload, AlertCircle, X, Truck } from 'lucide-react';
import PrescriptionUploadCard from '@/components/PrescriptionUploadCard';

interface Patient {
    patient_id: string;
    patient_name: string;
    patient_email: string;
    patient_phone: string;
}

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    extractedEntities?: any;
    safetyResult?: any;
    orderPreview?: any;
    order?: any;
    traceUrl?: string;
    aiAnnotation?: string;
    badges?: { label: string; color: string }[];
    expandable?: boolean;
}

// Helper function to format markdown content (bold, newlines)
const formatMarkdown = (text: string): string => {
    if (!text) return '';
    // Convert **bold** to <strong>bold</strong>
    let formatted = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Convert newlines to <br>
    formatted = formatted.replace(/\n/g, '<br>');
    return formatted;
};

export default function ChatPage() {
    const [patients, setPatients] = useState<Patient[]>([]);
    const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [currentEntities, setCurrentEntities] = useState<any>(null);
    const [latestTraceUrl, setLatestTraceUrl] = useState<string | null>(null);
    const [autoSaveChats, setAutoSaveChats] = useState(true);
    const [desktopNotifications, setDesktopNotifications] = useState(false);
    const [isPatientDropdownOpen, setIsPatientDropdownOpen] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const dropdownRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        fetchPatients();
    }, []);

    // Load messages from localStorage when patient changes
    useEffect(() => {
        if (selectedPatient) {
            const savedMessages = localStorage.getItem(`chat_messages_${selectedPatient.patient_id}`);
            if (savedMessages) {
                try {
                    const parsed = JSON.parse(savedMessages);
                    // Convert timestamp strings back to Date objects
                    const messagesWithDates = parsed.map((msg: any) => ({
                        ...msg,
                        timestamp: new Date(msg.timestamp)
                    }));
                    setMessages(messagesWithDates);
                } catch (e) {
                    console.error('Failed to parse saved messages:', e);
                }
            } else {
                setMessages([]); // Clear messages for new patient
            }

            // Load saved entities for this patient
            const savedEntities = localStorage.getItem(`patient_entities_${selectedPatient.patient_id}`);
            if (savedEntities) {
                try {
                    setCurrentEntities(JSON.parse(savedEntities));
                } catch (e) {
                    console.error('Failed to parse saved entities:', e);
                    setCurrentEntities(null);
                }
            } else {
                setCurrentEntities(null);
            }
        }
    }, [selectedPatient?.patient_id]);

    // Save messages to localStorage whenever they change
    useEffect(() => {
        if (selectedPatient && messages.length > 0 && autoSaveChats) {
            localStorage.setItem(`chat_messages_${selectedPatient.patient_id}`, JSON.stringify(messages));
        }
    }, [messages, selectedPatient?.patient_id, autoSaveChats]);

    // Save entities to localStorage whenever they change
    useEffect(() => {
        if (selectedPatient) {
            if (currentEntities) {
                localStorage.setItem(`patient_entities_${selectedPatient.patient_id}`, JSON.stringify(currentEntities));
            } else {
                // Optional: decide if we want to remove it when null, or keep last known state?
                // User said "dont change it until that customer next order".
                // So if it becomes null (explicit clear), we might want to remove it.
                // But generally we only set it to null on patient switch (handled above) or if explicitly cleared.
                // Let's remove if null to keep it clean.
                // Actually, if we want to "persist", we simply setItem.
                // If it is null on patient switch, the logic above handles loading.
            }
        }
    }, [currentEntities, selectedPatient?.patient_id]);

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsPatientDropdownOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const fetchPatients = async () => {
        try {
            const res = await fetch('/api/patients');
            const data = await res.json();
            setPatients(data);

            // Try to restore previous selection
            const savedPatientId = localStorage.getItem('selected_patient_id');
            const savedPatient = data.find((p: Patient) => p.patient_id === savedPatientId);

            if (savedPatient) {
                setSelectedPatient(savedPatient);
            } else if (data.length > 0) {
                // Default to first patient if no save found
                setSelectedPatient(data[0]);
                localStorage.setItem('selected_patient_id', data[0].patient_id);
            }
        } catch (error) {
            console.error('Failed to fetch patients:', error);
        }
    };

    const handlePatientChange = (patient: Patient) => {
        setSelectedPatient(patient);
        localStorage.setItem('selected_patient_id', patient.patient_id);
        // Messages will be loaded from localStorage by the useEffect when selectedPatient changes
        setIsPatientDropdownOpen(false);
    };

    const sendMessage = async (text: string) => {
        if (!text.trim() || !selectedPatient || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: text,
            timestamp: new Date(),
        };

        setMessages(prev => [...prev, userMessage]);
        setInputValue('');
        setIsLoading(true);
        // setCurrentEntities(null); // Removed to persist entities across turns

        try {
            // Build conversation history from previous messages
            const conversationHistory = messages.map(msg => ({
                role: msg.role,
                content: msg.content
            }));

            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    patient_id: selectedPatient.patient_id,
                    message: text,
                    session_id: `session-${selectedPatient.patient_id}`,
                    conversation_history: conversationHistory,
                }),
            });


            const data = await res.json();

            const assistantMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: data.message,
                timestamp: new Date(),
                extractedEntities: data.extracted_entities,
                safetyResult: data.safety_result,
                orderPreview: data.order_preview,
                order: data.order,
                traceUrl: data.trace_url,
                aiAnnotation: data.ai_annotation,
                badges: data.badges,
                expandable: true,
            };

            setMessages(prev => [...prev, assistantMessage]);

            // Logic to persist or update extracted entities
            if (data.extracted_entities && data.extracted_entities.entities && data.extracted_entities.entities.length > 0) {
                // 1. If new entities are extracted, update the view (Start of new order)
                setCurrentEntities(data.extracted_entities);
            } else if (data.order) {
                // 2. If order is confirmed, persist the confirmed items as entities
                const confirmedEntities = {
                    entities: data.order.items.map((item: any) => ({
                        medicine: item.medicine_name,
                        dosage: item.strength,
                        quantity: item.quantity,
                        frequency: null,
                        duration: null
                    }))
                };
                setCurrentEntities(confirmedEntities);
            }
            // 3. Otherwise, keep existing entities (Propagates persistence)

            if (data.trace_url) {
                setLatestTraceUrl(data.trace_url);
            }

        } catch (error) {
            console.error('Failed to send message:', error);
            setMessages(prev => [...prev, {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: 'Sorry, I encountered an error. Please try again.',
                timestamp: new Date(),
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        sendMessage(inputValue);
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                audioChunksRef.current.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                const reader = new FileReader();
                reader.onloadend = async () => {
                    const base64Audio = (reader.result as string).split(',')[1];
                    await sendVoiceMessage(base64Audio);
                };
                reader.readAsDataURL(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);
        } catch (error) {
            console.error('Failed to start recording:', error);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const cancelRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.onstop = null; // Prevent sending
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            audioChunksRef.current = [];
        }
    };

    const sendVoiceMessage = async (audioBase64: string) => {
        if (!selectedPatient) return;
        setIsLoading(true);

        try {
            const res = await fetch('/api/voice', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    audio_base64: audioBase64,
                    patient_id: selectedPatient.patient_id,
                    session_id: `session-${selectedPatient.patient_id}`,
                }),
            });

            const data = await res.json();

            if (data.transcript) {
                setMessages(prev => [...prev, {
                    id: Date.now().toString(),
                    role: 'user',
                    content: data.transcript,
                    timestamp: new Date(),
                }]);
            }

            const chatResponse = data.chat_response;
            setMessages(prev => [...prev, {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: chatResponse.message,
                timestamp: new Date(),
                extractedEntities: chatResponse.extracted_entities,
                safetyResult: chatResponse.safety_result,
                orderPreview: chatResponse.order_preview,
                order: chatResponse.order,
                traceUrl: chatResponse.trace_url,
            }]);

            if (data.audio_response_base64) {
                const audio = new Audio(`data:audio/mp3;base64,${data.audio_response_base64}`);
                audio.play();
            }

        } catch (error) {
            console.error('Failed to process voice:', error);
        } finally {
            setIsLoading(false);
        }
    };

    // Mock extracted entities for display
    const displayEntities = currentEntities?.entities?.[0] || null;

    return (
        <div className="flex flex-1 h-full min-h-0">
            {/* Left Panel - Patient Context */}
            <aside className="w-80 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 flex flex-col overflow-y-auto transition-colors duration-300">
                {/* Patient Context Section */}
                <div className="p-6 border-b border-gray-200 dark:border-gray-800">
                    <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 mb-1">
                        <User className="w-4 h-4" />
                        <span className="text-xs font-bold uppercase tracking-wider">Patient Context</span>
                    </div>

                    {/* Patient Dropdown */}
                    <div className="relative mt-2" ref={dropdownRef}>
                        <button
                            onClick={() => setIsPatientDropdownOpen(!isPatientDropdownOpen)}
                            className="w-full flex items-center gap-3 px-3 py-2 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 transition-colors"
                        >
                            <div className="w-8 h-8 rounded-full overflow-hidden bg-gray-200">
                                {selectedPatient ? (
                                    <img
                                        src={`/patients/${selectedPatient.patient_id}.png`}
                                        alt={selectedPatient.patient_name}
                                        className="w-full h-full object-cover"
                                    />
                                ) : (
                                    <span className="w-full h-full flex items-center justify-center text-gray-400 text-xs">?</span>
                                )}
                            </div>
                            <span className="flex-1 text-left text-sm font-medium text-gray-900 dark:text-white">
                                {selectedPatient?.patient_name?.split(' ')[0] || 'Jane'}
                            </span>
                            <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${isPatientDropdownOpen ? 'rotate-180' : ''}`} />
                        </button>

                        {isPatientDropdownOpen && (
                            <div className="absolute top-full left-0 right-0 mt-1 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-lg z-10 overflow-hidden">
                                {patients.map((patient) => (
                                    <button
                                        key={patient.patient_id}
                                        onClick={() => handlePatientChange(patient)}
                                        className={`w-full flex items-center gap-3 px-3 py-2 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors ${selectedPatient?.patient_id === patient.patient_id ? 'bg-indigo-50 dark:bg-indigo-900/30' : ''
                                            }`}
                                    >
                                        <div className="w-8 h-8 rounded-full bg-gray-200 overflow-hidden">
                                            <img
                                                src={`/patients/${patient.patient_id}.png`}
                                                alt={patient.patient_name}
                                                className="w-full h-full object-cover"
                                            />
                                        </div>
                                        <span className="text-sm text-gray-900 dark:text-gray-200">{patient.patient_name}</span>
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {/* Extracted Entities Section */}
                <div className="p-4 border-b border-gray-200 dark:border-gray-800">
                    <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Extracted Entities</h3>

                    <div className="space-y-3">
                        <div>
                            <p className="text-xs text-gray-500 dark:text-gray-400">Medicine:</p>
                            <p className="text-sm font-medium text-gray-900 dark:text-gray-200">
                                {displayEntities?.medicine || 'Metformin 500mg'}
                            </p>
                        </div>
                        <div>
                            <p className="text-xs text-gray-500 dark:text-gray-400">Dosage:</p>
                            <p className="text-sm font-medium text-gray-900 dark:text-gray-200">
                                {displayEntities?.dosage || 'Two tablets daily'}
                            </p>
                        </div>
                        <div>
                            <p className="text-xs text-gray-500 dark:text-gray-400">Quantity:</p>
                            <p className="text-sm font-medium text-gray-900 dark:text-gray-200">
                                {displayEntities?.quantity || '90'} tablets
                            </p>
                        </div>
                        <div>
                            <p className="text-xs text-gray-500 dark:text-gray-400">Supply Duration:</p>
                            <p className="text-sm font-medium text-gray-900 dark:text-gray-200">45 days</p>
                        </div>

                        {/* Badges */}
                        <div className="flex gap-2 mt-2">
                            <span className="px-2 py-1 bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 text-xs font-medium rounded">
                                Prescription: Yes
                            </span>
                            <span className="px-2 py-1 bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300 text-xs font-medium rounded">
                                Stock: OK
                            </span>
                        </div>
                    </div>
                </div>

                {/* Next Check-up */}
                <div className="p-4 border-b border-gray-200 dark:border-gray-800">
                    <div className="flex items-center gap-2 text-orange-500">
                        <Clock className="w-4 h-4" />
                        <span className="text-sm">Next check-up in 14 days</span>
                    </div>
                </div>


            </aside>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col bg-gray-50 dark:bg-gray-950 transition-colors duration-300">
                {/* Chat Header */}
                <div className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 px-6 py-4 transition-colors duration-300">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Conversational Log</h2>
                </div>

                {/* AI Prediction Banner */}
                {/* AI Prediction Banner Removed */}

                {/* Messages */}
                <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
                    {messages.map((message) => (
                        <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-md ${message.role === 'user' ? 'order-1' : ''}`}>
                                {/* Message Bubble */}
                                <div className={`px-4 py-3 rounded-2xl ${message.role === 'user'
                                    ? 'bg-indigo-600 text-white rounded-br-md shadow-md'
                                    : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-800 dark:text-gray-200 rounded-bl-md shadow-sm'
                                    }`}>
                                    <p className="text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: formatMarkdown(message.content) }} />
                                </div>

                                {/* AI Annotation */}
                                {message.role === 'assistant' && message.aiAnnotation && (
                                    <div className="mt-1 flex items-start gap-1 text-xs text-gray-500 dark:text-gray-400">
                                        <Volume2 className="w-3 h-3 mt-0.5" />
                                        <span>{message.aiAnnotation}</span>
                                    </div>
                                )}

                                {/* Badges */}
                                {message.badges && message.badges.length > 0 && (
                                    <div className="mt-2 flex gap-2">
                                        {message.badges.map((badge, idx) => (
                                            <span key={idx} className={`px-2 py-1 text-xs font-medium rounded ${badge.color === 'indigo' ? 'bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300' :
                                                badge.color === 'green' ? 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300' :
                                                    badge.color === 'yellow' ? 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-400' :
                                                        'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                                                }`}>
                                                {badge.label}
                                            </span>
                                        ))}
                                    </div>
                                )}

                                {/* Order Confirmation Card */}
                                {message.orderPreview && (() => {
                                    const subtotal = message.orderPreview.total_amount || 0;
                                    const tax = subtotal * 0.05;
                                    const total = subtotal + tax;
                                    const totalQuantity = message.orderPreview.items?.reduce((sum: number, item: any) => sum + (item.quantity || 0), 0) || 1;
                                    const pricePerUnit = totalQuantity > 0 ? subtotal / totalQuantity : 0;
                                    const requiresPrescription = message.orderPreview.requires_prescription ||
                                        message.orderPreview.items?.some((item: any) => item.prescription_required);
                                    const prescriptionMedicines = message.orderPreview.items?.filter((item: any) => item.prescription_required).map((item: any) => item.medicine_name) || [];

                                    return (
                                        <div className="mt-3 space-y-3 max-w-sm">
                                            {/* Prescription Upload Card - Show if prescription required */}
                                            {requiresPrescription && (
                                                <PrescriptionUploadCard
                                                    medicineName={prescriptionMedicines.join(', ') || message.orderPreview.items?.[0]?.medicine_name || 'This medicine'}
                                                    onUpload={(file) => {
                                                        console.log('Prescription uploaded:', file.name);
                                                        // In a real app, upload to backend and then confirm
                                                        sendMessage('confirm');
                                                    }}
                                                    onSkip={() => sendMessage('cancel')}
                                                />
                                            )}

                                            {/* Order Card with Left Accent Border */}
                                            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border-l-4 border-l-indigo-600 border border-gray-200 dark:border-gray-700 p-4 transition-colors duration-300">
                                                <div className="mb-3">
                                                    <h4 className="font-semibold text-gray-900 dark:text-white">
                                                        {requiresPrescription ? 'Order Summary' : 'Confirm Your Order'}
                                                    </h4>
                                                    <p className="text-xs text-gray-500 dark:text-gray-400">Review details before confirming home delivery</p>
                                                </div>

                                                <p className="text-xs text-gray-400 uppercase tracking-wider mb-2 font-medium">Medicine Details</p>

                                                {message.orderPreview.items?.map((item: any, idx: number) => {
                                                    const itemTotal = item.unit_price ? item.unit_price * item.quantity : 0;
                                                    const supplyDays = item.supply_days || Math.round((item.quantity || 2) * 0.5) || 45;

                                                    return (
                                                        <div key={idx} className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg mb-2">
                                                            <div className="w-10 h-10 rounded-full bg-indigo-100 dark:bg-indigo-900/50 flex items-center justify-center flex-shrink-0">
                                                                <Pill className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                                                            </div>
                                                            <div className="flex-1 min-w-0">
                                                                <div className="flex items-center justify-between gap-2">
                                                                    <p className="font-medium text-gray-900 dark:text-gray-100 text-sm">{item.medicine_name}</p>
                                                                    {item.unit_price && (
                                                                        <p className="text-sm font-semibold text-gray-700 dark:text-gray-200 whitespace-nowrap">
                                                                            ${itemTotal.toFixed(2)}
                                                                        </p>
                                                                    )}
                                                                </div>
                                                                <p className="text-xs text-gray-500 dark:text-gray-400">
                                                                    {item.strength} • {item.quantity} units
                                                                </p>
                                                                <p className="text-xs text-cyan-600 dark:text-cyan-400 mt-1">
                                                                    Supply: ~{supplyDays} days
                                                                </p>
                                                            </div>
                                                        </div>
                                                    );
                                                })}

                                                {/* Price Breakdown */}
                                                <div className="border-t border-gray-100 dark:border-gray-600 pt-3 mt-3">
                                                    <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-2">
                                                        <span>Price per unit</span>
                                                        <span>${pricePerUnit.toFixed(2)}/unit</span>
                                                    </div>
                                                    <div className="flex justify-between items-baseline">
                                                        <div>
                                                            <p className="font-semibold text-gray-900 dark:text-white text-sm">Total Amount</p>
                                                            <p className="text-[10px] text-gray-400 dark:text-gray-500">Inclusive of all taxes</p>
                                                        </div>
                                                        <span className="text-xl font-bold text-indigo-600 dark:text-indigo-400">
                                                            ${total.toFixed(2)}
                                                        </span>
                                                    </div>
                                                </div>

                                                {/* Home Delivery Section */}
                                                <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-700/40 rounded-lg flex items-center gap-3">
                                                    <div className="w-10 h-10 rounded-full bg-indigo-100 dark:bg-indigo-900/50 flex items-center justify-center flex-shrink-0">
                                                        <Truck className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                                                    </div>
                                                    <div>
                                                        <p className="font-medium text-gray-900 dark:text-white text-sm">Home Delivery</p>
                                                        <p className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
                                                            <Clock className="w-3 h-3" />
                                                            Est. Tomorrow by 9:00 PM
                                                        </p>
                                                    </div>
                                                </div>

                                                {/* Show confirm/cancel buttons only if prescription not required */}
                                                {!requiresPrescription && (
                                                    <div className="flex gap-2 mt-4">
                                                        <button
                                                            onClick={() => sendMessage('cancel')}
                                                            className="flex-1 py-2.5 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 text-sm font-medium rounded-lg transition-colors"
                                                        >
                                                            Cancel
                                                        </button>
                                                        <button
                                                            onClick={() => sendMessage('confirm')}
                                                            className="flex-1 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium rounded-lg transition-colors"
                                                        >
                                                            Confirm Order
                                                        </button>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    );
                                })()}

                                {message.order && (
                                    <div className="mt-2 flex gap-2">
                                        <span className="px-2 py-1 bg-indigo-100 text-indigo-700 text-xs font-medium rounded">
                                            Refill placed by AI
                                        </span>
                                        <span className="px-2 py-1 bg-green-100 text-green-700 text-xs font-medium rounded">
                                            Inventory updated
                                        </span>
                                    </div>
                                )}

                                {/* Timestamp */}
                                <p className={`text-xs text-gray-400 mt-1 ${message.role === 'user' ? 'text-right' : ''}`}>
                                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </p>
                            </div>
                        </div>
                    ))}

                    {isLoading && (
                        <div className="flex items-center gap-2 text-gray-500">
                            <div className="flex gap-1">
                                <div className="w-2 h-2 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                <div className="w-2 h-2 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                <div className="w-2 h-2 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                            </div>
                            <span className="text-sm">AI is thinking...</span>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                {/* Input Area or Voice UI */}
                <div className="bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 p-4 transition-colors duration-300 relative">
                    {/* Floating Listening Status */}
                    {isRecording && (
                        <div className="absolute bottom-24 left-1/2 -translate-x-1/2 bg-white dark:bg-gray-800 rounded-full px-6 py-3 shadow-lg border border-gray-100 dark:border-gray-700 flex items-center gap-4 animate-in fade-in slide-in-from-bottom-4 duration-300 z-10">
                            <div className="flex gap-1 items-end h-6">
                                <div className="w-1 bg-indigo-500 rounded-full voice-wave h-3"></div>
                                <div className="w-1 bg-indigo-500 rounded-full voice-wave h-full animation-delay-100"></div>
                                <div className="w-1 bg-indigo-500 rounded-full voice-wave h-4 animation-delay-200"></div>
                                <div className="w-1 bg-indigo-500 rounded-full voice-wave h-2 animation-delay-300"></div>
                            </div>
                            <div className="flex flex-col">
                                <span className="text-indigo-600 dark:text-indigo-400 font-bold text-lg">Listening...</span>
                                <span className="text-gray-500 dark:text-gray-400 text-xs">Speak naturally, AI is listening</span>
                            </div>
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="flex items-center gap-3">
                        {/* Left Actions */}
                        {isRecording ? (
                            <button
                                type="button"
                                onClick={cancelRecording}
                                className="p-2 w-10 h-10 flex items-center justify-center rounded-full bg-red-100 dark:bg-red-900/30 text-red-500 hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors"
                                title="Cancel Recording"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        ) : (
                            <button
                                type="button"
                                className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                            >
                                <Paperclip className="w-5 h-5" />
                            </button>
                        )}

                        {/* Input Field */}
                        <div className="flex-1 relative">
                            <input
                                type="text"
                                value={isRecording ? "" : inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                placeholder={isRecording ? "Listening..." : "Type your message..."}
                                className={`w-full px-4 py-3 rounded-lg border border-gray-200 dark:border-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-sm text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 transition-all ${isRecording ? 'bg-gray-100 dark:bg-gray-800 text-transparent' : 'bg-gray-50 dark:bg-gray-800'}`}
                                disabled={isLoading || isRecording}
                            />
                        </div>

                        {/* Right Actions */}
                        {isRecording ? (
                            <div className="relative group cursor-pointer" onClick={stopRecording}>
                                <div className="absolute inset-0 bg-blue-500 rounded-full opacity-20 animate-ping group-hover:opacity-30"></div>
                                <div className="absolute inset-[-4px] bg-blue-500 rounded-full opacity-10 animate-pulse"></div>
                                <div className="relative z-10 w-14 h-14 bg-gradient-to-b from-blue-500 to-blue-600 rounded-full shadow-xl flex items-center justify-center transition-transform group-active:scale-95 text-white">
                                    <Mic className="w-7 h-7" />
                                </div>
                            </div>
                        ) : (
                            <>
                                <button
                                    type="button"
                                    onClick={startRecording}
                                    className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                                >
                                    <Mic className="w-5 h-5" />
                                </button>

                                <button
                                    type="submit"
                                    disabled={!inputValue.trim() || isLoading}
                                    className="p-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                                >
                                    {isLoading ? (
                                        <Loader2 className="w-5 h-5 animate-spin" />
                                    ) : (
                                        <Send className="w-5 h-5" />
                                    )}
                                </button>
                            </>
                        )}
                    </form>
                </div>
            </div>

            {/* Right Panel - AI Decision Summary & Settings */}
            <aside className="w-80 bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-800 p-0 flex flex-col transition-colors duration-300">


                {/* Chat Settings */}
                <div className="p-4 border-b border-gray-200 dark:border-gray-800">
                    <div className="flex items-center gap-2 mb-3">
                        <Settings className="w-4 h-4 text-gray-500" />
                        <h3 className="text-sm font-semibold text-gray-900">Chat Settings</h3>
                    </div>
                    <p className="text-xs text-gray-500 mb-4">Personalize your chat experience.</p>

                    <div className="space-y-4">
                        {/* Auto-save chats toggle */}
                        <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-700">Auto-save chats</span>
                            <button
                                onClick={() => {
                                    const newState = !autoSaveChats;
                                    setAutoSaveChats(newState);
                                    if (!newState && selectedPatient) {
                                        // Clear all chats if auto-save is disabled
                                        setMessages([]);
                                        localStorage.removeItem(`chat_messages_${selectedPatient.patient_id}`);
                                    }
                                }}
                                className={`w-10 h-6 rounded-full transition-colors ${autoSaveChats ? 'bg-indigo-600' : 'bg-gray-300'
                                    }`}
                            >
                                <div className={`w-4 h-4 bg-white rounded-full shadow transition-transform mx-1 ${autoSaveChats ? 'translate-x-4' : 'translate-x-0'
                                    }`} />
                            </button>
                        </div>

                        {/* Desktop Notifications toggle */}
                        <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-700">Desktop Notifications</span>
                            <button
                                onClick={() => setDesktopNotifications(!desktopNotifications)}
                                className={`w-10 h-6 rounded-full transition-colors ${desktopNotifications ? 'bg-indigo-600' : 'bg-gray-300'
                                    }`}
                            >
                                <div className={`w-4 h-4 bg-white rounded-full shadow transition-transform mx-1 ${desktopNotifications ? 'translate-x-4' : 'translate-x-0'
                                    }`} />
                            </button>
                        </div>

                        {/* Archive Conversation */}
                        <button className="flex items-center gap-2 text-sm text-gray-700 hover:text-gray-900">
                            <ChevronRight className="w-4 h-4" />
                            Archive Conversation
                        </button>
                    </div>
                </div>
            </aside>
        </div>
    );
}
