'use client';

import { Bot, User } from 'lucide-react';

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
}

interface ChatMessageProps {
    message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
    const isUser = message.role === 'user';

    // Format message content with markdown-like styling
    const formatContent = (content: string) => {
        // Handle bold text
        let formatted = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Handle bullet points
        formatted = formatted.replace(/^• /gm, '<span class="text-primary-500 mr-2">•</span>');
        // Handle line breaks
        formatted = formatted.replace(/\n/g, '<br />');
        return formatted;
    };

    return (
        <div className={`flex gap-3 chat-bubble ${isUser ? 'flex-row-reverse' : ''}`}>
            {/* Avatar */}
            <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${isUser
                    ? 'bg-gradient-to-br from-purple-500 to-pink-500'
                    : 'bg-gradient-to-br from-primary-500 to-primary-600'
                }`}>
                {isUser ? (
                    <User className="w-5 h-5 text-white" />
                ) : (
                    <Bot className="w-5 h-5 text-white" />
                )}
            </div>

            {/* Message Bubble */}
            <div className={`max-w-2xl ${isUser ? 'text-right' : ''}`}>
                <div className={`inline-block px-4 py-3 rounded-2xl ${isUser
                        ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-tr-md'
                        : 'bg-white shadow-md border border-gray-100 text-gray-800 rounded-tl-md'
                    }`}>
                    <div
                        className="text-sm leading-relaxed"
                        dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
                    />
                </div>

                {/* Timestamp */}
                <p className={`text-xs text-gray-400 mt-1 ${isUser ? 'text-right' : ''}`}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>

                {/* Order Confirmation Badge */}
                {message.order && (
                    <div className="mt-2 inline-flex items-center gap-2 px-3 py-1 bg-green-50 text-green-700 rounded-full text-xs font-medium">
                        <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                        Order {message.order.order_id} Confirmed
                    </div>
                )}
            </div>
        </div>
    );
}
