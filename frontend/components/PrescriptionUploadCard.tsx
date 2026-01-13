'use client';

import { useState, useRef } from 'react';
import { Upload, FileText, X, CheckCircle, AlertCircle } from 'lucide-react';

interface PrescriptionUploadCardProps {
    medicineName: string;
    onUpload: (file: File) => void;
    onSkip: () => void;
}

export default function PrescriptionUploadCard({ medicineName, onUpload, onSkip }: PrescriptionUploadCardProps) {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [uploaded, setUploaded] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedFile(file);
        }
    };

    const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        const file = event.dataTransfer.files[0];
        if (file && (file.type.startsWith('image/') || file.type === 'application/pdf')) {
            setSelectedFile(file);
        }
    };

    const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault();
    };

    const handleUpload = async () => {
        if (!selectedFile) return;

        setUploading(true);
        // Simulate upload delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        setUploading(false);
        setUploaded(true);
        onUpload(selectedFile);
    };

    const handleRemoveFile = () => {
        setSelectedFile(null);
        setUploaded(false);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <div className="mt-3 bg-white rounded-xl shadow-lg border border-amber-200 p-4 max-w-sm">
            <div className="flex items-start gap-3 mb-3">
                <div className="w-10 h-10 rounded-full bg-amber-100 flex items-center justify-center flex-shrink-0">
                    <AlertCircle className="w-5 h-5 text-amber-600" />
                </div>
                <div>
                    <h4 className="font-semibold text-gray-900">Prescription Required</h4>
                    <p className="text-sm text-gray-500">
                        <span className="font-medium text-amber-700">{medicineName}</span> requires a valid prescription
                    </p>
                </div>
            </div>

            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                accept="image/*,.pdf"
                className="hidden"
            />

            {!selectedFile && !uploaded && (
                <div
                    onClick={() => fileInputRef.current?.click()}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-indigo-400 hover:bg-indigo-50 transition-colors"
                >
                    <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                    <p className="text-sm text-gray-600">Click to upload or drag & drop</p>
                    <p className="text-xs text-gray-400 mt-1">JPG, PNG, or PDF (max 5MB)</p>
                </div>
            )}

            {selectedFile && !uploaded && (
                <div className="bg-gray-50 rounded-lg p-3 flex items-center gap-3">
                    <FileText className="w-8 h-8 text-indigo-500" />
                    <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">{selectedFile.name}</p>
                        <p className="text-xs text-gray-500">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                    </div>
                    <button
                        onClick={handleRemoveFile}
                        className="p-1 hover:bg-gray-200 rounded-full transition-colors"
                    >
                        <X className="w-4 h-4 text-gray-500" />
                    </button>
                </div>
            )}

            {uploaded && (
                <div className="bg-green-50 rounded-lg p-3 flex items-center gap-3">
                    <CheckCircle className="w-8 h-8 text-green-500" />
                    <div className="flex-1">
                        <p className="text-sm font-medium text-green-700">Prescription Uploaded</p>
                        <p className="text-xs text-green-600">Your prescription has been verified</p>
                    </div>
                </div>
            )}

            <div className="flex gap-2 mt-4">
                {!uploaded ? (
                    <>
                        <button
                            onClick={handleUpload}
                            disabled={!selectedFile || uploading}
                            className={`flex-1 py-2 text-sm font-medium rounded-lg transition-colors ${selectedFile && !uploading
                                    ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
                                    : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                                }`}
                        >
                            {uploading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                                    Uploading...
                                </span>
                            ) : (
                                'Upload Prescription'
                            )}
                        </button>
                        <button
                            onClick={onSkip}
                            className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium rounded-lg transition-colors"
                        >
                            Skip
                        </button>
                    </>
                ) : (
                    <button
                        onClick={() => onUpload(selectedFile!)}
                        className="flex-1 py-2 bg-green-600 hover:bg-green-700 text-white text-sm font-medium rounded-lg transition-colors"
                    >
                        Continue with Order
                    </button>
                )}
            </div>
        </div>
    );
}
