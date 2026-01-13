'use client';

import { User, ChevronDown } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';

interface Patient {
    patient_id: string;
    patient_name: string;
    patient_email: string;
    patient_phone: string;
}

interface PatientSelectorProps {
    patients: Patient[];
    selectedPatient: Patient | null;
    onSelect: (patient: Patient) => void;
}

export default function PatientSelector({ patients, selectedPatient, onSelect }: PatientSelectorProps) {
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <div className="relative" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-3 px-4 py-2 bg-white rounded-lg border border-gray-200 hover:border-primary-300 transition-colors min-w-[200px]"
            >
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-400 to-purple-500 flex items-center justify-center">
                    <User className="w-4 h-4 text-white" />
                </div>
                <div className="flex-1 text-left">
                    <p className="text-sm font-medium text-gray-900">
                        {selectedPatient?.patient_name || 'Select Patient'}
                    </p>
                    <p className="text-xs text-gray-500">
                        {selectedPatient?.patient_id || 'No patient selected'}
                    </p>
                </div>
                <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>

            {isOpen && (
                <div className="absolute top-full left-0 right-0 mt-2 bg-white rounded-lg border border-gray-200 shadow-lg z-50 overflow-hidden">
                    <div className="p-2">
                        <p className="text-xs text-gray-500 px-2 py-1">Select Patient Context</p>
                    </div>
                    <div className="max-h-60 overflow-y-auto">
                        {patients.map((patient) => (
                            <button
                                key={patient.patient_id}
                                onClick={() => {
                                    onSelect(patient);
                                    setIsOpen(false);
                                }}
                                className={`w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-50 transition-colors ${selectedPatient?.patient_id === patient.patient_id ? 'bg-primary-50' : ''
                                    }`}
                            >
                                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-400 to-purple-500 flex items-center justify-center">
                                    <span className="text-white text-sm font-medium">
                                        {patient.patient_name.charAt(0)}
                                    </span>
                                </div>
                                <div className="text-left">
                                    <p className="text-sm font-medium text-gray-900">{patient.patient_name}</p>
                                    <p className="text-xs text-gray-500">{patient.patient_email}</p>
                                </div>
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
