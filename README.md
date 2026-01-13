# Your Pharma - Agentic AI Pharmacy System 🏥💊

An advanced, multi-agent AI system designed to revolutionize the pharmacy experience. This project demonstrates the power of **Agentic AI** in healthcare, featuring autonomous agents that handle patient interactions, safety checks, inventory fulfillment, and predictive refill management.

![Status](https://img.shields.io/badge/Status-Live-green)
![Tech](https://img.shields.io/badge/Tech-FastAPI%20%7C%20Next.js%20%7C%20OpenAI%20%7C%20LangSmith-blue)

## 🚀 Features

*   **🤖 Multi-Agent Orchestration**: A central `OrchestratorAgent` coordinates specialized agents for seamless operations.
*   **🗣️ Conversational Ordering**: `ExtractionAgent` parses natural language orders (text & voice) into structured data.
*   **🛡️ Safety & Policy Enforcement**: `SafetyAgent` rigorously checks prescriptions, interactions, and dosage limits before approval.
*   **📦 Automated Fulfillment**: `FulfillmentAgent` manages inventory, creates orders, and triggers warehouse webhooks.
*   **🔮 Predictive Refills**: `RefillAgent` analyzes patient history to proactively suggest refills before medication runs out.
*   **📊 Full Observability**: Integrated with **LangSmith** for deep tracing of agent reasoning, decision spans, and latency.

## 🛠️ Tech Stack

### Backend
*   **Framework**: FastAPI (Python)
*   **AI Models**: OpenAI GPT-4o-mini / GPT-5.2 (Simulated)
*   **Orchestration**: Custom Agent Framework with function calling
*   **Observability**: LangSmith (@traceable)

### Frontend
*   **Framework**: Next.js (TypeScript)
*   **Styling**: Tailwind CSS, Lucide Icons
*   **UI Components**: Custom Dashboard, Chat Interface, Real-time Status Board


## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Frontend ["FRONTEND (Next.js)"]
        A[Chat Page]
        B[Admin Dashboard]
        C[Refills Page]
        D[Orders Page]
    end

    subgraph Backend ["BACKEND (FastAPI)"]
        O[ORCHESTRATOR AGENT (GPT-5.2)]
        
        subgraph Agents
            E[Extraction Agent]
            S[Safety Agent]
            R[Refill Agent]
            F[Fulfillment Agent]
        end
        
        subgraph Services
            DS[Data Service]
            VS[Voice Service]
            L[LangSmith Tracing]
        end
    end

    Frontend --> |REST API| O
    O --> E
    O --> S
    O --> R
    O --> F
```

```ascii
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │   Chat   │  │  Admin   │  │  Refills │  │  Orders  │         │
│  │   Page   │  │Dashboard │  │   Page   │  │   Page   │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        │             │             │             │
        └─────────────┴──────┬──────┴─────────────┘
                             │ REST API
┌────────────────────────────┼────────────────────────────────────┐
│                    BACKEND (FastAPI)                            │
│  ┌─────────────────────────┴─────────────────────────┐          │
│  │              ORCHESTRATOR AGENT (GPT-5.2)          │         │
│  │         Coordinates all agents & maintains state   │         │
│  └──────┬──────────┬──────────┬──────────┬───────────┘          │
│         │          │          │          │                      │
│  ┌──────┴───┐ ┌────┴────┐ ┌───┴────┐ ┌───┴──────┐               │
│  │Extraction│ │ Safety  │ │ Refill │ │Fulfillment│              │
│  │  Agent   │ │  Agent  │ │ Agent  │ │  Agent   │               │
│  │gpt-5-mini│ │ gpt-5.2 │ │gpt-5.2 │ │gpt-5-mini│               │
│  └──────────┘ └─────────┘ └────────┘ └──────────┘               │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Data Service│  │Voice Service│  │  LangSmith  │              │
│  │  (CSV/Excel)│  │ (STT/TTS)   │  │   Tracing   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## 📂 Project Structure

```
Your-Pharma/
├── backend/
│   ├── agents/           # Specialized AI Agents (Orchestrator, Safety, Refill, etc.)
│   ├── services/         # Core business logic (Data, Voice)
│   ├── utils/            # Tracing and shared utilities
│   └── main.py           # FastAPI entry point
└── frontend/
    ├── app/              # Next.js App Router pages
    └── components/       # Reusable UI components
```

## ⚡ Getting Started

### Prerequisites
*   Python 3.10+
*   Node.js 18+
*   OpenAI API Key
*   LangSmith API Key (Optional, for tracing)

### Backend Setup
1.  Navigate to `backend`:
    ```bash
    cd backend
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Create a `.env` file with your keys:
    ```bash
    OPENAI_API_KEY=sk-...
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=lsv2_...
    ```
4.  Run the server:
    ```bash
    uvicorn main:app --reload
    ```

### Frontend Setup
1.  Navigate to `frontend`:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Run the development server:
    ```bash
    npm run dev
    ```

## 🔒 Security Note
This repository is configured to exclude sensitive files like `.env`. **Do not commit your API keys.**

## 📄 License
MIT
