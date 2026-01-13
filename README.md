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

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **Node.js 18+**
- **OpenAI API Key** (for GPT-5-mini and GPT-5.2 agents)
- **Langchain API Keys** (for observability tracing)

  
## 🛠️ Tech Stack

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create .env file with your keys
copy .env.example .env
# Edit .env and add your API keys

# Run the server
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### 3. Open the Application

Navigate to **http://localhost:3000**

---

## 🏗️ System Architecture

```
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
│  │ Data Service│  │Voice Service│  │   Langfuse  │              │
│  │  (CSV/Excel)│  │ (STT/TTS)   │  │   Tracing   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Agent Architecture

| Agent | Model | Purpose |
|-------|-------|---------|
| **Extraction Agent** | gpt-5-mini | Parse natural language to extract medicine, dosage, quantity |
| **Safety Agent** | gpt-5.2 | High-stakes policy enforcement, prescription validation |
| **Refill Agent** | gpt-5.2 | Predict refill needs, trigger reminders/auto-refills |
| **Fulfillment Agent** | gpt-5-mini | Execute orders, update inventory, trigger webhooks |
| **Orchestrator Agent** | gpt-5.2 | Coordinate all agents, route intents, manage state |

### Voice Support

| Feature | Model |
|---------|-------|
| Speech-to-Text | gpt-4o-mini-transcribe |
| Text-to-Speech | gpt-4o-mini-tts |

---

## 📦 Features

### 1. Conversational Chat Interface
- Natural language medicine ordering
- Voice input with live transcription
- Real-time extracted entities panel
- AI decision summary (Approve/Reject/Conditional)
- Order preview with confirm/cancel

### 2. Admin Inventory Dashboard
- Real-time stock levels
- Out-of-stock/low-stock alerts
- Prescription required badges
- Controlled substance flags
- AI trace links (read-only)

### 3. Proactive Refill Alerts
- Predictive refill intelligence
- Auto-refill eligibility
- Reminder scheduling
- Block conditions (discontinued, expired Rx)

### 4. Order Tracking
- Full order lifecycle visibility
- Status timeline (Pending → Confirmed → Preparing → Processing → Delivered)
- Payment status (mock)
- Fulfillment details

---

## 📊 Observability (Langfuse)

Every agent call is traced:
- Agent inputs/outputs
- Model used, tokens consumed
- Latency metrics
- Agent-to-agent handoffs
- Final decisions

**Trace links** are exposed in the UI header for each conversation.

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Process chat message through agents |
| `/api/voice` | POST | Process voice input (STT → Agents → TTS) |
| `/api/patients` | GET | List all patients |
| `/api/orders` | GET | List all orders |
| `/api/orders/{id}/confirm` | POST | Confirm pending order |
| `/api/inventory` | GET | List all medicines |
| `/api/inventory/stats` | GET | Inventory statistics |
| `/api/refills` | GET | Proactive refill alerts |
| `/api/webhook/warehouse` | POST | Mock warehouse fulfillment |
| `/api/agents/status` | GET | Agent status overview |

---

## 📁 Project Structure

```
final demo/
├── backend/
│   ├── agents/
│   │   ├── extraction_agent.py    # gpt-5-mini
│   │   ├── safety_agent.py        # gpt-5.2
│   │   ├── refill_agent.py        # gpt-5.2
│   │   ├── fulfillment_agent.py   # gpt-5-mini
│   │   └── orchestrator_agent.py  # gpt-5.2
│   ├── data/
│   │   ├── medicine_master.csv    # 40 medicines
│   │   └── order_history.csv      # 20 orders, 5 patients
│   ├── models/
│   │   └── schemas.py             # Pydantic models
│   ├── services/
│   │   ├── data_service.py        # CSV data access
│   │   └── voice_service.py       # STT/TTS
│   ├── main.py                    # FastAPI app
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx               # Chat interface
│   │   ├── admin/page.tsx         # Inventory dashboard
│   │   ├── refills/page.tsx       # Refill alerts
│   │   └── orders/page.tsx        # Order management
│   ├── components/
│   │   ├── Sidebar.tsx
│   │   ├── PatientSelector.tsx
│   │   ├── ChatMessage.tsx
│   │   ├── ExtractedEntitiesPanel.tsx
│   │   ├── SafetyDecisionPanel.tsx
│   │   └── OrderPreviewCard.tsx
│   ├── package.json
│   └── tailwind.config.js
│
└── README.md
```

---

## 🔐 Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Required
OPENAI_API_KEY=sk-...

# langsmith (for observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2-...
LANGCHAIN_PROJECT=pharma ai


# Optional
WAREHOUSE_WEBHOOK_URL=http://localhost:8000/api/webhook/warehouse
```

---

## 🧪 Testing the System

### Complete Order Flow Test

1. Open http://localhost:3000
2. Select patient **"John Doe"** from dropdown
3. Type: `"I need Metformin 500mg, 60 tablets"`
4. Observe:
   - Extracted entities appear in right panel
   - Safety decision shows APPROVE
   - Order preview card appears
5. Click **"Confirm Order"**
6. Observe:
   - Order confirmed message
   - Receipt displayed inline
   - Inventory updated

### Prescription Enforcement Test

1. Type: `"I need Morphine 10mg"`
2. Observe:
   - Safety Agent blocks (controlled substance)
   - Decision shows REJECT with reasons

### Voice Test

1. Click the microphone button
2. Say: "I need my blood pressure medication refilled"
3. Observe transcription and AI response

---

## 📈 Order Lifecycle

```
User Request
    │
    ▼
┌─────────────────┐
│ Extraction Agent│ → Parse message, extract entities
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Safety Agent   │ → Check prescriptions, validate quantities
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Inventory Check │ → Verify stock availability
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Order Preview  │ → Show preview in UI
└────────┬────────┘
         │ (User Confirms)
         ▼
┌─────────────────┐
│Fulfillment Agent│ → Create order, update stock
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Warehouse Webhook│ → Trigger mock fulfillment
└────────┬────────┘
         │
         ▼
   Order Complete
```

---

## 🎯 Key Design Decisions

1. **No Hardcoded Logic**: All decisions come from AI agents
2. **CSV-Based Data**: Easy to modify without database setup
3. **Agent Isolation**: Each agent has single responsibility
4. **Full Traceability**: Every decision can be traced in Langfuse
5. **Real State Changes**: Orders persist, inventory updates

---

## 📝 License

MIT License - Built for demonstration purposes.
