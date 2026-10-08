# AegisOT AI

> AI & Agentic Intelligence Layer for the AegisOT Smart Factory Graduation Project.

AegisOT is a smart-factory platform that combines **Industrial IoT, Machine Learning, Agentic AI, Network Security, and Digital Twin technology**.

This repository contains the **AI team's work**, including:

- Sensor data processing
- Feature engineering
- Anomaly detection
- ML model training and inference
- Agentic AI diagnosis
- LangGraph workflows
- AI tools
- FastAPI integration
- AI evaluation and testing

---

## 📌 Project Architecture

The AI system follows this general flow:

```text
        Sensors / IoT Devices
                │
                ▼
          Sensor Telemetry
                │
                ▼
        Data Preprocessing
                │
                ▼
        Feature Engineering
                │
                ▼
       ┌──────────────────┐
       │   ML Detection   │
       │                  │
       │ Anomaly / Normal │
       └────────┬─────────┘
                │
          Anomaly Event
                │
                ▼
       ┌──────────────────┐
       │   Agentic AI     │
       │                  │
       │ Monitoring       │
       │ Network          │
       │ Security         │
       │ Diagnosis        │
       │ Maintenance      │
       └────────┬─────────┘
                │
                ▼
            Diagnosis
                │
                ▼
             FastAPI
                │
                ▼
            Laravel
                │
                ▼
          AegisOT Platform
```

---

## 📁 Project Structure

```text
aegisot-ai/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_anomaly_detection.ipynb
│   └── 03_agent_testing.ipynb
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── features.py
│   │
│   ├── ml/
│   │   ├── baseline.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── agents/
│   │   ├── agents.py
│   │   ├── graph.py
│   │   └── tools.py
│   │
│   └── api/
│       └── main.py
│
├── models/
│   └── anomaly_model.pkl
│
├── tests/
│   ├── test_ml.py
│   └── test_agents.py
│
└── configs/
    └── config.yaml
```

---

## 📂 Folder & File Explanation

### `data/`

Contains datasets used by the AI system.

```text
data/
├── raw/
└── processed/
```

#### `raw/`

Original data. Examples:

```text
sensor_data.csv
machine_data.csv
network_data.csv
```

> **Do not modify the original raw data.**

#### `processed/`

Cleaned and prepared data used by the ML pipeline. Examples:

```text
clean_sensor_data.csv
features.csv
train.csv
test.csv
```

---

### 📓 `notebooks/`

Jupyter notebooks are used for experimentation and exploration.

#### `01_data_analysis.ipynb`

Used for:

- Understanding the dataset
- Checking missing values
- Visualization
- Statistical analysis
- Understanding sensor distributions

#### `02_anomaly_detection.ipynb`

Used for:

- Feature experimentation
- Testing algorithms
- Comparing models
- Visualizing anomalies
- Finding suitable thresholds

#### `03_agent_testing.ipynb`

Used for:

- Testing LLM behavior
- Testing prompts
- Testing agent outputs
- Testing diagnosis scenarios

> **Important:** Notebooks are for experimentation. Once code becomes part of the actual system, move it into `src/`.

---

### 🧹 `src/data/`

Responsible for preparing sensor data.

#### `preprocessing.py`

Contains data preprocessing logic. Examples:

- Missing values
- Duplicate records
- Invalid values
- Scaling
- Normalization
- Timestamp processing

```python
def preprocess_data(df):
    ...
```

#### `features.py`

Contains feature engineering. Examples:

- Temperature mean
- Temperature std
- Vibration RMS
- Vibration variance
- Current mean
- RPM statistics
- Rolling statistics

```python
def create_features(df):
    ...
```

The goal is to transform raw telemetry into useful ML features.

---

### 🤖 `src/ml/`

Contains the Machine Learning part of AegisOT.

```text
ml/
├── baseline.py
├── train.py
└── predict.py
```

#### `baseline.py`

Contains the simplest detection approach. Examples:

- Temperature > threshold
- Vibration > threshold
- Current > threshold
- RPM outside expected range

The baseline is important because we need something simple to compare our ML models against.

```python
def detect_anomaly(features):
    ...
```

#### `train.py`

Responsible for training the ML model. Example workflow:

```text
Load data
    ↓
Preprocess
    ↓
Create features
    ↓
Train model
    ↓
Evaluate
    ↓
Save model
```

```bash
python -m src.ml.train
```

#### `predict.py`

Responsible for inference. It receives new telemetry/features and returns something like:

```json
{
  "is_anomaly": true,
  "score": 0.91,
  "model": "isolation_forest"
}
```

```bash
python -m src.ml.predict
```

---

### 🧠 `src/agents/`

Contains the Agentic AI system.

```text
agents/
├── agents.py
├── graph.py
└── tools.py
```

#### `agents.py`

Contains the different AI agents. Possible agents:

- Monitoring Agent
- Network Agent
- Security Agent
- Diagnosis Agent
- Maintenance Agent

Each agent should have a clear responsibility:

```text
Monitoring Agent
    ↓
Analyze telemetry

Network Agent
    ↓
Analyze connectivity

Security Agent
    ↓
Analyze identity/security events

Diagnosis Agent
    ↓
Combine evidence and determine likely cause

Maintenance Agent
    ↓
Recommend maintenance action
```

> **Agents should not have unrestricted access to the system.**

#### 🔗 `graph.py`

Contains the LangGraph workflow.

```text
Incident
   │
   ▼
Monitoring Agent
   │
   ▼
Network Agent
   │
   ▼
Security Agent
   │
   ▼
Diagnosis Agent
   │
   ▼
Maintenance Agent
   │
   ▼
Final Result
```

The graph controls:

- Agent execution
- State
- Routing
- Conditions
- Final result

#### 🛠️ `tools.py`

Contains the tools available to the agents. Examples:

```python
get_sensor_data()
get_machine_status()
get_network_status()
get_security_events()
get_machine_history()
```

**Important Security Rule**

Agents must **not** directly execute arbitrary:

- Shell commands
- SQL queries
- Router commands
- Device commands
- Network configuration

Tools must be explicitly defined and controlled.

---

### 🌐 `src/api/`

Contains the FastAPI integration layer.

#### `main.py`

Starts the AI API.

```bash
uvicorn src.api.main:app --reload
```

The API will be used by the AegisOT backend. Possible endpoints:

```text
GET  /health

POST /api/v1/anomaly/predict

POST /api/v1/diagnosis/analyze

POST /api/v1/incident/analyze
```

The exact API contract should be agreed upon with the Backend team.

---

### 🧠 `models/`

Contains trained ML models.

```text
models/
└── anomaly_model.pkl
```

Other possible models:

```text
isolation_forest.pkl
autoencoder.pt
classifier.pkl
```

> **Important:** Do not commit very large model files to Git. For large models, use an appropriate artifact/model storage solution.

---

### 🧪 `tests/`

Contains automated tests.

```text
tests/
├── test_ml.py
└── test_agents.py
```

#### `test_ml.py`

Tests:

- Preprocessing
- Feature engineering
- Model prediction
- Anomaly detection

#### `test_agents.py`

Tests:

- Agent outputs
- Graph workflow
- Tool behavior
- Invalid inputs
- Diagnosis results

Run:

```bash
pytest
```

---

### ⚙️ `configs/`

Contains project configuration.

```text
configs/
└── config.yaml
```

Example:

```yaml
model:
  name: isolation_forest
  threshold: 0.5

api:
  host: 0.0.0.0
  port: 8000
```

> Do not hardcode configuration values throughout the code.

---

## 🔐 Environment Variables

Create your local environment file:

```bash
cp .env.example .env
```

Example:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key

API_HOST=0.0.0.0
API_PORT=8000

BACKEND_URL=http://localhost:8000
```

> **Never commit `.env`.** The `.env` file should be included in `.gitignore`.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd aegisot-ai
```

### 🐍 2. Create Python Environment

We recommend **Python 3.10+**.

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / WSL / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your terminal.

### 📦 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install requirements:

```bash
pip install -r requirements.txt
```

### 🔑 4. Configure Environment

```bash
cp .env.example .env
```

Windows CMD:

```bat
copy .env.example .env
```

Then add the required API keys/configuration.

---

## ▶️ Running the Project

### Run the FastAPI server

```bash
uvicorn src.api.main:app --reload
```

Open: <http://localhost:8000>

FastAPI documentation: <http://localhost:8000/docs>

### 🤖 Train the ML Model

Put the dataset inside `data/raw/`, then run:

```bash
python -m src.ml.train
```

The trained model should be saved inside `models/`.

### 🔍 Run Prediction

After training:

```bash
python -m src.ml.predict
```

Expected output:

```text
Prediction:
Normal

Anomaly Score:
0.12
```

or:

```text
Prediction:
Anomaly

Anomaly Score:
0.91
```

### 🧠 Run Agent Workflow

After configuring the LLM:

```bash
python -m src.agents.graph
```

The workflow should process an incident approximately like:

```text
Telemetry
    ↓
Anomaly detected
    ↓
Create incident
    ↓
Monitoring Agent
    ↓
Network Agent
    ↓
Security Agent
    ↓
Diagnosis Agent
    ↓
Maintenance Agent
    ↓
Final diagnosis
```

## 🔄 Complete AI Pipeline

The final AI workflow should look like:

```text
                SENSOR DATA
                     │
                     ▼
              PREPROCESSING
                     │
                     ▼
            FEATURE ENGINEERING
                     │
                     ▼
              ML DETECTION
                     │
             ┌───────┴───────┐
             │               │
          NORMAL          ANOMALY
             │               │
             │               ▼
             │          CREATE INCIDENT
             │               │
             │               ▼
             │        AGENTIC WORKFLOW
             │               │
             │       ┌───────┼────────┐
             │       ▼       ▼        ▼
             │   Network  Security  Monitoring
             │       │       │        │
             │       └───────┼────────┘
             │               ▼
             │          DIAGNOSIS
             │               │
             │               ▼
             │         RECOMMENDATION
             │               │
             └───────────────┴─────────────►
                              │
                              ▼
                           FASTAPI
                              │
                              ▼
                           LARAVEL
```

---

## 📊 ML Evaluation

The ML team should evaluate models using appropriate metrics. Important metrics include:

- Precision
- Recall
- F1 Score
- False Alarm Rate
- Detection Delay

> For anomaly detection, don't rely only on accuracy.

Example report:

```text
Model: Isolation Forest

Precision:       0.91
Recall:          0.87
F1 Score:        0.89
False Alarm:     4.2%
Detection Delay: 1.8 sec
```

---

## 🧪 Testing Scenarios

The AI team should test reproducible scenarios such as:

1. Normal operation
2. High temperature
3. Abnormal vibration
4. Abnormal motor current
5. RPM anomaly
6. Sensor failure
7. Network failure
8. Multiple simultaneous problems

Each scenario should have:

```text
Input
↓
Expected behavior
↓
Actual result
↓
Evaluation
```

---

## 🌿 Git Workflow

Do not work directly on `main`. Create a branch:

```bash
git checkout -b feature/anomaly-detection
```

Branch name examples:

```text
feature/data-preprocessing
feature/anomaly-model
feature/langgraph-agents
feature/diagnosis-agent
feature/fastapi
feature/agent-tools
```

After finishing:

```bash
git add .
git commit -m "Add anomaly detection model"
git push origin feature/anomaly-model
```

Then create a Pull Request.

### 📝 Commit Convention

Use clear commit messages. Examples:

```text
feat: add anomaly detection model
feat: add diagnosis agent
feat: add FastAPI prediction endpoint

fix: handle missing sensor values
fix: fix agent state transition

refactor: improve feature engineering

test: add anomaly detection tests

docs: update README
```

### 🚫 What Should NOT Be Committed

- `.env`
- API keys
- Passwords
- Large datasets
- Large model files
- Temporary files
- Python cache

The `.gitignore` should handle most of these.

---

## 🤝 Working With Other AegisOT Teams

The AI team does not own the whole AegisOT system. Our main integration points are:

```text
                    AegisOT
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
   Hardware         Backend          Network
       │               │                │
       │               │                │
       └───────────────┼────────────────┘
                       │
                       ▼
                    AI TEAM
                       │
              ┌────────┴────────┐
              ▼                 ▼
             ML              Agents
              │                 │
              └────────┬────────┘
                       ▼
                    FastAPI
                       │
                       ▼
                    Backend
```

The AI team should agree with the Backend team on:

- Request format
- Response format
- Asset/device ID
- Incident ID
- Telemetry format
- Anomaly result format
- Diagnosis result format

### 📦 AI Result Example

A model result can follow a structure similar to:

```json
{
  "asset_id": "motor_01",
  "incident_id": "INC-001",
  "is_anomaly": true,
  "anomaly_score": 0.91,
  "model": "isolation_forest",
  "timestamp": "2026-10-01T12:30:00Z"
}
```

Agent diagnosis can return:

```json
{
  "incident_id": "INC-001",
  "diagnosis": "Possible motor overload",
  "confidence": 0.87,
  "evidence": ["High motor current", "Abnormal vibration", "RPM decrease"],
  "recommended_action": "Inspect motor load",
  "approval_required": true
}
```

> The exact production schema should be agreed upon with the Backend team.

---

## 🔒 AI Safety Rules

AegisOT is an OT/industrial project. Therefore:

**Agents must NOT:**

- Execute arbitrary commands
- Access unrestricted databases
- Modify network configuration directly
- Stop machines without authorization
- Execute unknown tools

**Agents SHOULD:**

- Use allowlisted tools
- Return structured outputs
- Provide evidence
- Provide confidence
- Recommend actions
- Require approval for sensitive actions

The AI system should **recommend** actions; authorization and bounded execution belong to the controlled AegisOT workflow.

## 🚀 Welcome to the AegisOT AI Team

**Build → Test → Integrate → Evaluate → Document**

Keep the AI system simple, measurable, safe, and integrated with the rest of AegisOT.
