# AutoMeet

Multi-agent system for meeting insights and project management using Ollama and QwQ.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install and start Ollama:
```bash
# Install Ollama (Mac/Linux)
curl https://ollama.ai/install.sh | sh
# Start Ollama
ollama serve
```

3. Pull QwQ model:
```bash
ollama pull qwq
```

## Usage

1. Place meeting transcripts in Markdown format in the `transcripts/` directory
2. Run the multi-agent system:
```bash
python automeet/multi_agent.py
```

## Directory Structure

```
automeet/
├── transcripts/         # Raw meeting transcripts (.md)
├── meeting_analysis/    # Extracted insights (.json)
└── project_summaries/   # Aggregated project updates (.md)
```

## Features

- Transcript parsing by speaker
- Insight extraction (tasks, decisions, risks, etc.)
- Automatic workstream classification
- Chronological reconciliation across meetings
- Project status aggregation
- Structured data using Pydantic models
