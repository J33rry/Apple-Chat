# Apple-Chat: AI Support Agent for AppleSupport

A personal project exploring how to build a grounded, retrieval-augmented AI agent for customer support. This system handles Twitter-style support queries directed at **AppleSupport** — it classifies intent, drafts replies grounded in historical knowledge via RAG, and decides whether an issue should be escalated to a human.

## Setup Instructions

1. **Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # macOS/Linux
   # venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

2. **Download the Dataset:**
   ```bash
   python setup_data.py
   ```
   This script automatically downloads the [TWCS dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) from Kaggle if `data/twcs.csv` is not already present. It tries `kagglehub` first, then the `kaggle` CLI, and falls back to manual download instructions if neither is available.

   > To use the automatic download, install kagglehub: `pip install kagglehub`

3. **Set the Gemini API Key:**
   ```bash
   export GEMINI_API_KEY="your-api-key"       # macOS/Linux
   # $env:GEMINI_API_KEY="your-api-key"       # Windows PowerShell
   ```

## LLM Provider

The project requires an LLM for intent classification, reply drafting, escalation decisions, and evaluation judging.

- **Gemini API (recommended):** Set the `GEMINI_API_KEY` environment variable. When present, the system automatically uses `gemini-2.0-flash` for all LLM calls. This produces the best results for both the agent and the LLM-as-judge evaluator.
- **Local LLM (fallback):** If no API key is set, the system falls back to a local OpenAI-compatible endpoint at `http://127.0.0.1:1234/v1/chat/completions` using the `google/gemma-4-e4b` model. You can run this via [LM Studio](https://lmstudio.ai/) or [Ollama](https://ollama.com/) — just pull the model and start the server before running the scripts.

> **Tip:** The Gemini API key is needed for best results, but you can run the entire pipeline locally by pulling `google/gemma-4-e4b` in LM Studio or Ollama without any API key.

## Reproducing the Results

```bash
# 1. Parse dataset and build knowledge base
python data_pipeline.py

# 2. Auto-label the golden evaluation set (hybrid LLM + heuristic)
python label_golden_set.py

# 3. Evaluate Baselines vs RAG Agent
python evaluate.py
```

This will process the data and output comparative evaluation metrics (LLM judge scores, ROUGE, intent/escalation accuracy) in `data/evaluation_results.json`.

## Evaluation Results

| Metric | RAG Agent | Simple Baseline | Trivial Baseline |
|--------|:---------:|:---------------:|:----------------:|
| Judge Score (1-5) | **3.25** | 2.55 | 2.00 |
| Intent Accuracy | **90.0%** | 90.0% | N/A |
| Escalation Accuracy | **61.0%** | 59.0% | N/A |
| Escalation Rate | 40.0% | 42.0% | 100.0% |
| ROUGE-1 | 0.1465 | 0.0808 | 0.1847 |
| ROUGE-L | 0.1038 | 0.0590 | 0.1386 |

*Evaluated on 100 golden-set queries using `google/gemma-4-e4b` as both the agent and judge.*

## Project Structure

| File | Description |
|------|-------------|
| `agent.py` | LLM provider abstraction + SupportAgent with TF-IDF RAG |
| `data_pipeline.py` | Extracts AppleSupport conversation threads from TWCS CSV |
| `label_golden_set.py` | Hybrid (LLM + heuristic) labeling for the golden evaluation set |
| `evaluate.py` | 3-way evaluation: Trivial vs Simple vs RAG agent |
| `setup_data.py` | Downloads the TWCS dataset from Kaggle if not present |
| `Report.md` | Detailed report: architecture, evaluation, failure modes |
| `Decision_Log.md` | Engineering decisions and rationale |
