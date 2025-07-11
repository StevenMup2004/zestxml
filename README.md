
# 🔬  RAGGuard - Strategy 2

This repository contains experiments Strategy 2 for evaluating defense mechanisms against **poisoning attacks** in RACG systems using **CodeLlama** and **Gemini** models.

---

## 📁 Project Structure

```
.
├── Experiment/
│   ├── Scenario1/                  # Targeted poisoning: query-aligned injection
│   │   ├── CodeLlama/
│   │   └── Gemini/
│   └── Scenario2/                  # Untargeted poisoning: stealthy and generic injection
│       ├── CodeLlama/
│       └── Gemini/
│
├── VulnerabilitySummaryModule/
│   ├── Finetune/                   # Fine-tuning a vulnerability summarizer
│   └── Inference/                  # Inference-time generation of summaries
```

---

## 📘 Modules

### 🧪 Experiment

Each scenario simulates a different type of poisoning:

- **Scenario 1** – *Targeted Poisoning*: malicious code is inserted that closely matches the query intent to fool retrieval + generation.
- **Scenario 2** – *Untargeted Poisoning*: general-purpose vulnerable code is injected to stealthily degrade generation quality.

Supported base LLMs:
- `CodeLlama`
- `Gemini`

---

### 🛡️ Vulnerability Summary Module

A lightweight sub-LLM trained to summarize retrieved code with:
- **Vulnerability serverity**
- **Associated security intent**

Helps reduce generation of unsafe code when used as a gating or prompt-enhancement module.

- `Finetune/`: contains finetune LLM.
- `Inference/`: inference scripts for generating summaries.

---


