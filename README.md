
# 🔬 Secure RACG Experiments

This repository contains experiments for evaluating defense mechanisms against **poisoning attacks** in Retrieval-Augmented Code Generation (RACG) systems using **CodeLlama** and **Gemini** models.

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

Each subfolder contains scripts for:
- Corpus poisoning
- Retrieval using FAISS
- Prompt construction and LLM invocation
- Output saving and evaluation

Supported base LLMs:
- `CodeLlama`
- `Gemini`

---

### 🛡️ Vulnerability Summary Module

A lightweight sub-LLM trained to summarize retrieved code with:
- **Vulnerability type**
- **Exploitability context**
- **Associated security intent**

Helps reduce generation of unsafe code when used as a gating or prompt-enhancement module.

- `Finetune/`: contains dataset and training code for the summary model.
- `Inference/`: inference scripts for generating summaries at runtime.

---

## 🚀 Run Instructions

### Run an Experiment

```bash
cd Experiment/Scenario1/CodeLlama
python run_experiment.py
```

Modify `config.yaml` to switch between different poisoning settings.

### Generate Vulnerability Summaries

```bash
cd VulnerabilitySummaryModule/Inference
python generate_summary.py --input ./retrieved_examples.json --output ./summaries.json
```

---

## 📊 Evaluation Metric

- **Secure Rate (SR)**: Proportion of LLM-generated code that is free from high/critical vulnerabilities, as detected by static analyzers (e.g., Semgrep).
- **Poison Impact (PI)**: Difference in SR between clean and poisoned settings.

---

## 📌 Notes

- CodeLlama requires local inference or API wrapper.
- Gemini requires access to Gemini API (via Google Cloud or third-party wrapper).
- All experiments can be extended to integrate with secure prompting strategies.

---

## 📜 License

MIT License.

---

## ✏️ Citation

> Coming soon – under submission to [Venue Hidden].

---
