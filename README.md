# 23MID0420_Email_Classification_LLM_Drafting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https.mit-license.org)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![OpenAI SDK](https://img.shields.io/badge/OpenAI-API-green.svg)](https://platform.openai.com/)

**Student Name:** Balasubramaniyan M  
**Register Number:** 23MID0420  
**Course:** MDI3003 - Advanced Predictive Analytics (Lab 03 - Core Laboratory Scope)  
**Project Title:** Benchmark-Aligned Email Classification and LLM API-Based Automatic Email Draft Generation  

---

## Executive Overview

This repository houses an industry-level, privacy-conscious predictive analytics system designed to automate email categorization and generate context-aware response drafts. The architecture links traditional sparse-text machine learning classifiers with an OpenAI API Large Language Model (LLM) draft generator under strict security controls.

```
+-------------------+      +-------------------------+      +--------------------------+
|  Incoming Email   | ---> | Sparse TF-IDF Pipeline  | ---> |   Predicted Category &   |
| (Subject + Body)  |      |   (MultinomialNB / SVC) |      | Confidence Margin Signal |
+-------------------+      +-------------------------+      +--------------------------+
                                                                         |
                                                                         v
+-------------------+      +-------------------------+      +--------------------------+
| JSON Audit Record | <--- |   LLM Draft Generator   | <--- | Selective Review Routing |
| (outputs/drafts/) |      |  (Regex PII Redaction)  |      |  (Mandatory Review Flag) |
+-------------------+      +-------------------------+      +--------------------------+
```

---

## Key Features

- **Multi-Classifier Benchmark:** Evaluates 5 machine learning pipelines (`DummyClassifier`, `MultinomialNB`, `ComplementNB`, `LogisticRegression`, `LinearSVC`) across two datasets.
- **Zero Data Leakage:** 80/20 train/test split locked upfront (`outputs/split_manifest.json`). Feature extraction fitted strictly inside training folds during 5-fold cross-validation.
- **Uncertainty-Aware Routing:** Decision margin thresholding (< 0.15) and mandatory review flagging for high-risk categories (`urgent_action`).
- **Selective Draft Policy:** Automatic draft suppression for `spam`; placeholder generation (`[PLACEHOLDER]`) for missing factual details.
- **Prompt-Injection Defense:** Untrusted email text delimited within `<email_data>` tags; explicit prohibition of instruction overrides or secret extraction.
- **Local JSON Audit Trail:** Audit logs stored under `outputs/drafts/{case_id}.json` tracking timestamp, prediction signal, review status, and draft text.

---

## Directory Structure

```
23MID0420_Email_Classification_LLM_Drafting
│
├── README.md                      # Primary open-source documentation
├── 23MID0420_Lab03_README.md      # Detailed environment setup & execution guide
├── LICENSE                        # MIT License
├── requirements.txt               # Required Python dependencies
├── .gitignore                     # Git exclusion rules
├── run_pipeline.py                # Standalone end-to-end execution runner
├── build_notebook.py              # Notebook generation script
│
├── data/
│   ├── README.md                  # Dataset manifest, SHA-256 hashes & schema
│   ├── business_email_intent.csv  # D1: Multiclass intent dataset (synthetic)
│   └── enron_spam.csv             # D2: Binary spam benchmark (Enron subset)
│
├── notebooks/
│   └── 23MID0420_Lab03_EmailAI.ipynb  # Executable core-lab notebook
│
├── src/                           # Modular Python packages (PEP 8, fully typed)
│   ├── __init__.py
│   ├── data_loader.py             # Schema validator & text derivation
│   ├── audit.py                   # Data audit & SHA-256 integrity
│   ├── pipelines.py               # TF-IDF pipeline registry
│   ├── evaluation.py              # 5-Fold CV & locked test evaluation
│   ├── routing.py                 # Margin calculation & review routing
│   ├── llm_draft.py               # Regex PII sanitizer & LLM draft generator
│   └── utils.py                   # Reproducibility seeding & plotting
│
├── outputs/
│   ├── split_manifest.json        # Locked 80/20 train/test IDs
│   ├── cv_results_all_datasets.csv# 5-Fold CV benchmark results
│   ├── test_results.csv           # Locked test set evaluation metrics
│   ├── draft_quality_ratings.csv  # 6-case human rubric ratings
│   ├── models/                    # Joblib fitted pipelines
│   └── drafts/                    # Local JSON audit records
│
├── draft_examples/                # Exported draft examples for submission
│
└── reports/
    ├── 23MID0420_Lab03_Report.md  # Comprehensive industry-style technical report
    └── 23MID0420_Lab03_Prompt.txt # Versioned system instructions (v1.0)
```

---

## Quickstart Guide

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Key (Optional for live LLM API calls):**
   ```powershell
   $env:OPENAI_API_KEY="your-api-key"
   ```

3. **Run Pipeline End-to-End:**
   ```bash
   python run_pipeline.py
   ```

---

## Operational Boundary & Educational Disclaimer

> **OPERATIONAL BOUNDARY:** This system is strictly an educational research prototype that generates email response **drafts only**. It contains **zero automated sending code** and **zero live mailbox scraping**. Every generated draft must be reviewed and approved by a human user before sending.

---

## Academic Integrity Disclosure

> **REQUIRED DISCLOSURE:** This repository and associated technical documentation were scaffolded with AI assistance per the course required-disclosure policy. All code, model benchmarks, evaluation metrics, and technical reports were verified, executed, and validated by Balasubramaniyan M (23MID0420).
