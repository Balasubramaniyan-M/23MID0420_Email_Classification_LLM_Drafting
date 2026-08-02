# MDI3003 Lab 03: Email Classification & Automatic LLM Draft Generation

**Student Name:** Balasubramaniyan M  
**Register Number:** 23MID0420  
**Course:** MDI3003 - Advanced Predictive Analytics (Core Laboratory Scope)  
**Date:** August 02, 2026  

---

## 1. Overview & Operational Boundary

This repository implements the complete Core Scope of MDI3003 Lab 03. The system combines sparse text machine learning classification with Large Language Model (LLM) API draft generation under strict privacy, security, and human-in-the-loop review policies.

> **MANDATORY OPERATIONAL BOUNDARY:** The system generates reply **drafts only**. The codebase contains **zero automated sending logic**, zero live mailbox scraping, and zero external mail server connections. All generated drafts are logged locally as JSON audit records for human inspection.

---

## 2. Environment Setup & Installation

### Requirements
- Python 3.10+
- Virtual environment recommended

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/BalasubramaniyanM/23MID0420_Email_Classification_LLM_Drafting.git
cd 23MID0420_Email_Classification_LLM_Drafting

# Install required Python packages
pip install -r requirements.txt
```

---

## 3. Secure API Key Configuration

The system uses the OpenAI Python SDK to generate automatic email drafts. The API key must **NEVER** be hardcoded or committed to version control.

### Setting the Environment Variable
On Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="your-actual-api-key-here"
$env:OPENAI_MODEL="gpt-5-mini"
```

On Linux / macOS:
```bash
export OPENAI_API_KEY="your-actual-api-key-here"
export OPENAI_MODEL="gpt-5-mini"
```

*Note: If `OPENAI_API_KEY` is not set, the pipeline gracefully utilizes a deterministic, policy-compliant local fallback draft generator so that all notebook cells and pipeline tests execute end-to-end without network errors.*

---

## 4. Reproducibility & Execution Order

To run the complete benchmark and reproduce all results:

### Option A: Running the Pipeline Execution Script
```bash
python run_pipeline.py
```

### Option B: Running the Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/23MID0420_Lab03_EmailAI.ipynb
```

Execution sequence in the notebook:
1. Environment setup & seeding (`RANDOM_STATE = 42`)
2. Dataset loading & schema validation
3. Data audit & SHA-256 integrity verification
4. Locked 80/20 train/test split generation (`outputs/split_manifest.json`)
5. TF-IDF pipeline registry setup (5 classifiers)
6. 5-Fold Stratified Cross-Validation on training split (`outputs/cv_results_all_datasets.csv`)
7. Model selection & single evaluation on locked test split (`outputs/test_results.csv`)
8. Selective routing & review signal logic
9. PII regex redaction & LLM draft generator
10. Prompt-injection defense demonstration
11. 6-case draft quality evaluation worksheet (`outputs/draft_quality_ratings.csv`)
12. Visualization plots & discussion
13. Minimal acceptance test verification

---

## 5. Known System Limitations

1. **Regex PII Redaction:** The teaching baseline relies on regex patterns for emails and phone numbers. Production DLP requires named entity recognition (NER).
2. **Synthetic Dataset Scope:** Dataset D1 is synthetically constructed for laboratory benchmarking and may not capture full corporate jargon complexity.
3. **Draft Evaluation:** Core scope relies on single-rater evaluation across representative test cases.

---

## 6. Required Academic Integrity Disclosure

> **ACADEMIC INTEGRITY DISCLOSURE:** This repository and associated artifacts were scaffolded with AI assistance per the course required-disclosure policy. All code, experimental design choices, data pipelines, model evaluations, and report contents were fully reviewed, verified, executed, and validated by the student author.
