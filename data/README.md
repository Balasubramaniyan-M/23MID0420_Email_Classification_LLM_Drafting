# Data Manifest: Lab 03 Email Datasets

This directory contains the core email datasets used in the benchmark experiments for MDI3003 Lab 03: **Benchmark-Aligned Email Classification and LLM API-Based Automatic Email Draft Generation**.

---

## Dataset 1: Business Email Intent (`business_email_intent.csv`)

- **Dataset ID:** `business_intent`
- **Role in Project:** Primary dataset for multiclass operational intent classification, uncertainty routing, and LLM reply draft generation.
- **Task Type:** Multiclass Text Classification (6 classes)
- **Source & Provenance:** Synthetic instructor-style business intent corpus created strictly for laboratory benchmarking. Generated to model enterprise shared inbox communications.
- **Version & Hash:** Version 1.0 (SHA-256 computed dynamically during dataset audit step).
- **License / Permitted Use:** Educational and research use under MIT License.
- **Privacy Treatment:** 100% synthetic data containing zero real PII, zero confidential corporate data, and zero live email records.
- **Schema:**
  - `email_id` (string): Unique anonymized record ID (e.g., `D1_1000`)
  - `subject` (string): Email subject header
  - `body` (string): Plain-text email body
  - `label` (string): Ground-truth category (`request`, `meeting`, `complaint`, `information`, `urgent_action`, `spam`)
  - `dataset_id` (string): `business_intent`
  - `thread_id` (string, optional): Anonymized thread identifier (`TH_100`)
  - `sender_group` (string, optional): Anonymized sender group (`GRP_1`)
  - `timestamp` (datetime, optional): ISO 8601 UTC timestamp
- **Text Construction Rule:** Derived combined text field: `"subject: " + subject.strip() + "\nbody: " + body.strip()`.
- **Row Count & Class Distribution:** Total 360 rows (60 rows per class, 16.67% balanced distribution across 6 classes).
- **Known Limitations:** Controlled vocabulary length, synthetic phrasing patterns, lack of nested email headers or multi-language content.

---

## Dataset 2: Enron-Spam Subset (`enron_spam.csv`)

- **Dataset ID:** `enron_spam`
- **Role in Project:** Secondary dataset for binary spam vs legitimate email classification benchmarking.
- **Task Type:** Binary Text Classification (2 classes)
- **Source & Provenance:** Public benchmark representative subset derived from the Enron-Spam Corpus (Metsis et al., 2006).
- **Version & Hash:** Version 1.0 (SHA-256 computed dynamically during dataset audit step).
- **License / Permitted Use:** Public domain / research use as documented by Metsis et al. (2006) and CEAS 2006 benchmark guidelines.
- **Privacy Treatment:** Public benchmark corpus; sanitized and de-identified synthetic subset.
- **Schema:** Same unified schema as D1:
  - `email_id` (string): Unique anonymized record ID (e.g., `D2_2000`)
  - `subject` (string): Email subject header
  - `body` (string): Plain-text email body
  - `label` (string): Ground-truth label (`legitimate` vs `spam`)
  - `dataset_id` (string): `enron_spam`
  - `thread_id` (string, optional): Anonymized thread identifier
  - `sender_group` (string, optional): Anonymized sender group
  - `timestamp` (datetime, optional): ISO 8601 UTC timestamp
- **Text Construction Rule:** Derived combined text field: `"subject: " + subject.strip() + "\nbody: " + body.strip()`.
- **Row Count & Class Distribution:** Total 320 rows (160 `legitimate`, 160 `spam`, 50/50 balanced binary distribution).
- **Known Limitations:** Historical email domain shift (early 2000s corporate language), specific corporate vocabulary context.

---

## Strict Policy on Label Separation

> **MANDATORY RULE:** Datasets D1 and D2 MUST NOT be pooled into a single training file. D1 represents a 6-class operational intent taxonomy, whereas D2 represents a binary spam/ham task. Models are evaluated and selected independently per dataset.
