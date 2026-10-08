# VyadhiDrishti AI: Medical Disease Classification Through Natural Language Understanding

[![NLP Research](https://img.shields.io/badge/Domain-Clinical%20NLP-blue.svg)](https://github.com)
[![JMIR AI 2026](https://img.shields.io/badge/Systematic%20Review-Sakai%20%26%20Lam%20(2026)-green.svg)](https://ai.jmir.org/2026/1/e79202)
[![Benchmark Comparison](https://img.shields.io/badge/Benchmark-PoojaR24%20Medical--Text--Classification-orange.svg)](https://github.com/PoojaR24/Medical-Text-Classification.git)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Research%20Prototype-red.svg)](file:///e:/gls/%23%23SEM-7/NLPL/NLP%20Project/medical_text_disease_classification.ipynb)

**VyadhiDrishti AI** (*व्याधि-दृष्टि: Deep Clinical Insight into Disease from Symptom Narratives*) is an end-to-end Natural Language Processing (NLP) and Machine Learning research framework designed for automated diagnostic triage and closed-set disease classification from unstructured patient symptom descriptions and photographed clinical documents.

---

## 📚 Theoretical Foundation & Research Grounding

### 1. Research Paper Foundation: Sakai & Lam (2026)
This project is systematically grounded in the recent comprehensive review:
> **Sakai, H., & Lam, S. S. (2026).** *Large Language Models for Health Care Text Classification: Systematic Review.* **JMIR AI**, Vol. 5, e79202. [https://doi.org/10.2196/79202](https://doi.org/10.2196/79202).  
> *(Archived locally in project root as [`ai-2026-1-e79202.pdf`](file:///e:/gls/%23%23SEM-7/NLPL/NLP%20Project/ai-2026-1-e79202.pdf))*.

#### Key Syntheses from Sakai & Lam (2026) Applied in VyadhiDrishti AI:
1. **Clinical Decision Support Primacy**: Sakai & Lam synthesized 65 healthcare text classification papers (2020–2024), demonstrating that clinical decision support is the #1 application (29/65 studies).
2. **The Case for Lightweight Statistical ML**: While LLMs offer strong few-shot capacity, fine-tuning large foundation models on moderate clinical cohorts ($N \approx 1,000$) frequently suffers from high computational overhead, overfitting, and severe calibration drift. Furthermore, cloud-based LLM APIs introduce patient data privacy and HIPAA compliance barriers. In contrast, regularized TF-IDF with Calibrated Linear SVM and Logistic Regression offers:
   - Complete local on-premise privacy (zero cloud data leakage)
   - Exact mathematical explainability and feature coefficient auditability
   - Ultra-fast deterministic inference (<5ms latency)
   - Zero hallucination risk on clinical terminology

### 2. Prior Benchmark Comparison: PoojaR24/Medical-Text-Classification
We systematically contrast and advance beyond the baseline repository by **PoojaR24** ([GitHub: PoojaR24/Medical-Text-Classification](https://github.com/PoojaR24/Medical-Text-Classification.git)):

| Architectural Dimension | PoojaR24 Benchmark (`Medical-Text-Classification`) | **VyadhiDrishti AI (Our Framework)** |
| :--- | :--- | :--- |
| **Data Modality** | Formal PubMed literature abstracts | **Conversational patient symptom narratives (Triage domain)** |
| **Class Cardinality** | 5 broad macro-categories | **22 granular, specific disease classes** |
| **Model Families** | Basic K-Nearest Neighbors (KNN) | **Naive Bayes, Logistic Regression, Calibrated LinearSVM, RF** |
| **Validation Rigor** | Single holdout split | **5-Fold Stratified Cross-Validation + Exhaustive GridSearchCV** |
| **Negation Handling** | Default stop-word removal | **Negation-Preserving clinical tokenization (`no`, `not`, `without`)** |
| **Clinical Output** | Single deterministic class prediction | **Ranked Top-3 Differential Diagnoses + Calibrated Probabilities** |
| **Safety Protocol** | None (Forced closed-set prediction) | **Uncertainty Quantification with automated Abstention Gate ($T=0.35$)** |
| **Input Modality** | Text input only | **Multimodal: Raw text + Image OCR (Prescriptions / Clinic notes)** |

---

## 🔬 Core System Architecture (25 Sections in Notebook)

The complete implementation is delivered in a single, fully-executed Jupyter Notebook:  
**[`medical_text_disease_classification.ipynb`](file:///e:/gls/%23%23SEM-7/NLPL/NLP%20Project/medical_text_disease_classification.ipynb)**

```
├── 1. Project Title, Academic Abstract & Research Context (Sakai & Lam 2026 foundation)
├── 2. Clinical Problem Statement & Closed-Set Formulation
├── 3. Research Objectives & Methodological Roadmap
├── 4. Literature Review, Clinical NLP Evolution & Benchmark Comparison (PoojaR24 comparison)
├── 5. Environment Setup, Version Logging & Seed Locking (random_state=42)
├── 6. Dataset Acquisition & Multi-Tier Loader (Hugging Face + Local Fallback)
├── 7. Exploratory Data Analysis (EDA) & Lexical Profiling (Class Balance & Length KDEs)
├── 8. Clinical Text Cleaning & Negation Safeguarding (Preserving "no", "not", "without")
├── 9. Stratified Data Partitioning (80% Train, 20% Held-Out Test, Nested Validation)
├── 10. Baseline Reference Model (Zero-Rule Majority Classifier: ~4.7% accuracy lower bound)
├── 11. Feature Engineering: Sublinear N-Gram TF-IDF Vectorization ((1, 2) n-grams)
├── 12. Classical Machine Learning Benchmarking (Naive Bayes, Logistic Regression, LinearSVM, Random Forest)
├── 13. Systematic 5-Fold Stratified Cross-Validation (Mean ± Std Dev metrics)
├── 14. Hyperparameter Optimization via GridSearchCV
├── 15. Comprehensive Test Set Evaluation (Multi-Metric Performance Comparison)
├── 16. Annotated 22x22 Confusion Matrix Heatmap
├── 17. Granular Clinical Error Analysis & Qualitative Case Studies
├── 18. Linguistic Interpretability: Top Discriminative Keywords per Disease
├── 19. Custom Inference Engine with Ranked Top-3 Differential Diagnoses
├── 20. Uncertainty Quantification & Safe Clinical Abstention Protocol
├── 21. Deep Learning / Transformer Fine-Tuning Pipeline (DistilBERT / BioBERT)
├── 22. Multimodal Image OCR Document Ingestion & Diagnostic Triage
├── 23. Interactive File Upload Widget (Upload Custom Prescription / Report Images)
├── 24. Pipeline Serialization & Production Reload Verification
└── 25. Clinical Safety Disclaimers, Limitations, Conclusions & References
```

---

## 📊 Empirical Experimental Results

Evaluated on the held-out stratified test set (20% partition):

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Top-3 Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VyadhiDrishti Calibrated High-Precision SVC (Best)** | **95.75%** | **0.9592** | **0.9575** | **0.9568** | **0.9572** | **99.06%** |
| **TF-IDF + Calibrated Linear SVM** | 94.34% | 0.9450 | 0.9434 | 0.9431 | 0.9428 | 98.11% |
| **TF-IDF + Tuned Logistic Regression** | 93.40% | 0.9380 | 0.9340 | 0.9328 | 0.9328 | 98.58% |
| **TF-IDF + Multinomial Naive Bayes** | 91.04% | 0.9220 | 0.9104 | 0.9082 | 0.9072 | 97.64% |
| **TF-IDF + Random Forest Ensemble** | 89.62% | 0.9061 | 0.8962 | 0.8927 | 0.8931 | 96.70% |
| Majority Class Baseline | 4.72% | 0.0000 | 0.0000 | 0.0000 | 0.0042 | 13.64% |

---

## 📷 Multimodal Image OCR Feature

VyadhiDrishti AI integrates document ingestion for clinical reports and prescriptions:
1. **Ingestion & Enhancement**: Accepts `.png`, `.jpg`, `.jpeg` images, converts to grayscale, and optimizes contrast.
2. **Dual-Engine OCR**:
   - **Native Windows WinRT OCR (`winocr`)**: 100% native execution on Windows without external binary installations.
   - **Tesseract OCR (`pytesseract`)**: Cross-platform fallback.
3. **Automated Triage Inference**: Feeds extracted narrative into the classification pipeline to display:
   - Primary Predicted Disease
   - Calibrated Confidence Score
   - Ranked Top-3 Differential Diagnoses with percentage probabilities
   - Educational Medical Safety Notice
4. **Interactive Widget**: Embedded `ipywidgets.FileUpload` button allows uploading custom images right inside Jupyter Notebook.

---

## 📁 Repository Contents

```text
├── medical_text_disease_classification.ipynb   # Main Jupyter Notebook (Pre-executed with all plots & outputs)
├── ai-2026-1-e79202.pdf                        # Grounding Research Paper (Sakai & Lam, 2026, JMIR AI)
├── medical_disease_classifier_pipeline.joblib  # Serialized production-ready inference pipeline
├── disease_metadata.json                       # Model metadata, class list, and optimal hyperparameters
├── symptom_to_diagnosis.csv                    # Local cached dataset (100% offline reproducible)
├── sample_symptom_note.png                     # Sample medical intake report (Dengue symptoms) for OCR testing
├── sample_asthma_note.png                      # Sample medical intake report (Asthma symptoms) for OCR testing
├── requirements.txt                            # Python environment dependencies
└── README.md                                   # Comprehensive academic documentation
```

---

## 🚀 Execution Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter Notebook
jupyter notebook medical_text_disease_classification.ipynb
```

---

## ⚖️ Clinical Safety & Ethics Mandate

> **IMPORTANT NOTICE**: VyadhiDrishti AI is strictly an **academic research prototype** developed for educational and experimental evaluation. It is **NOT** a certified clinical diagnostic medical device and must **NEVER** replace clinical evaluations, diagnostic laboratory testing, or physician judgment.

## TEAM MEMBERS:
**Parshav Shah (202302626010119)**,
**Dhruvi Patel (202302626010049)**,
**Arya Doshi (202302626010019)**
