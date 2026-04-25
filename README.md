# 📧 Email Spam Detector

A production-ready spam detection web application extending the research paper:

> *"Machine Learning-Based Email Spam Detection: Accuracy, Overfitting and Robustness Analysis"*
> Published in EJASET, Volume 3, Issue 6, 2025

Built by **Ghulam Muhayyudin** — Computer Science Undergraduate Researcher

---

## 🚀 Live Demo
👉 **[https://email-spam-detector-1003.streamlit.app](https://email-spam-detector-1003.streamlit.app)**

---

## 🧠 About This Project

This system classifies emails as **spam** or **ham (legitimate)** using:
- **TF-IDF** vectorisation (5,000 features + bigrams)
- **Random Forest** — Best overall model (Accuracy: 98.21%, F1: 98.11%)
- **Logistic Regression CV** — Strong alternative (Accuracy: 97.84%, F1: 97.73%)
- **Tesseract OCR** — Detects text hidden inside image attachments (multimodal spam)

---

## 📊 Model Performance (Actual Training Results)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| ★ Random Forest | **98.21%** | **98.14%** | **98.08%** | **98.11%** |
| Logistic Regression CV | 97.84% | 97.34% | 98.12% | 97.73% |

> ★ Best model on this dataset by all metrics.
> Both models trained on 70/30 stratified split with 5-fold cross-validation.

---

## 🗂️ Project Structure

```
Email-Spam-Detector/
├── app.py                  ← Streamlit web application
├── train.py                ← Model training script
├── requirements.txt        ← Python dependencies
├── packages.txt            ← System packages (Tesseract OCR)
├── data/
│   └── dataset.csv         ← Email dataset (spam/ham)
├── models/                 ← Saved model files (generated after training)
└── src/
    ├── preprocessor.py     ← Text cleaning pipeline
    ├── trainer.py          ← Model definitions and persistence
    ├── evaluator.py        ← Metrics and evaluation reports
    └── ocr_extractor.py    ← Image OCR for multimodal spam
```



## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/ghulammuhayyudin1003/Email-Spam-Detector.git
cd Email-Spam-Detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR (for image spam detection)
- **Windows:** [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt-get install tesseract-ocr`

### 4. Train the models
```bash
python train.py
```

### 5. Launch the web app
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🌐 Deployment

✅ **Live on Streamlit Community Cloud (free)**
🔗 https://email-spam-detector-1003.streamlit.app

- `requirements.txt` → auto pip installs all Python packages
- `packages.txt` → auto installs `tesseract-ocr` system binary
- Dataset auto-downloads on first launch (UCI SMS Spam Collection)
- Models auto-train on first launch (~2-3 minutes)

---

## 🔬 Research Background

This project is the practical implementation of the paper's key findings:

- **Why Random Forest?** Ensemble of 300 trees with `min_samples_leaf=2` produces the lowest variance across folds (std=0.0058), confirming the paper's robustness analysis.
- **Why Logistic Regression CV?** Built-in cross-validated regularisation eliminates manual hyperparameter tuning. Strong alternative to Random Forest with 97.84% accuracy.
- **Why TF-IDF over word embeddings?** The paper confirmed TF-IDF achieves near-identical accuracy to more complex representations on this domain, with far lower inference cost.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| scikit-learn | ML models + TF-IDF |
| NLTK | Text preprocessing |
| Tesseract + pytesseract | OCR for image spam |
| Streamlit | Web interface |
| joblib | Model serialisation |
| pandas | Data handling |

---

## 👤 Author

**Ghulam Muhayyu Din**
Computer Science Undergraduate
GitHub: [@ghulammuhayyudin1003](https://github.com/ghulammuhayyudin1003)
Google Scholar: (https://scholar.google.com/citations?user=2H5SwVkAAAAJ&hl=en&authuser=2&oi=ao)
