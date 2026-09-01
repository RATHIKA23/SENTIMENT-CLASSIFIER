# 💬 Sentiment Classifier — LSTM vs. Fine-Tuned DistilBERT

A binary sentiment classification project comparing a classical deep learning approach (Bidirectional LSTM, trained from scratch) against a fine-tuned transformer (DistilBERT), deployed as an interactive Streamlit app.

**🔗 Live Demo:** [Add your Streamlit Cloud URL here after deployment]

![App Screenshot](images/app_screenshot.png)
*(Add a screenshot of your running Streamlit app here)*

---

## 📌 Project Overview

Customer/movie reviews carry sentiment that's expensive to label manually at scale. This project builds and compares two approaches to automatically classify review sentiment as **Positive** or **Negative**:

1. A **Bidirectional LSTM** trained from scratch on the dataset — a classical deep learning baseline.
2. A **fine-tuned DistilBERT** transformer — leveraging transfer learning from a model pretrained on large-scale text.

The goal was to build the full ML lifecycle end-to-end: data exploration → preprocessing → model training → evaluation → error analysis → deployment.

---

## 📊 Dataset

- **Source:** [IMDB Dataset of 50K Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 reviews, perfectly balanced (25,000 positive / 25,000 negative)
- **Labels:** Binary — `positive` / `negative`
- **Split:** 80% train / 20% test, stratified

---

## 🧠 Approach

### 1. Exploratory Data Analysis
- Verified class balance (50/50 split — no class imbalance handling needed)
- Analyzed review length distribution to inform sequence length decisions (95th percentile ≈ 579 words)
- Generated word clouds per sentiment class to sanity-check label quality

### 2. Preprocessing
- Cleaned HTML tags (`<br />`), punctuation, and numbers; lowercased all text
- **LSTM path:** stopwords removed (helps a model with no attention mechanism focus on signal words)
- **BERT path:** stopwords retained (transformers rely on full context, including function words like "not")
- Tokenized with Keras `Tokenizer` (vocab size 10,000, `max_len=300`) for the LSTM path
- Tokenized with HuggingFace `AutoTokenizer` (`max_length=256`) for the BERT path

### 3. Modeling

**Bidirectional LSTM** (TensorFlow/Keras, trained from scratch)
- Embedding (128-dim) → 2x Bidirectional LSTM → Dense → Dropout → Sigmoid output
- Built on `tf.data.Dataset` pipelines (`.shuffle()`, `.batch()`, `.prefetch()`) for efficient input handling
- ~1.4M trainable parameters

**Fine-Tuned DistilBERT** (HuggingFace Transformers, PyTorch)
- `distilbert-base-uncased` fine-tuned with a classification head
- Trained via HuggingFace `Trainer` API on Google Colab (T4 GPU)
- 3 epochs, best checkpoint selected by validation loss (`load_best_model_at_end=True`)

### 4. Evaluation

| Model | Accuracy | Precision (avg) | Recall (avg) | F1 (macro) |
|---|---|---|---|---|
| Bi-LSTM (from scratch) | 89% | 0.89 | 0.89 | 0.89 |
| DistilBERT (fine-tuned) | **92%** | **0.92** | **0.92** | **0.92** |

Fine-tuning DistilBERT improved F1 by ~3 points over the LSTM baseline — reflecting the value of transfer learning from large-scale pretraining versus learning purely from a 40K-example dataset.

### 5. Key Learnings
- The LSTM's recall on negative reviews (0.86) trailed its precision (0.91), suggesting it misses more true negatives than it wrongly flags — DistilBERT closed this gap, staying balanced across both classes.
- Both models are expected to struggle most with sarcasm, mixed reviews (praising one aspect while criticizing another), and negation patterns like "not as bad as expected" — areas classical bag-of-words-adjacent approaches typically underperform on.

---

## 🚀 Deployment

The trained DistilBERT model is served through an interactive **Streamlit** app (`senti_app.py`), where a user can type a review and get a real-time sentiment prediction with confidence scores.

### Run locally
```bash
git clone <your-repo-url>
cd sentiment-classifier
pip install -r requirements.txt
streamlit run senti_app.py
```

---

## 🛠️ Tech Stack

`TensorFlow` · `Keras` · `PyTorch` · `HuggingFace Transformers` · `Streamlit` · `scikit-learn` · `pandas` · `NLTK` · `Google Colab (GPU training)`

---

## 📁 Project Structure

```
sentiment-classifier/
├── data/
│   └── IMDB Dataset.csv
├── src/
│   └── preprocess.py          # stopword removal, tokenization/padding utilities
├── models/
│   ├── lstm_sentiment/
│   │   ├── lstm_model.keras
│   │   └── tokenizer.pkl
│   └── bert_sentiment/        # fine-tuned DistilBERT (PyTorch)
├              
├── senti.ipynb                # main notebook: EDA, preprocessing, training, evaluation
├── senti_app.py                # Streamlit app
├
└── README.md
```

---

## 🔮 Future Improvements
- Extend to 3-class sentiment (positive/neutral/negative) using star-rating-based labeling on Amazon Reviews
- Add explainability (SHAP/LIME) to show which words drove each prediction
- Batch prediction mode: upload a CSV of reviews, get sentiment for all of them
- Containerize with Docker for more portable deployment
