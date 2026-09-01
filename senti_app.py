import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

st.set_page_config(page_title="Sentiment Classifier", page_icon="💬")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("models/bert_sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("models/bert_sentiment")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
labels = ["Negative", "Positive"]

st.title("💬 Review Sentiment Classifier")
st.write("Enter a product/movie review and get its predicted sentiment.")

user_input = st.text_area("Your review:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)

        st.subheader(f"Prediction: **{labels[pred_idx]}**")
        st.write("Confidence scores:")
        for label, prob in zip(labels, probs):
            st.progress(float(prob), text=f"{label}: {prob:.2%}")

st.markdown("---")
st.caption("Model: Fine-tuned DistilBERT (PyTorch) | Built with HuggingFace + Streamlit")