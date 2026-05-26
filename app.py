import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}

.main {
    background-color: #0E1117;
    color: white;
}

.stTextArea textarea {
    background-color: #262730;
    color: white;
    border-radius: 10px;
}

.stButton button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 200px;
    font-size: 18px;
}

h1 {
    color: #ff4b4b;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("model/")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📰 Fake News Detector")
st.sidebar.write("AI-powered Fake News Detection using BERT")

st.sidebar.info("""
Features:
✔ Real/Fake Prediction
✔ Confidence Score
✔ Prediction History
✔ Charts
✔ Dark UI
""")

# ---------------- TITLE ----------------
st.title("📰 Fake News Detection System")
st.write("Enter a news article below to detect whether it is REAL or FAKE.")

# ---------------- HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- INPUT ----------------
news = st.text_area(
    "Enter News Text",
    height=200,
    placeholder="Paste your news article here..."
)

# ---------------- BUTTON ----------------
if st.button("Detect News"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:

        # Loading animation
        with st.spinner("Analyzing News..."):
            time.sleep(2)

            inputs = tokenizer(
                news,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            outputs = model(**inputs)

            probs = F.softmax(outputs.logits, dim=1)

            confidence = torch.max(probs).item() * 100

            prediction = torch.argmax(outputs.logits).item()

        # Result
        if prediction == 1:
            result = "REAL NEWS"
            st.success(f"✅ {result}")
        else:
            result = "FAKE NEWS"
            st.error(f"❌ {result}")

        # Confidence Score
        st.subheader("Confidence Score")
        st.progress(int(confidence))
        st.write(f"Prediction Confidence: {confidence:.2f}%")

        # Save History
        st.session_state.history.append({
            "News": news[:50] + "...",
            "Result": result,
            "Confidence": round(confidence, 2)
        })

# ---------------- HISTORY TABLE ----------------
if len(st.session_state.history) > 0:

    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df)

    # ---------------- CHART ----------------
    st.subheader("📊 Prediction Statistics")

    chart_data = history_df["Result"].value_counts()

    fig, ax = plt.subplots()

    ax.pie(
        chart_data,
        labels=chart_data.index,
        autopct='%1.1f%%'
    )

    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Developed using BERT + Streamlit")