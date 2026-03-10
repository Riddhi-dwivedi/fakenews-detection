import streamlit as st
import pickle
import os

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "models/best_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "models/vectorizer.pkl"), "rb"))

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Custom Background + Styling
# -----------------------------
st.markdown("""
<style>
/* Full page background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #e3f2fd, #f8f9fa);
}



/* Card styling */
.main-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.1);
}

/* Common button style */
.stButton > button {
    border-radius: 10px;
    height: 48px;
    font-weight: 600;
    font-size: 16px;
    transition: 0.3s ease;
}

/* First button (Analyze) */
div[data-testid="column"]:first-child .stButton > button {
    background-color: #ff4b4b;
    color: white;
    border: none;
}

/* Hover effect */
div[data-testid="column"]:first-child .stButton > button:hover {
    background-color: #e63939;
}

/* Second button (Reset) */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background-color: transparent;
    color: #333;
    border: 1px solid #bbb;
}

/* Reset hover */
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background-color: #f2f2f2;
}



/* Footer */
.footer {
    text-align: center;
    font-size: 13px;
    color: #555;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("📰 Fake News Detection System")
st.markdown("Enter news text below to check whether it is TRUE or FALSE")

# -----------------------------
# Card Container
# -----------------------------
# st.markdown('<div class="main-card">', unsafe_allow_html=True)

news_input = st.text_area("Paste News Content Here", height=170)

# Buttons in one row
col1, col2 = st.columns(2, gap="small")

with col1:
    predict_btn = st.button("Analyze News", use_container_width=True)

with col2:
    clear_btn = st.button("Reset", use_container_width=True)

if clear_btn:
    st.rerun()

# -----------------------------
# Prediction
# -----------------------------
if predict_btn:
    if news_input.strip() == "":
        st.warning("⚠ Please enter some text.")
    else:
        with st.spinner("Analyzing news..."):
            vector = vectorizer.transform([news_input])
            prediction = model.predict(vector)

        st.markdown("---")

        if prediction[0] == 0:
            st.success("✅ This News is TRUE")
        else:
            st.error("❌ This News is FALSE")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
'<div class="footer">Fake News Detection using Machine Learning</div>',
unsafe_allow_html=True
)
