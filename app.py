import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_models():
    model = joblib.load("xgbtfidf.pkl")
    vectorizer = joblib.load("tfidfvec.pkl")
    return model , vectorizer
model , vectorizer = load_models()
# Page

st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Smart MCQ Solver")
st.write("Enter an MCQ and let the model predict the most likely answers.")

# Input
question = st.text_area(
    "Enter your question",
    placeholder="Example: What is the capital of France?"
)

option_a = st.text_input("A")
option_b = st.text_input("B")
option_c = st.text_input("C")
option_d = st.text_input("D")
option_e = st.text_input("E")

options = [
    option_a,
    option_b,
    option_c,
    option_d,
    option_e
]
# Prediction

if st.button("Solve MCQ"):

    if not question.strip():
        st.warning("Please enter a question.")

    elif any(not option.strip() for option in options):
        st.warning("Please enter all five options.")

    else:

        scores = []

        for option in options:

            text = question + " <SEP> " + option

            X = vectorizer.transform([text])

            probability = model.predict_proba(X)[0][1]

            scores.append(probability)

        # Rank options
        ranked_indices = np.argsort(scores)[::-1]

        st.subheader("Predicted Answers")

        for rank, index in enumerate(ranked_indices[:3], start=1):

            letter = chr(65 + index)

            st.write(
                f"**{rank}. {letter}** — "
                f"{options[index]} "
                f"({scores[index]:.4f})"
            )