
import streamlit as st
import pandas as pd
import pickle
import io
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.markdown(
    "This app predicts whether a news headline/article is real or fake using a TF-IDF vectorizer + a scikit-learn classifier. You can enter a single headline, upload a CSV for batch predictions, or upload your trained `model.pkl` and `tfidf.pkl` files."
)

# Sidebar: model upload / instructions
st.sidebar.header("Model & Vectorizer")
uploaded_model = st.sidebar.file_uploader("Upload trained model (pickle .pkl)", type=["pkl"], key="model")
uploaded_tfidf = st.sidebar.file_uploader("Upload TF-IDF vectorizer (pickle .pkl)", type=["pkl"], key="tfidf")
use_default = st.sidebar.checkbox("Use default demo model if uploads missing", value=True)

@st.cache_resource
def load_pickle_file(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return pickle.load(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Failed to load pickle: {e}")
        return None

# try load uploaded model/vectorizer
model = load_pickle_file(uploaded_model)
tfidf = load_pickle_file(uploaded_tfidf)

# If not provided, optionally create a tiny demo pipeline (very simple) or warn
if model is None or tfidf is None:
    if use_default:
        # Create a tiny demo vectorizer + model fallback (very weak) so UI stays usable
        from sklearn.linear_model import LogisticRegression
        sample_corpus = [
            "This is real news from reliable sources",
            "Breaking: celebrity dies in shocking incident",
            "Government announces new policy for education",
            "Click here to win a free iPhone now",
            "You won't believe what happened next!"
        ]
        sample_labels = ["real", "real", "real", "fake", "fake"]
        tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
        X = tfidf.fit_transform(sample_corpus)
        model = LogisticRegression(max_iter=500)
        model.fit(X, sample_labels)
        st.sidebar.info("Using built-in demo model (only for testing). For real results upload trained model + tfidf in sidebar.")
    else:
        st.sidebar.warning("Upload trained model and TF-IDF vectorizer to make predictions.")

# Helper: preprocess text (basic)
def preprocess_text(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    return s

# Single headline prediction
st.header("Single headline check")
headline = st.text_area("Enter the news headline or short article:", height=80)
if st.button("Predict", key="single_predict"):
    if not headline or headline.strip() == "":
        st.warning("Please enter a headline or short article text to predict.")
    else:
        text = preprocess_text(headline)
        try:
            X = tfidf.transform([text])
            pred = model.predict(X)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[0]
                # find probability for predicted class
                if hasattr(model, "classes_"):
                    class_index = list(model.classes_).index(pred)
                    pred_prob = prob[class_index]
                else:
                    pred_prob = np.max(prob)
            else:
                pred_prob = None

            st.subheader("Result")
            if pred.lower() in ["fake", "0", "false", "f"]:
                st.error(f"Prediction: FAKE{f' (confidence {pred_prob:.2f})' if pred_prob is not None else ''}")
            else:
                st.success(f"Prediction: REAL{f' (confidence {pred_prob:.2f})' if pred_prob is not None else ''}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Batch prediction via CSV
st.header("Batch check (CSV)")
st.markdown("Upload a CSV with a column named `text` (or choose a column below). The app will append a `prediction` column.")
uploaded_csv = st.file_uploader("Upload CSV file for batch prediction", type=["csv"]) 

selected_col = None
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = None

    if df is not None:
        st.write("Preview of uploaded file:")
        st.dataframe(df.head())

        col_options = list(df.columns)
        selected_col = st.selectbox("Select text column to analyze", options=col_options, index=col_options.index("text") if "text" in col_options else 0)

        if st.button("Run batch prediction"):
            texts = df[selected_col].astype(str).apply(preprocess_text).tolist()
            try:
                X_batch = tfidf.transform(texts)
                preds = model.predict(X_batch)
                probs = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_batch)
                    # map class probabilities to predicted class
                    pred_probs = []
                    classes = list(model.classes_)
                    for i, p in enumerate(preds):
                        idx = classes.index(p)
                        pred_probs.append(probs[i][idx])
                else:
                    pred_probs = [None] * len(preds)

                df_out = df.copy()
                df_out["prediction"] = preds
                df_out["confidence"] = pred_probs

                st.success("Batch prediction complete")
                st.dataframe(df_out.head())

                # provide download
                csv_buffer = io.StringIO()
                df_out.to_csv(csv_buffer, index=False)
                b = csv_buffer.getvalue().encode()
                st.download_button("Download results as CSV", data=b, file_name="predictions.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

# Simple explanation / tips
st.markdown("---")
st.subheader("How to get accurate results")
st.markdown(
    "\n" 
    "1. Train a TF-IDF vectorizer and a classifier (e.g., LogisticRegression or SVM) on your labelled dataset.\n"
    "2. Save the trained vectorizer and model as `tfidf.pkl` and `model.pkl` (pickle).\n"
    "3. Upload both files in the sidebar for real predictions.\n"
)

st.expander("Example: save model and tfidf in Python")

code_example = '''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# after training
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
'''
st.code(code_example, language='python')

st.markdown("---")
st.write("Built with ❤️ — drop your trained model and TF-IDF in the sidebar to use your real pipeline.")
