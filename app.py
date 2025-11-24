# app.py
import streamlit as st
import joblib
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="AI Mental Health Support Chat", layout="wide")

# --- Helpers ---
MODEL_PATH = "models/emotion_clf.joblib"

def load_model():
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        clf = data["clf"]
        le = data["label_encoder"]
        embed_model_name = data.get("embed_model_name", "all-MiniLM-L6-v2")
        embedder = SentenceTransformer(embed_model_name)
        return {"clf": clf, "le": le, "embedder": embedder}
    return None

# Simple rule-based fallback
FALLBACK_KEYWORDS = {
    "sad": ["sad", "unhappy", "depressed", "lonely", "hopeless", "cry"],
    "happy": ["happy", "glad", "excited", "joy", "great", "thrill", "optimistic"],
    "anxious": ["anxious", "anxiety", "nervous", "scared", "worried", "panic"],
    "angry": ["angry", "mad", "furious", "annoyed", "irritated"]
}

DEFAULT_TIPS = {
    "sad": ["Talk to a trusted friend or family member.", "Try a short walk or breathing exercise.", "Consider journaling one thought you're grateful for."],
    "happy": ["Notice and savour the good moment — write it down!", "Share your joy with someone."],
    "anxious": ["Try 4-4-4 breathing (inhale 4s, hold 4s, exhale 4s).", "Ground yourself: name 5 things you can see."],
    "angry": ["Take a 5-minute pause and breathe deeply.", "If safe, move to a private space to cool down."]
}

def fallback_detect(text):
    text_lower = text.lower()
    scores = {k:0 for k in FALLBACK_KEYWORDS}
    for label, kws in FALLBACK_KEYWORDS.items():
        for kw in kws:
            if kw in text_lower:
                scores[label] += 1
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "neutral"
    return best

def generate_support_message(emotion):
    messages = {
        "sad": "I'm sorry you're feeling sad. It's valid to feel this way. Would you like a few grounding exercises or some self-care suggestions?",
        "happy": "That's wonderful to hear! Want to share what made you feel this way?",
        "anxious": "It sounds like you are feeling anxious. Let's try a short breathing exercise together. Would you like that?",
        "angry": "It sounds like you're feeling angry. Want tips to calm down or ways to express this safely?",
        "neutral": "Thanks for sharing. Tell me more if you like — I'm here to listen."
    }
    return messages.get(emotion, messages["neutral"])

# --- Load model (if available)
with st.spinner("Loading model..."):
    model_bundle = load_model()

# --- Initialize session state defaults (before widgets are created)
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {time,user,message,emotion,reply}
if "_warning" not in st.session_state:
    st.session_state._warning = ""

# --- UI layout
st.title("🧠 AI Mental Health Support Chat (Non-medical)")
col1, col2 = st.columns([2,1])

with col1:
    st.header("Chat")

    # Create the text area widget (key must be set)
    st.text_area("Type how you're feeling or what's on your mind:", height=120, key="input_area")

    # Define submit callback — do all session_state writes inside this
    def submit():
        text = st.session_state.input_area.strip()
        if text == "":
            st.session_state._warning = "Please type something before sending."
            return

        # detect emotion
        try:
            if model_bundle:
                embed = model_bundle["embedder"].encode([text])
                pred = model_bundle["clf"].predict(embed)[0]
                emotion = model_bundle["le"].inverse_transform([pred])[0]
            else:
                emotion = fallback_detect(text)
        except Exception as e:
            # if any model error happens, fallback to rule-based detection
            emotion = fallback_detect(text)

        reply = generate_support_message(emotion)
        now = datetime.utcnow().isoformat() + "Z"

        st.session_state.history.append({
            "time": now,
            "user": text,
            "emotion": emotion,
            "reply": reply
        })

        # clear the text area by updating session_state inside callback (allowed)
        st.session_state.input_area = ""
        st.session_state._warning = ""

    # attach callback to the button
    st.button("Send", on_click=submit)

    # show warning if set by callback
    if st.session_state.get("_warning"):
        st.warning(st.session_state["_warning"])

    # show conversation
    if st.session_state.history:
        for entry in reversed(st.session_state.history):
            st.markdown(f"**You ({entry['emotion']})** • {entry['time']}")
            st.write(entry["user"])
            st.markdown(f"**Support Bot:** {entry['reply']}")
            st.markdown("---")
    else:
        st.info("No messages yet. Type above to start.")

with col2:
    st.header("Detected Emotion Summary")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        counts = df_hist["emotion"].value_counts().to_dict()
        st.write("Emotion counts (this session):")
        st.write(counts)

        # Show tips for the most recent emotion
        latest = df_hist.iloc[-1]["emotion"]
        st.subheader("Suggestions")
        tips = DEFAULT_TIPS.get(latest, ["Try taking 3 deep breaths.", "If in crisis, contact local emergency services."])
        for t in tips:
            st.write("•", t)
    else:
        st.write("No messages yet. Type above to start.")

    st.markdown("---")
    st.header("Crisis & Safety")
    st.write("If you are in immediate danger, please contact your local emergency services right away.")
    st.write("- In India: National Helpline for Mental Health (example): 08046110007 (verify locally)")
    st.write("- International: Find local emergency / crisis numbers — consider including national suicide prevention hotlines in your region.")
    st.caption("This app does NOT provide medical advice. For professional help, consult a qualified mental health professional.")

    st.markdown("---")
    st.header("Session Actions")
    if st.session_state.history:
        df_export = pd.DataFrame(st.session_state.history)
        csv = df_export.to_csv(index=False)
        st.download_button("Download session (CSV)", csv, "session.csv", "text/csv")
        if st.button("Clear session"):
            st.session_state.history = []
            st.experimental_rerun()

st.markdown("---")
st.caption("Model loaded from models/emotion_clf.joblib if found. If you want better accuracy, train with a larger labeled dataset (see train_model.py).")