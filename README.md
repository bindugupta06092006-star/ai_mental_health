🧠 AI Mental Health Support Chat (Non-Medical)

A Streamlit-based web application that provides emotion-aware supportive responses, session summaries, and self-care suggestions.
This project is not medical advice — it is only for emotional support and educational purposes.

🚀 Features
✔ Emotion Detection (Two Modes)

ML Model (if available)

Uses SentenceTransformer embeddings (all-MiniLM-L6-v2)

Logistic Regression classifier

LabelEncoder for mapping emotions

Loaded from: models/emotion_clf.joblib

Rule-Based Fallback

Keyword-based emotion detection

Works even if the model file is missing

✔ Supportive AI Responses

The bot provides empathetic, safe messages like:

“I’m sorry you’re feeling sad. It’s valid to feel this way…”

Each emotion triggers a unique supportive response.

✔ Suggestions Panel

Shows self-care tips based on the latest detected emotion, e.g.:

Grounding exercises

Breathing techniques

Journaling

Walking/relaxation tips

✔ Crisis & Safety Section

Displays emergency mental health helplines and safety reminders.

✔ Session History & Export

Every message is stored in st.session_state

Shows timestamps + detected emotion + bot response

Download chat history as CSV

Option to clear session

📁 Project Structure
├── app.py                     # Main Streamlit application
├── train_model.py             # Script to train the emotion classifier
├── requirements.txt           # Python dependencies
├── models/
│   └── emotion_clf.joblib     # Saved model (generated after training)
└── data/
    └── ai_mental_health_dataset.csv   # Training dataset (auto-created if missing)

🔧 Installation
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ (Optional) Train the ML Model

If you want improved accuracy:

python train_model.py


This will:

Create a small demo dataset (if missing)

Train a Logistic Regression classifier

Save: models/emotion_clf.joblib

▶ Run the App
streamlit run app.py


If using Google Colab, you can expose it using PyNgrok.

💡 How It Works
🔹 Emotion Detection Flow
User Text 
     ↓
Encode via SentenceTransformer
     ↓
Model Predicts Emotion
     ↓
Bot Sends a Supportive Response
     ↓
Session History Updated


If model fails → fallback keyword-based detection activates.

🖼 Screenshots
Chat + Input Area

(From your uploaded screenshot)
![Chat UI](/mnt/data/ai mental health image 1.jpeg)

Emotion Summary + Suggestions

![Emotion Summary](/mnt/data/ai mental health image 2.jpeg)

🛡 Disclaimer

This app does not replace professional help.
If someone is in crisis or immediate danger, they must contact local emergency services or a qualified mental health professional.

📝 Future Enhancements

Add more emotion classes

Multi-turn emotional context analysis

Database storage (MongoDB / Firebase)

Option to fine-tune transformer models

Voice input / output support

❤️ Contributing

Pull requests and improvements are welcome!
