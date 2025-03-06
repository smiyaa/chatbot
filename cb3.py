import nltk
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
import streamlit as st



training_sentences=["Hello", "Hi there", "Hey", "Greetings",
    "Can you help me?", "I need assistance", "I need support",
    "What is the hospital's location?", "Where is the hospital?",
    "How do I book an appointment?", "I want to schedule a checkup",
    "What are the hospital timings?", "When does the hospital open?",
    "Do you have emergency services?", "Is there an emergency department?",
    "Which doctors are available?", "Can I get a list of doctors?",
    "What services do you offer?", "Tell me about hospital services.",
    "Do you accept insurance?", "Is insurance accepted at your hospital?",
    "How much is a consultation fee?", "What are the charges for consultation?",
    "How long is the waiting time?", "What is the average waiting time?",
    "Can I get a prescription refill?", "How do I refill my medicine?",
    "How do I contact the hospital?", "What is your contact number?"
]

intent_labels = [
    "greeting", "greeting", "greeting", "greeting",
    "assistance", "assistance", "assistance",
    "location", "location",
    "appointment", "appointment",
    "timing", "timing",
    "emergency", "emergency",
    "doctors", "doctors",
    "services", "services",
    "insurance", "insurance",
    "consultation_fee", "consultation_fee",
    "waiting_time", "waiting_time",
    "prescription", "prescription",
    "contact", "contact"
]

def custom_tokenizer(text):
    return word_tokenize(text)

vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, stop_words="english")
X = vectorizer.fit_transform(training_sentences)
y = np.array(intent_labels)

model=SVC(kernel="linear")
model.fit(X,y)


# Save model and vectorizer
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

with open("chatbot_model.pkl","rb") as f:
    vectorizer, model = pickle.load(f)

# predicts the response
def predict_intent(user_input):
    input_vector=vectorizer.transform([user_input])
    prediction=model.predict(input_vector)[0]
    return prediction


def generate_response(intent):
    responses = {
        "greeting": "Hello! How can I assist you today?",
        "assistance": "Sure! What do you need help with?",
        "location": "The Care Medical hospital is located at 123 Wellness Avenue, MedCity, Kochi, India .",
        "appointment": "You can book an appointment by calling our reception or visiting our website.",
        "timing": "Our hospital is open from 8 AM to 10 PM, Monday to Saturday.",
        "emergency": "Yes, we have a 24/7 emergency department. Please visit immediately in case of an emergency.",
        "doctors": "We have specialists in cardiology, neurology, orthopedics, and general medicine. Please check our website for more details.",
        "services": "We offer general check-ups, diagnostic tests, surgeries, and emergency care.",
        "insurance": "Yes, we accept major insurance providers. Please contact our billing department for details.",
        "consultation_fee": "Our consultation fee starts at 250rupees. It may vary depending on the specialist.",
        "waiting_time": "The average waiting time is around 30 minutes, but it depends on the number of patients.",
        "prescription": "You can request a prescription refill by contacting your doctor or visiting the pharmacy.",
        "contact": "You can reach us at +919876543210 or email us at contact@caremedical.com.",
        "unknown": "I'm not sure how to respond to that. Can you please rephrase?"
    }
    return responses.get(intent,responses["unknown"])

def chatbot(input_sentence):
    intent = predict_intent(input_sentence)
    response = generate_response(intent)
    return response


st.title("Care Medical chatBot")





# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
user_input = st.text_input("You: ", key="input_text")

if user_input:
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate bot response (Replace this with ML-based response)
    bot_response =chatbot(user_input)
    
    # Append bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response})

# Display chat history
for message in st.session_state.messages:
    role = "🧑‍💻" if message["role"] == "user" else "🤖"
    st.write(f"{role} {message['content']}")

# Clear input after submission
st.session_state.user_input = ""






