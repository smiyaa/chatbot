# Care Medical ChatBot

A simple AI-powered chatbot built using **Python**, **Streamlit**, and **Scikit-Learn** to assist users with hospital-related queries such as appointment scheduling, doctor availability, hospital location, and more.

## Features
- Provides quick responses to common hospital-related questions.
- Uses **TF-IDF Vectorization** and **Support Vector Machine (SVM)** for intent classification.
- **Streamlit-based UI** for easy interaction.
- Pre-trained intent detection model with a set of predefined responses.
- Can be expanded with **Deep Learning (TensorFlow/Keras)** for more advanced interactions.

## Installation

### Prerequisites
Ensure you have Python **3.7+** installed on your system. You also need the following Python libraries:

```sh
pip install streamlit nltk scikit-learn numpy pickle-mixin
```

### Clone the Repository
```sh
git clone https://github.com/your-repo/hospital-chatbot.git
cd hospital-chatbot
```

## How to Run the Chatbot
```sh
streamlit run chatbot.py
```

## Project Structure
```
├── chatbot.py              # Main chatbot script with Streamlit UI
├── chatbot_model.pkl       # Pre-trained model for intent classification
├── requirements.txt        # List of required dependencies
├── README.md               # Project documentation
```

## How It Works
1. **Training Data**: The chatbot is trained on predefined hospital-related questions.
2. **Text Preprocessing**: Input text is tokenized and vectorized using TF-IDF.
3. **Intent Classification**: SVM model classifies user input into predefined categories.
4. **Response Generation**: Based on detected intent, a predefined response is returned.
5. **Streamlit UI**: Users interact with the chatbot via a web interface.

## Example Interactions
```
User: Where is the hospital located?
Bot: The Care Medical hospital is located at 123 Wellness Avenue, MedCity, Kochi, India.

User: Can I book an appointment?
Bot: You can book an appointment by calling our reception or visiting our website.
```

## Future Enhancements
- **Deep Learning-based Intent Recognition** (TensorFlow/Keras).
- **Context-aware Conversations**.
- **Integration with Hospital APIs** for real-time data.
- **Multilingual Support**.


