from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Define possible intents and responses
intents = {
    "courses": "We offer courses in Computer Science, Mechanical, Civil, and Electronics Engineering.",
    "admission": "Admissions are open from May to July every year.",
    "fees": "The fee structure depends on the course. Contact our office for more details.",
}

# Preprocess data for vectorization
intent_phrases = [
    "What courses do you offer?",
    "Tell me about admissions.",
    "What is the fee structure?",
]

responses = list(intents.values())

# Vectorize intents
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(intent_phrases)

# Simulated NLP chatbot response function
def chatbot_response(user_input):
    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, X)

    # Get the best match based on cosine similarity
    best_match_index = np.argmax(similarity_scores)
    best_match_score = similarity_scores[0][best_match_index]

    # Set a similarity threshold to avoid irrelevant matches
    if best_match_score > 0.3:  # Threshold can be adjusted
        return responses[best_match_index]
    else:
        return "I'm sorry, I don't understand. Can you ask something else?"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
