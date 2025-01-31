from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'lakshmi'
app.config['MYSQL_DB'] = 'college_chatbot'

mysql = MySQL(app)

# Secret key for session management
app.secret_key = 'your_secret_key'

# Define possible intents and responses
intents = {
    "courses": "We offer courses in Computer Science, Mechanical, Civil, and Electronics Engineering.",
    "admission": "Admissions are open from May to July every year.",
    "fees": "The fee structure depends on the course. Contact our office for more details.",
    # Add more intents and responses as needed
}

# Preprocess data for vectorization
intent_phrases = [
    "What courses do you offer?",
    "Tell me about admissions.",
    "What is the fee structure?",
    # Add more phrases as needed
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
    if 'loggedin' in session:
        return render_template("index.html")
    return redirect(url_for('login'))

@app.route("/login", methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password,))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            return redirect(url_for('index'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password:
            msg = 'Please fill out the form!'
        else:
            cursor.execute('INSERT INTO users VALUES (NULL, %s, %s)', (username, password,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        msg = 'Please fill out the form!'
    return render_template('signup.html', msg=msg)

@app.route("/logout")
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)