from flask import Flask, request, render_template
import pickle

# Load trained model & vectorizer
model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email = request.form["email"]
        email_vec = vectorizer.transform([email])
        prediction = model.predict(email_vec)[0]
        return render_template("index.html", result="Spam" if prediction == 1 else "Not Spam")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
