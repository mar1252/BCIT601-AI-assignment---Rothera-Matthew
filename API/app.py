from flask import Flask, jsonify, request
import joblib
from pathlib import Path

model_path = Path(__file__).resolve().parent.parent /"model"
app = Flask(__name__)


#loading model and vectorizer from disk with joblib.

model = joblib.load(model_path / "model.joblib")
vectorizer = joblib.load(model_path / "vectorizer.joblib")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'api is working'})


@app.route('/categories', methods=['GET'])
def get_categories():
    categories = [str(category) for category in model.classes_]
    return jsonify({'categories:': categories})


@app.route('/predict', methods=['POST'])
def predict():
   data = request.get_json()
   issue = data.get("request_text")
   issue_tfidf = vectorizer.transform([issue])
   predict = model.predict(issue_tfidf)
   return jsonify({'category:': predict[0]}), 200


@app.route('/')
def hello_world():
    return "Hello world"

if __name__ == '__main__':
    app.run(debug=True)
    app.run()

