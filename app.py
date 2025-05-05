import webbrowser
from threading import Timer
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('best_model_random_forest.pkl')
vectorizer = joblib.load('vectorizer.pkl')
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

Timer(1, open_browser).start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        data = [news]
        vect = vectorizer.transform(data)
        prediction = model.predict(vect)
        result = 'FAKE NEWS ❌' if prediction[0] == 1 else 'REAL NEWS ✅'
        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
#http://127.0.0.1:5000/