from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("xgb_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    open_price = float(request.form["Open"])
    high_price = float(request.form["High"])
    low_price = float(request.form["Low"])
    volume = float(request.form["Volume"])

    data = np.array([[open_price, high_price, low_price, volume]])

    prediction = model.predict(data)

    return render_template("index.html", prediction_text=f"Predicted Close Price: {prediction[0]}")

if __name__ == "__main__":
    app.run(debug=True)