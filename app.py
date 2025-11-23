from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load model
model = load_model("model/my_model.keras")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        X = np.array([[ 
            data["nivel"],
            data["nota"],
            data["socio"],
            data["motiva"],
            data["respons"]
        ]], dtype=float)

        pred = model.predict(X)[0][0]
        return jsonify({"prediccion": float(pred)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/", methods=["GET"])
def home():
    return "API de predicción funcionando."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
