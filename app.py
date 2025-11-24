from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Cargar modelo ligero
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
        pred = float(model.predict(X, verbose=0)[0][0])
        return jsonify({"prediccion": pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def home():
    return "API de predicción ligera funcionando correctamente."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
