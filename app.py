from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(_name_)
CORS(app)  # HABILITA CORS PARA TODAS LAS RUTAS

# Cargar el modelo
model = load_model("model/my_model.keras")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        # Recibir variables EXACTAS que tu frontend envía
        X = np.array([[ 
            data["nivel"],
            data["nota"],
            data["socio"],
            data["motiva"],
            data["respons"]
        ]], dtype=float)

        # Realizar la predicción
        pred = model.predict(X)[0][0]

        return jsonify({"prediccion": float(pred)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/", methods=["GET"])
def home():
    return "API de predicción funcionando."

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=10000)