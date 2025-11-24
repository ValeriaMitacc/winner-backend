from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import os  # Para leer la variable de entorno PORT

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# Cargar el modelo
# Asegúrate de que la carpeta y el archivo existan con este nombre exacto
model = load_model("model/my_model.keras")  

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Mejor que request.json para evitar advertencias

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
    return "API de predicción funcionando correctamente."

if __name__ == "_main_":
    # Render asigna el puerto vía variable de entorno
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    