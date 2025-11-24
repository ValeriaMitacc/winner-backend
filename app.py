from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# Modelo ligero
# ---------------------------------------------------------
MODEL_PATH = "model/my_model_light.h5"

# Si el modelo no existe, lo creamos y guardamos
if not os.path.exists(MODEL_PATH):
    model = Sequential([
        Dense(10, activation='relu', input_shape=(5,)),  # capa pequeña
        Dense(1, activation='sigmoid')  # salida
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.save(MODEL_PATH)
else:
    model = load_model(MODEL_PATH)

# ---------------------------------------------------------
# Rutas de la API
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
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
    return "API de predicción ligera funcionando correctamente."

# ---------------------------------------------------------
# Arranque de la app
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
