from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import pandas as pd
import json
import requests  # 🔥 Para descargar el modelo desde la otra EC2
from collections import defaultdict
import os

app = Flask(__name__)

# 🔥 **IP PÚBLICA de la EC2 de entrenamiento**
TRAINING_SERVER_IP = "http://ec2-44-221-50-62.compute-1.amazonaws.com:5000"  # ⚠️ Reemplázala con la IP real de tu EC2 de entrenamiento

# ✅ **Función para descargar el modelo actualizado**
def download_model():
    model_url = f"{TRAINING_SERVER_IP}/get_model"
    encoder_url = f"{TRAINING_SERVER_IP}/get_encoders"
    status_url = f"{TRAINING_SERVER_IP}/status"

    model_path = "models/modelo_xgboost.pkl"
    encoder_path = "encoders/encoders_xgboost.pkl"

    os.makedirs("models", exist_ok=True)
    os.makedirs("encoders", exist_ok=True)

    try:
        # Verificar si la EC2 de entrenamiento está activa
        response = requests.get(status_url, timeout=5)
        if response.status_code != 200:
            print("⚠️ La EC2 de entrenamiento no está activa. No se descargará el modelo.")
            return False  # Devuelve False si no se pudo actualizar

        # Descargar el modelo
        print("⬇️ Descargando modelo actualizado...")
        model_response = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(model_response.content)

        # Descargar los encoders
        print("⬇️ Descargando encoders actualizados...")
        encoder_response = requests.get(encoder_url)
        with open(encoder_path, "wb") as f:
            f.write(encoder_response.content)

        print("✅ Modelo y encoders descargados correctamente.")
        return True  # Devuelve True si la actualización fue exitosa

    except Exception as e:
        print(f"⚠️ Error al descargar el modelo o encoders: {e}")
        return False  # Devuelve False si hubo un error

# 🔥 **Descargar modelo al iniciar la API**
download_model()

# 📥 Cargar el modelo entrenado y los encoders guardados
MODEL_PATH = "models/modelo_xgboost.pkl"
ENCODER_PATH = "encoders/encoders_xgboost.pkl"
DATA_PATH = "data/data_FINAL2.csv"

modelo_rf = joblib.load(MODEL_PATH)
with open(ENCODER_PATH, 'rb') as f:
    encoders = pickle.load(f)

# 🔄 Definir las columnas categóricas y el orden de las features
categorical_cols = ['Marca', 'Modelo', 'Provincia', 'Transmisión', 'Dirección', 'Tracción', 'Color', 'Combustible']
features = ['Marca', 'Modelo', 'Provincia', 'Año', 'Kilometraje', 'Transmisión', 'Dirección', 'Motor', 'Tracción', 'Color', 'Combustible']

# 📋 Preparar las opciones para los combo boxes
dropdown_options = {col: list(encoders[col].classes_) for col in categorical_cols}

# 🔄 Generar un mapeo de Marca -> Modelos
df_data = pd.read_csv(DATA_PATH)
marca_modelo_map = defaultdict(list)
for _, row in df_data[['Marca', 'Modelo']].drop_duplicates().iterrows():
    marca = row['Marca']
    modelo = row['Modelo']
    marca_modelo_map[marca].append(modelo)

# Convertir el mapeo a JSON para enviarlo al frontend
marca_modelo_json = json.dumps(dict(marca_modelo_map))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_data = {}

    if request.method == "POST":
        # 📥 Recoger los datos del formulario
        data = {col: request.form.get(col) for col in features}
        form_data = data

        # 📌 Convertir campos numéricos
        df_input = pd.DataFrame([data])
        df_input['Año'] = pd.to_numeric(df_input['Año'], errors='coerce')
        df_input['Kilometraje'] = pd.to_numeric(df_input['Kilometraje'], errors='coerce')
        df_input['Motor'] = pd.to_numeric(df_input['Motor'], errors='coerce')

        # 🔄 Aplicar Label Encoding a las columnas categóricas
        for col in categorical_cols:
            if df_input.loc[0, col] not in encoders[col].classes_:
                df_input.loc[0, col] = encoders[col].classes_[0]
            df_input[col] = encoders[col].transform(df_input[col].astype(str))

        df_input = df_input[features]
        prediction = modelo_rf.predict(df_input)[0]

    return render_template("index.html",
                           dropdown_options=dropdown_options,
                           prediction=prediction,
                           form_data=form_data,
                           marca_modelo_json=marca_modelo_json)

# 🔥 **Endpoint para actualizar manualmente el modelo**
@app.route("/update_model", methods=["GET"])
def update_model():
    """ Permite descargar manualmente el modelo actualizado desde la EC2 de entrenamiento """
    success = download_model()
    if success:
        global modelo_rf, encoders
        modelo_rf = joblib.load(MODEL_PATH)
        with open(ENCODER_PATH, 'rb') as f:
            encoders = pickle.load(f)
        return jsonify({"status": "success", "message": "Modelo actualizado correctamente"}), 200
    else:
        return jsonify({"status": "error", "message": "No se pudo actualizar el modelo"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # ✅ Escuchar en todas las interfaces para Docker
