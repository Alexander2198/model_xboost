from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import requests
from collections import defaultdict
import os
import pickle

app = Flask(__name__)

# 🔥 **IP PÚBLICA de la EC2 de entrenamiento**
TRAINING_SERVER_IP = "http://54.204.86.64:5000"

# 📌 Rutas de los archivos de modelo y encoders
MODEL_PATH = "models/modelo_xgboost.pkl"
ENCODER_PATH = "encoders/encoders_xgboost.pkl"
DATA_PATH = "data/data_FINAL2.csv"

# 📥 **Función para descargar el modelo y encoders si no existen**
def ensure_model_files():
    """Descarga el modelo y encoders si no están presentes en el contenedor."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("⬇️ Descargando modelo y encoders...")
        model_url = f"{TRAINING_SERVER_IP}/get_model"
        encoder_url = f"{TRAINING_SERVER_IP}/get_encoders"
        os.makedirs("models", exist_ok=True)
        os.makedirs("encoders", exist_ok=True)
        try:
            model_response = requests.get(model_url, timeout=10)
            with open(MODEL_PATH, "wb") as f:
                f.write(model_response.content)

            encoder_response = requests.get(encoder_url, timeout=10)
            with open(ENCODER_PATH, "wb") as f:
                f.write(encoder_response.content)

            print("✅ Modelo y encoders descargados correctamente.")
        except Exception as e:
            print(f"⚠️ Error al descargar modelo/encoders: {e}")

# 🔄 **Función para cargar el modelo bajo demanda**
def load_model():
    """Carga el modelo solo cuando se necesite."""
    if not os.path.exists(MODEL_PATH):
        ensure_model_files()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# 🔄 **Función para cargar los encoders bajo demanda**
def load_encoders():
    """Carga los encoders solo cuando se necesiten."""
    if not os.path.exists(ENCODER_PATH):
        ensure_model_files()
    with open(ENCODER_PATH, 'rb') as f:
        return pickle.load(f)

# 📋 **Cargar información para los dropdowns (Combo Box)**
df_data = pd.read_csv(DATA_PATH)
categorical_cols = ['Marca', 'Modelo', 'Provincia', 'Transmisión', 'Dirección', 'Tracción', 'Color', 'Combustible']
features = ['Marca', 'Modelo', 'Provincia', 'Año', 'Kilometraje', 'Transmisión', 'Dirección', 'Motor', 'Tracción', 'Color', 'Combustible']
dropdown_options = {col: df_data[col].unique().tolist() for col in categorical_cols}

# 🔄 **Generar un mapeo de Marca → Modelos**
marca_modelo_map = defaultdict(list)
for _, row in df_data[['Marca', 'Modelo']].drop_duplicates().iterrows():
    marca = row['Marca']
    modelo = row['Modelo']
    marca_modelo_map[marca].append(modelo)

marca_modelo_json = json.dumps(dict(marca_modelo_map))

# 🔥 **Página Principal**
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_data = {}

    if request.method == "POST":
        # 📥 Recoger datos del formulario
        data = {col: request.form.get(col) for col in features}
        form_data = data

        # 📌 Convertir valores numéricos
        df_input = pd.DataFrame([data])
        df_input['Año'] = pd.to_numeric(df_input['Año'], errors='coerce')
        df_input['Kilometraje'] = pd.to_numeric(df_input['Kilometraje'], errors='coerce')
        df_input['Motor'] = pd.to_numeric(df_input['Motor'], errors='coerce')

        # 🔄 Cargar encoders dinámicamente
        encoders = load_encoders()

        # 🔄 Aplicar Label Encoding a las columnas categóricas
        for col in categorical_cols:
            if df_input.loc[0, col] not in encoders[col].classes_:
                df_input.loc[0, col] = encoders[col].classes_[0]
            df_input[col] = encoders[col].transform(df_input[col].astype(str))

        df_input = df_input[features]

        # 🚀 Cargar el modelo dinámicamente y hacer la predicción
        modelo_rf = load_model()
        prediction = modelo_rf.predict(df_input)[0]

    return render_template("index.html",
                           dropdown_options=dropdown_options,
                           prediction=prediction,
                           form_data=form_data,
                           marca_modelo_json=marca_modelo_json)

# 🔥 **Endpoint para actualizar manualmente el modelo**
@app.route("/update_model", methods=["GET"])
def update_model():
    """Permite descargar manualmente el modelo actualizado desde la EC2 de entrenamiento."""
    ensure_model_files()
    return jsonify({"status": "success", "message": "Modelo actualizado correctamente"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # ✅ Escuchar en todas las interfaces para Docker
