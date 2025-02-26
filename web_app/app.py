from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd
import json
from collections import defaultdict

app = Flask(__name__)

# ✅ Ajuste de rutas para que funcione en Docker
MODEL_PATH = "models/modelo_xgboost.pkl"
ENCODER_PATH = "encoders/encoders_xgboost.pkl"
DATA_PATH = "data/data_FINAL2.csv"

# 📥 Cargar el modelo entrenado y los encoders guardados
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # ✅ Escuchar en todas las interfaces para Docker
