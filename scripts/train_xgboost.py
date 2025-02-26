from flask import Flask, send_file, jsonify
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import pickle
import os

app = Flask(__name__)

# 📥 1️⃣ Cargar dataset
data_path = os.path.join("data", "data_FINAL2.csv")  # Ruta flexible para Docker
df = pd.read_csv(data_path)

# 🔄 2️⃣ Convertir "Precio" a numérico y eliminar valores nulos
df['Precio'] = pd.to_numeric(df['Precio'], errors='coerce')
df = df.dropna(subset=['Precio'])

# 🎯 3️⃣ Definir variables predictoras y objetivo
y = df['Precio']
features = ['Marca', 'Modelo', 'Provincia', 'Año', 'Kilometraje', 
            'Transmisión', 'Dirección', 'Motor', 'Tracción', 'Color', 'Combustible']
X = df[features]

# 🔄 4️⃣ Aplicar Label Encoding y guardar los encoders
categorical_cols = ['Marca', 'Modelo', 'Provincia', 'Transmisión', 'Dirección', 'Tracción', 'Color', 'Combustible']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le  # Guardamos el encoder

# 📥 Guardar los encoders
os.makedirs("encoders", exist_ok=True)
encoder_path = os.path.join("encoders", "encoders_xgboost.pkl")
with open(encoder_path, 'wb') as f:
    pickle.dump(encoders, f)

print(f"✅ Encoders guardados en '{encoder_path}'")

# 📌 5️⃣ Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89)

# 🚀 6️⃣ Configurar y entrenar el modelo XGBoost con los mejores parámetros encontrados
best_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=0.7,
    gamma=0,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=700,
    subsample=0.8,
    random_state=82
)

best_xgb.fit(X_train, y_train)

# 📊 7️⃣ Evaluar el modelo
y_pred = best_xgb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Evaluación del Modelo XGBoost:")
print(f"📉 MSE: {mse:.2f}")
print(f"📉 MAE: {mae:.2f}")
print(f"📈 R²: {r2:.4f}")

# 💾 8️⃣ Guardar modelo entrenado
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "modelo_xgboost.pkl")
joblib.dump(best_xgb, model_path)
print(f"\n✅ Modelo guardado en '{model_path}'")

# 🌐 🔥 9️⃣ Servidor Flask para compartir el modelo con la API Web
@app.route("/get_model", methods=["GET"])
def get_model():
    """ Servir el modelo entrenado para que la API Web pueda descargarlo """
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return "❌ Modelo no encontrado", 404

@app.route("/get_encoders", methods=["GET"])
def get_encoders():
    """ Servir los encoders para la API Web """
    if os.path.exists(encoder_path):
        return send_file(encoder_path, as_attachment=True)
    else:
        return "❌ Encoders no encontrados", 404

@app.route("/status", methods=["GET"])
def status():
    """ Endpoint para verificar que la EC2 de entrenamiento está corriendo """
    return jsonify({"status": "running", "message": "Servidor de entrenamiento activo"}), 200

if __name__ == '__main__':
    print("🚀 Servidor de entrenamiento listo en el puerto 5000...")
    app.run(host="0.0.0.0", port=5000)
