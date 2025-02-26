import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import pickle
import os

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
os.makedirs("encoders", exist_ok=True)  # Asegurar que la carpeta existe
with open(os.path.join("encoders", "encoders_xgboost.pkl"), 'wb') as f:
    pickle.dump(encoders, f)

# 📌 5️⃣ Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89)

# 🚀 6️⃣ Configurar modelo XGBoost y optimización de hiperparámetros
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=82)

param_grid = {
    'n_estimators': [300, 500, 700],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
}

grid_search = GridSearchCV(estimator=xgb_model, 
                           param_grid=param_grid, 
                           cv=5, 
                           scoring='r2', 
                           n_jobs=-1, 
                           verbose=1)

grid_search.fit(X_train, y_train)

print("\n✅ Mejores parámetros encontrados:", grid_search.best_params_)

# 🏆 7️⃣ Entrenar el mejor modelo
best_xgb = grid_search.best_estimator_

# 📊 8️⃣ Evaluar el modelo
y_pred = best_xgb.predict(X_test)
print("\n📊 Evaluación del Modelo XGBoost:")
print("📉 MSE:", mean_squared_error(y_test, y_pred))
print("📉 MAE:", mean_absolute_error(y_test, y_pred))
print("📈 R²:", r2_score(y_test, y_pred))

# 💾 9️⃣ Guardar modelo entrenado
os.makedirs("models", exist_ok=True)  # Asegurar que la carpeta existe
model_path = os.path.join("models", "modelo_xgboost.pkl")
joblib.dump(best_xgb, model_path)
print(f"\n✅ Modelo guardado en '{model_path}'")

