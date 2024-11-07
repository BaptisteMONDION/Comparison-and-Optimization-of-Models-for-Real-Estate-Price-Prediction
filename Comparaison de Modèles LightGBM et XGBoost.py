import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Charger les données (California Housing Dataset)
data = fetch_california_housing(as_frame=True)
X = data['data']
y = data['target']

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle LightGBM
lgb_train_data = lgb.Dataset(X_train, label=y_train)

params_lgb = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'verbose': -1
}

print("Training LightGBM model...")
start_time = time.time()
model_lgb = lgb.train(params_lgb, lgb_train_data, num_boost_round=100)
lgb_train_time = time.time() - start_time

# Prédictions et évaluation pour LightGBM
y_pred_lgb = model_lgb.predict(X_test)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)

# Entraîner un modèle XGBoost
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

print("Training XGBoost model...")
start_time = time.time()
model_xgb.fit(X_train, y_train)
xgb_train_time = time.time() - start_time

# Prédictions et évaluation pour XGBoost
y_pred_xgb = model_xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)

# Afficher les résultats
print(f"LightGBM - MSE: {mse_lgb:.4f}, Training time: {lgb_train_time:.4f} seconds")
print(f"XGBoost - MSE: {mse_xgb:.4f}, Training time: {xgb_train_time:.4f} seconds")

# Visualisation des résultats
models = ['LightGBM', 'XGBoost']
mse_scores = [mse_lgb, mse_xgb]
train_times = [lgb_train_time, xgb_train_time]

fig, ax1 = plt.subplots(figsize=(10, 6))  # Taille de la figure

# Créer les barres pour le MSE
bars = ax1.bar(models, mse_scores, color='tab:red', alpha=0.6, label='MSE')
ax1.set_ylabel('Mean Squared Error (MSE)', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.set_title('Comparaison des performances de LightGBM et XGBoost', fontsize=16)

# Ajouter les valeurs de MSE au-dessus des barres
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", ha='center', va='bottom')

# Créer une seconde axe pour le temps d'entraînement
ax2 = ax1.twinx()
ax2.plot(models, train_times, color='tab:blue', marker='o', label='Temps d\'entraînement', linewidth=2)
ax2.set_ylabel('Temps d\'entraînement (s)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Ajouter une légende
fig.tight_layout()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Afficher la figure
plt.show()
