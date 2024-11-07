import numpy as np
import matplotlib.pyplot as plt
import time  # Pour mesurer le temps d'exécution
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# Charger le dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Échantillonner les données si nécessaire (ex. : utiliser 10% des données)
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, random_state=42)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Initialiser les modèles
models = {
    "Régression Linéaire": LinearRegression(),
    "Régression Ridge": Ridge(alpha=1.0),  # Exemple d'hyperparamètre alpha
    "Régression Lasso": Lasso(alpha=0.1),  # Exemple d'hyperparamètre alpha
    "Arbre de Décision": DecisionTreeRegressor(max_depth=5),  # Exemple d'hyperparamètre max_depth
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5),  # n_estimators et max_depth
}

# Dictionnaire pour stocker les MSE et le temps d'exécution
mse_results = {}
execution_times = {model_name: [] for model_name in models}  # Liste pour stocker les temps d'exécution

# Nombre d'itérations pour chaque modèle
n_iterations = 5

# Entraîner et évaluer chaque modèle plusieurs fois
for model_name, model in models.items():
    for _ in range(n_iterations):
        start_time = time.time()  # Démarrer le chronomètre
        model.fit(X_train, y_train)  # Entraînement
        y_pred = model.predict(X_test)  # Prédiction
        mse = mean_squared_error(y_test, y_pred)  # Calculer le MSE
        
        # Stocker le MSE et le temps d'exécution
        mse_results[model_name] = mse  # On garde le dernier MSE (optionnel)
        execution_times[model_name].append(time.time() - start_time)  # Temps d'exécution

# Afficher les résultats
for model_name, mse in mse_results.items():
    print(f"{model_name}: Mean Squared Error = {mse:.2f}, Temps d'exécution = {execution_times[model_name]} secondes")

# Visualisation des résultats : MSE en abscisse et Temps d'exécution en ordonnée
plt.figure(figsize=(10, 6))

# Boucle pour afficher chaque temps d'exécution
for model_name in execution_times:
    plt.scatter([mse_results[model_name]] * n_iterations, execution_times[model_name], label=model_name)

plt.xlabel('Mean Squared Error (MSE)')
plt.ylabel('Temps d\'exécution (secondes)')
plt.title('Comparaison des modèles de régression : MSE vs Temps d\'exécution')
plt.grid()
plt.legend()  # Afficher la légende

plt.show()  # Affiche le graphique
plt.pause(0.001)  # Ajoute un court délai
