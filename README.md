# Comparison-and-Optimization-of-Models-for-Real-Estate-Price-Prediction

Ce projet vise à développer un pipeline de prédiction de prix immobiliers basé sur plusieurs modèles de machine learning, en comparant leurs performances et en optimisant les meilleurs d'entre eux. Le but est d'explorer, à travers une méthodologie rigoureuse, les performances de différents algorithmes de régression pour sélectionner les plus adaptés, puis de se concentrer sur des améliorations de modèles spécifiques pour la précision, la stabilité et la résistance au bruit.

Objectifs

1.	Comparaison de Modèles : Comparer plusieurs modèles de régression (Random Forest, SVM, LightGBM, XGBoost) selon le MSE pour évaluer les performances de base sur les données de prix immobiliers.
2.	Optimisation : Optimiser les modèles XGBoost et LightGBM pour un MSE minimal, adapté aux besoins spécifiques de la prédiction de prix.
3.	Analyse de Robustesse : Analyser la variance du MSE et la robustesse des modèles optimisés face au bruit pour garantir une prédiction stable.

Bibliothèques et Ressources Utilisées

•	Python 3.11 : Langage principal pour le développement
•	Pandas : Manipulation de données et gestion des jeux de données
•	Numpy : Calculs numériques et manipulation de tableaux
•	Scikit-learn : Pour les modèles de régression de base (Random Forest, SVM) et le calcul des métriques de performance (MSE)
•	LightGBM : Modèle de gradient boosting optimisé pour les données volumineuses
•	XGBoost : Modèle de boosting performant avec des capacités d’optimisation avancées
•	Matplotlib : Visualisation des résultats de comparaison des modèles
•	California Housing Dataset : Ensemble de données utilisé pour les prévisions de prix de maison

Structure du Projet

1. Préparation des Données
•	Chargement et exploration des données (California Housing Dataset).
•	Séparation en ensembles d'entraînement et de test pour l’évaluation.

2. Comparaison de Modèles
•	Modèles Testés : Régression Linéaire, Régression Ridge, Arbre de Décision, Random Forest, Support Vector Regression
•	Évaluation des Performances : Calcul du MSE pour chaque modèle pour une comparaison directe de précision.
•	Temps d'Entraînement : Mesure du temps nécessaire pour l’entraînement de chaque modèle afin d’évaluer leur efficacité.

3. Optimisation du Modèle Sélectionné Random Forest
•	Hyperparamétrage de LightGBM et XGBoost : Amélioration des performances de ces optimisations du modèle à travers un ajustement des hyperparamètres pour obtenir un MSE optimal.
•	Comparaison avec Random Forest : Analyse de l’amélioration des modèles LightGBM et XGBoost par rapport à la version de base de Random Forest.

4. Analyse de Variance et Résistance au Bruit
•	Variance du MSE : Analyse de la stabilité des modèles optimisés, en examinant la variation du MSE sur plusieurs essais.
•	Test de Résistance au Bruit : Ajout de bruit dans les données de test pour évaluer la robustesse des modèles optimisés (LightGBM et XGBoost) face aux perturbations dans les données.

Fichiers Importants

•	comparaison_modeles.py : Programme de comparaison des modèles avec évaluation du MSE et du temps d'entraînement.
•	optimisation_xgboost_lightgbm.py : Script pour l’optimisation des modèles LightGBM et XGBoost.
•	analyse_variance_robustesse.py : Code pour l'analyse de la variance du MSE et les tests de résistance au bruit.

Ce projet fournit une analyse complète des modèles de prédiction, des étapes de comparaison et optimisation jusqu'à l’évaluation de la robustesse, en vue d'identifier le modèle le plus précis et stable pour des prévisions de prix immobiliers.
