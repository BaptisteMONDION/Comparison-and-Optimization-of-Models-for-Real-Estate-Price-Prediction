import pandas as pd

# Charger les données CSV
file_path = 'ton_fichier.csv'
data = pd.read_csv(file_path, sep=';')  # Remplacer 'sep' par le bon séparateur

# Afficher un aperçu des données
print(data.head())

# Vérifier les colonnes et corriger les espaces dans les noms des colonnes
data.columns = data.columns.str.strip()

# Remplir les valeurs manquantes pour 'Montant_Vente' par la moyenne
data['Montant_Vente'].fillna(data['Montant_Vente'].mean(), inplace=True)

# Afficher le DataFrame mis à jour
print(data.head())
