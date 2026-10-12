# Classification de la Qualité du Vin

**Auteurs / Enseignants :** Ahmad TAY, Ahmed-Ghazi BLAIECH, Hassan SHRAIM  
**Cours :** Machine Learning 1  
**Année académique :** 2026-2027  

---

## 1. Contexte et Objectif

La qualité d'un vin est influencée par de nombreux paramètres physico-chimiques tels que l'acidité, le pH, le taux d'alcool ou encore la teneur en sucre résiduel.

L'objectif de ce projet est de développer un modèle de **classification supervisée** capable de prédire la **qualité du vin** (sur une échelle de 0 à 10, ou sous forme binaire : *bon/mauvais*) à partir de ces caractéristiques mesurables.

Le jeu de données à utiliser se trouve ici :  
🔗 [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

---

## 2. Modalités du projet

### 2.1 Prétraitement des données
- **Chargement des données** et statistiques descriptives.
- **Gestion des valeurs manquantes** et détection des valeurs aberrantes (outliers).
- **Encodage correct** des variables catégorielles (si présentes).
- **Transformation du problème en classification binaire** : $\text{qualité} \ge 7 \implies \text{bon vin}$.
- **Analyse de la distribution des classes** pour vérifier l'équilibre du jeu de données (imbalanced classes).
- **Analyse des corrélations** entre les variables.
- **Visualisations** : matrice de corrélation (heatmap), histogrammes, boxplots.
- **Sélection de variables** : utilisation de méthodes de sélection basées sur les méthodes vues en cours.

### 2.2 Modélisation et évaluation
- **Modèles envisagés** :
  - Régression logistique
  - $k$-NN ($k$-Nearest Neighbors)
  - Arbre de décision (Decision Tree)
  - Naive Bayes
  - SVM (Support Vector Machine)
- **Justification du choix des modèles** selon :
  - Leurs propriétés théoriques (linéaires vs non linéaires).
  - Leur adaptabilité au type de données (distributions continues, bruit, etc.).

### 2.3 Évaluation des modèles
- **Évaluation à l'aide de métriques pertinentes** :
  - Accuracy, précision, rappel (recall), F1-score.
  - Matrice de confusion.
  - Courbe ROC et score AUC.
- **Comparaison des performances** et interprétation des résultats.
- **Identification du modèle le plus performant** selon un compromis biais/variance.

### 2.4 Optimisation des hyperparamètres
- Utilisation de techniques d'optimisation telles que :
  - **Grid Search** : recherche exhaustive sur une grille d'hyperparamètres.
  - **Random Search** : recherche aléatoire pour un gain de temps.
- **Validation croisée ($k$-fold cross-validation)** pour l'évaluation des modèles optimisés.

### 2.5 Bonus
- Mettre votre solution sur **GitHub** ou **GitLab**.
- Ajouter des **tests unitaires**.
- **Conteneuriser** le code (ex: Docker) pour qu'il puisse être exécuté facilement par un tiers.
- Mettre en place une **application web** (ex: Streamlit / FastAPI) qui donne la prédiction pour une observation donnée en entrée.

---

## 3. Livrables attendus
- **Rapport technique** détaillant les étapes de traitement, d'analyse et de modélisation.
- **Notebook Python** (Jupyter / Google Colab) contenant le code complet et commenté.
