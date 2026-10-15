# Classification de la Qualité du Vin (Machine Learning 1)

Projet académique réalisé dans le cadre du cursus d'ingénieur à l'**EFREI Paris Panthéon-Assas Université** (Promotion 2026-2027, ING2-BDML).

**Enseignants référents :** Ahmad TAY, Ahmed-Ghazi BLAIECH, Hassan SHRAIM.

---

## 1. Présentation du Projet

L'objectif de ce projet est de concevoir un système de **classification supervisée** capable d'évaluer la qualité d'un vin à partir de ses caractéristiques physico-chimiques objectives.

Le jeu de données utilisé est le **UCI Wine Quality Dataset** (Paulo Cortez et al., 2009), regroupant 6 497 vins portugais de l'appellation *Vinho Verde* (1 599 rouges et 4 898 blancs).

La tâche est modélisée sous forme de classification binaire :
* **Classe 1 (Grand Vin) :** Note sensorielle $\ge 7$ (1 277 observations, 19.66%).
* **Classe 0 (Vin Ordinaire) :** Note sensorielle $< 7$ (5 220 observations, 80.34%).

### Sélection méthodologique des variables
En combinant le **seuil de 90% de variance expliquée de l'ACP** (93.8% atteints à 8 composantes) et l'analyse de l'**Information Mutuelle** (qui capte les dépendances non-linéaires négligées par la corrélation linéaire), nous retenons un ensemble optimal de **8 descripteurs physico-chimiques** :
`alcohol`, `density`, `total_sulfur_dioxide`, `chlorides`, `citric_acid`, `residual_sugar`, `volatile_acidity`, `free_sulfur_dioxide`.

---

## 2. Structure du Répertoire

```text
├── wine_quality.ipynb          # Notebook Jupyter complet (pipeline 8 variables)
├── rapport_technique.md        # Rapport d'ingénierie complet (8 sections détaillées)
├── presentation_qualite_vin.pptx # Diapositives académiques de soutenance (format 16:9)
├── best_knn_model.joblib       # Pipeline k-NN optimisé sérialisé (8 variables)
├── scaler.joblib               # StandardScaler ajusté sur les 8 descripteurs
├── app.py                      # Application web interactive Streamlit (8 curseurs)
├── Dockerfile                  # Configuration de conteneurisation
├── docker-compose.yml          # Déploiement multi-conteneur simplifié
├── requirements.txt            # Dépendances Python nécessaires
├── tests/
│   └── test_model.py           # Suite de tests unitaires automatisés (pytest)
├── Project_2_efrei_v2026_2027_ML.pdf # Sujet officiel du projet
└── Project_2_efrei_v2026_2027_ML.md  # Sujet au format Markdown
```

---

## 3. Synthèse des Résultats Expérimentaux (Pipeline 8 Variables)

Les modèles ont été évalués sur un ensemble de test indépendant de 1 300 observations (20% du jeu total, partition stratifiée).

| Modèle | Accuracy | Précision | Rappel | F1-score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Régression Logistique | 0.819 | 0.598 | 0.238 | 0.341 | 0.790 |
| Arbre de Décision (p=5, entropie) | 0.825 | 0.605 | 0.316 | 0.415 | 0.813 |
| Naive Bayes Gaussien | 0.752 | 0.414 | 0.629 | 0.499 | 0.755 |
| SVM (noyau RBF initial) | 0.823 | 0.658 | 0.211 | 0.320 | 0.799 |
| SVM Optimisé (C=10, gamma=auto) | 0.827 | 0.630 | 0.293 | 0.400 | 0.820 |
| **k-NN Optimisé (k=7, pondération distance)** | **0.867** | **0.683** | **0.605** | **0.642** | **0.890** |

Le modèle des **k plus proches voisins avec vote pondéré par l'inverse de la distance** ($k=7$) s'impose sur l'ensemble des critères. Il détecte 155 vrais bons vins sur 256 avec une précision de 68.3%, un F1-score de 0.642 et une aire sous la courbe ROC de 0.890.

---

## 4. Installation et Utilisation

### Prérequis
* Python 3.9 ou supérieur
* Git

### Installation locale
```bash
git clone https://github.com/votre-compte/Machine-Learning.git
cd Machine-Learning
python3 -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### Lancement des tests unitaires
Les tests automatisés vérifient l'intégrité des artefacts sérialisés et le comportement fonctionnel du modèle sur les 8 variables :
```bash
python3 -m pytest tests/test_model.py
```

### Lancement de l'application web Streamlit
Une interface web permet de tester la prédiction en modifiant les 8 curseurs chimiques retenus :
```bash
streamlit run app.py
```
L'interface est accessible sur `http://localhost:8501`.

### Déploiement avec Docker
Pour exécuter l'application web dans un conteneur isolé :
```bash
docker compose up --build
```

---

## 5. Livrables Inclus

* **Notebook interactif :** [`wine_quality.ipynb`](wine_quality.ipynb), exécuté sans aucun commentaire parasite dans le code, structuré avec titres et commentaires méthodologiques.
* **Rapport technique écrit :** [`rapport_technique.md`](rapport_technique.md), couvrant l'analyse exploratoire, la justification du choix des 8 variables (seuil de 90% d'ACP et Information Mutuelle) et le benchmark complet des algorithmes.
* **Support de présentation :** [`presentation_qualite_vin.pptx`](presentation_qualite_vin.pptx), diapositives au format 16:9 en typographie Times New Roman sobre, intégrant les tableaux réels et la matrice de confusion.
* **Industrialisation :** Modèles sérialisés (`joblib`), application web (`Streamlit`), tests (`pytest`) et conteneurisation (`Docker`).
