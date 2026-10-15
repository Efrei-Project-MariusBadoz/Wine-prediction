# Rapport Technique : Classification de la Qualité du Vin par Apprentissage Supervisé
**Auteur :** Marius BADOZ  
**Projet :** Machine Learning 1 (ING2-BDML)  
**Établissement :** EFREI Paris Panthéon-Assas Université  
**Année académique :** 2026-2027  
**Enseignants référents :** Ahmad TAY, Ahmed-Ghazi BLAIECH, Hassan SHRAIM  

---

## 1. Contexte et Formulation du Problème

L'évaluation de la qualité d'un vin repose traditionnellement sur des jurys d'experts lors de dégustations sensorielles à l'aveugle. Ce procédé présente deux limites pratiques : un coût humain important et une part de subjectivité liée aux dégustateurs. L'objectif de ce projet consiste à concevoir et comparer des algorithmes d'apprentissage supervisé capables de prédire la qualité d'un vin à partir de ses seules mesures physico-chimiques objectives.

Le jeu de données utilisé provient de la base de référence de l'UCI (*Wine Quality Dataset*, Paulo Cortez et al., 2009). Il rassemble 6 497 vins portugais issus de l'appellation d'origine contrôlée *Vinho Verde* (1 599 vins rouges et 4 898 vins blancs). Chaque échantillon dispose de 11 descripteurs physico-chimiques continus et d'une note sensorielle globale comprise entre 0 et 10, attribuée par la médiane d'au moins trois experts.

Conformément au cahier des charges, nous formulons la tâche sous forme d'une classification binaire :
* Classe 1 (*Bon vin*) : Note de qualité supérieure ou égale à 7.
* Classe 0 (*Vin ordinaire*) : Note de qualité strictement inférieure à 7.

---

## 2. Analyse Exploratoire et Préparation des Données

### 2.1 Contrôle d'intégrité et valeurs manquantes
Le jeu de données ne présente aucune valeur manquante (`df.isna().sum() == 0`). Les 6 497 observations sont immédiatement exploitables.

### 2.2 Encodage de la variable catégorielle
Le dataset fusionne deux sous-ensembles : le vin rouge et le vin blanc. Nous avons introduit une variable indicatrice `color` définie par 1 pour le vin rouge et 0 pour le vin blanc (`wine_quality.data.original["color"].map({"red": 1, "white": 0})`).

### 2.3 Échelles de mesure et valeurs aberrantes
L'analyse descriptive révèle de forts écarts d'échelle entre les descripteurs :
* Le dioxyde de soufre total (`total_sulfur_dioxide`) affiche une moyenne de 115.75 mg/L avec un maximum à 440 mg/L.
* Le chlorure de sodium (`chlorides`) présente une moyenne de 0.056 g/L.

L'application de la méthode de l'écart interquartile (IQR avec seuils à $Q1 - 1.5 	imes IQR$ et $Q3 + 1.5 	imes IQR$) détecte des valeurs extrêmes, notamment sur l'acidité volatile (5.80% d'outliers), le sucre résiduel (4.56%) et les chlorures (4.40%). 

Ces valeurs correspondent à des profils biologiques réels (comme les vins doux à forte teneur en sucre ou des cuvées spécifiques) et non à des erreurs de capteurs. Nous avons fait le choix méthodologique de conserver l'intégralité de ces points pour éviter de biaiser la distribution réelle de la production viticole.

### 2.4 Traitement des doublons
Le dataset compte 1 177 lignes en doublon strict (mêmes paramètres chimiques et même note finale). Nous avons vérifié qu'aucun profil chimique identique n'affiche de note divergente. Ces observations proviennent de prélèvements distincts au sein de mêmes cuves de vinification certifiées par la commission viticole. Conformément à la publication de référence de Cortez et al., ces échantillons sont conservés.

### 2.5 Déséquilibre des classes
La distribution de la cible montre une forte asymétrie :
* Classe 0 (Vins ordinaires, notes de 3 à 6) : 5 220 échantillons (80.34%).
* Classe 1 (Bons vins, notes de 7 à 9) : 1 277 échantillons (19.66%).

Ce ratio d'environ 4 contre 1 interdit l'usage de la précision globale (*Accuracy*) comme critère d'optimisation principal. Un classifieur naïf prédisant systématiquement la classe 0 afficherait 80.3% de bonnes réponses sans jamais identifier aucun vin d'exception. Nous orientons notre évaluation sur le Rappel, la Précision, le F1-score et l'aire sous la courbe ROC (AUC).

---

## 3. Analyse des Corrélations et Distributions

### 3.1 Corrélations linéaires (Matrice de Pearson)
Le calcul de la matrice de corrélation met en avant quatre relations majeures avec la qualité :
1. `alcohol` ($r = +0.39$) : Principal facteur associé positivement à une note élevée.
2. `density` ($r = -0.27$) : Facteur corrélé négativement avec la note.
3. `volatile_acidity` ($r = -0.23$) : L'acide acétique apporte un goût aigre de vinaigre, dégradant la note.
4. `chlorides` ($r = -0.20$) : Une concentration élevée en sel nuit à la finesse du vin.

Entre variables explicatives, nous observons une forte corrélation négative entre l'alcool et la densité ($r = -0.69$). L'éthanol ayant une densité inférieure à l'eau (0.789 g/cm³), un vin plus alcoolisé est naturellement moins dense.

### 3.2 Visualisation par Boxplots et Séparabilité
La comparaison des distributions par boxplots confirme ces constats : la médiane du taux d'alcool des bons vins se situe à 11.5%, contre 10.0% pour les vins ordinaires.

Néanmoins, les boîtes et moustaches des deux classes se chevauchent sur chaque variable prise individuellement. Pour un taux d'alcool donné (par exemple 11.0%), on trouve à la fois des vins ordinaires et des bons vins. Ce chevauchement partiel montre qu'un seuil simple ou un hyperplan rigide ne peut pas séparer proprement les deux groupes, justifiant l'emploi de modèles non-linéaires.

---

## 4. Sélection des Variables : Justification du Choix des 8 Descripteurs

L'analyse exploratoire initiale suggérait de ne retenir que les 4 variables les plus fortement corrélées. Pour valider ou infirmer cette réduction, nous avons confronté l'analyse linéaire aux critères de l'Analyse en Composantes Principales (ACP) et de l'Information Mutuelle (MI).

### 4.1 Test F d'ANOVA univarié
Le test $F$ mesure le rapport entre la variance inter-classes et la variance intra-classe. Il confirme la hiérarchie observée sur les corrélations linéaires :
* `alcohol` : $F = 907.4$ ($p < 10^{-100}$)
* `density` : $F = 373.1$ ($p < 10^{-78}$)
* `volatile_acidity` : $F = 299.8$ ($p < 10^{-64}$)
* `chlorides` : $F = 210.5$ ($p < 10^{-46}$)
* Variables secondaires : de $F = 44.1$ (`total_sulfur_dioxide`) à $F = 0.8$ (`pH`).

### 4.2 Le seuil de 90% de variance par l'ACP
L'Analyse en Composantes Principales appliquée sur les variables centrées-réduites fournit la variance cumulée selon le nombre d'axes :
* 1 composante : 31.8%
* 4 composantes : 74.1% (perte critique de 25.9% de l'information physico-chimique)
* 6 composantes : 85.2%
* **8 composantes : 93.8% (franchissement net du seuil standard de 90%)**
* 12 composantes : 100.0%

Ce résultat démontre mathématiquement que limiter la modélisation à 4 variables élimine plus d'un quart de l'information totale du jeu de données. Retenir 8 composantes permet d'atteindre 93.8% de variance expliquée.

### 4.3 Dépendances non-linéaires par l'Information Mutuelle
La corrélation de Pearson ne quantifie que les relations strictement en ligne droite. L'Information Mutuelle évalue la réduction d'entropie globale, englobant les effets non-linéaires et les relations en cloche.
Sur le jeu d'apprentissage, le classement s'établit ainsi :
1. `alcohol` : 0.0924
2. `density` : 0.0848
3. `total_sulfur_dioxide` : 0.0422
4. `chlorides` : 0.0382
5. `citric_acid` : 0.0234
6. `residual_sugar` : 0.0228
7. `volatile_acidity` : 0.0209
8. `free_sulfur_dioxide` : 0.0137
9. `sulphates` : 0.0106
10. `pH` : 0.0053
11. `fixed_acidity` : 0.0015
12. `color` : 0.0000

L'Information Mutuelle met en lumière des descripteurs essentiels complètement sous-évalués par l'analyse linéaire : le dioxyde de soufre total, l'acide citrique et le sucre résiduel. À l'inverse, les 4 dernières variables (`sulphates`, `pH`, `fixed_acidity`, `color`) affichent un apport d'information quasi-nul.

### 4.4 Synthèse et arbitrage de la sélection
En croisant le seuil de 90% de l'ACP et l'Information Mutuelle, nous retenons un ensemble optimal de **8 descripteurs** :
`alcohol`, `density`, `total_sulfur_dioxide`, `chlorides`, `citric_acid`, `residual_sugar`, `volatile_acidity`, `free_sulfur_dioxide`.

Ce sous-ensemble conserve 93.8% de la variance, intègre les signaux non-linéaires majeurs et écarte les 4 variables parasites qui perturbaient le calcul des distances géométriques.

---

## 5. Modélisation et Comparaison des Algorithmes (Pipeline 8 Variables)

### 5.1 Protocole expérimental
Les données sont découpées avec `train_test_split` :
* 80% en apprentissage ($N = 5\,197$)
* 20% en test indépendant ($N = 1\,300$)
* Stratification sur la variable cible (`stratify=y_data`) pour maintenir la proportion exacte de 19.66% de bons vins dans les deux ensembles.
* Normalisation par `StandardScaler` ajusté sur les 8 variables d'apprentissage puis appliqué sur l'ensemble de test, évitant tout risque de fuite de données (*data leakage*).

### 5.2 Propriétés théoriques des cinq modèles étudiés
1. **Régression Logistique :** Modèle linéaire paramétrique. Il calcule une combinaison linéaire $z = w^T x + b$ transformée par la fonction sigmoïde $\sigma(z) = 1 / (1 + e^{-z})$. Sa frontière est un hyperplan rigide.
2. **k-NN (k Plus Proches Voisins) :** Méthode non-paramétrique géométrique. Aucune fonction explicite n'est apprise pendant l'entraînement. Lors du test, l'algorithme calcule les distances dans l'espace à 8 dimensions et effectue un vote parmi les $k$ plus proches voisins.
3. **Arbre de Décision (CART) :** Modèle non-linéaire partitionnant l'espace de manière récursive avec des coupures orthogonales aux axes. Critère de séparation fixé sur l'entropie, profondeur maximale limitée à 5 pour restreindre le surapprentissage.
4. **Naive Bayes Gaussien :** Classifieur probabiliste appliquant le théorème de Bayes sous l'hypothèse d'indépendance conditionnelle des descripteurs sachant la classe.
5. **Support Vector Machine (SVM) :** Recherche l'hyperplan séparateur garantissant la marge géométrique maximale entre les vecteurs de support. L'emploi d'un noyau gaussien RBF (*Radial Basis Function*) projette implicitement les observations dans un espace de dimension supérieure.

### 5.3 Résultats des modèles de base (Test set, N = 1 300, 8 variables)

| Modèle | Accuracy | Précision | Rappel | F1-score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Régression Logistique | 0.819 | 0.598 | 0.238 | 0.341 | 0.790 |
| k-NN (k=5, uniforme) | 0.829 | 0.576 | 0.504 | 0.538 | 0.826 |
| Arbre de Décision (p=5, entropie) | 0.825 | 0.605 | 0.316 | 0.415 | 0.813 |
| Naive Bayes Gaussien | 0.752 | 0.414 | 0.629 | 0.499 | 0.755 |
| SVM (noyau RBF initial) | 0.823 | 0.658 | 0.211 | 0.320 | 0.799 |

### 5.4 Analyse critique des résultats initiaux
* La Régression Logistique et le SVM de base favorisent la classe majoritaire en raison de la rigidité de leur frontière initiale : ils obtiennent un rappel très bas (23.8% et 21.1%), laissant passer les trois quarts des bons vins.
* Naive Bayes présente le rappel initial le plus haut (62.9%), mais sa précision reste médiocre à 41.4% en raison des corrélations résiduelles violant l'hypothèse d'indépendance.
* k-NN s'avère le plus performant et équilibré dès sa version initiale : il combine le meilleur F1-score (0.538) et un rappel supérieur à 50%.

---

## 6. Optimisation des Hyperparamètres

Nous avons sélectionné les deux familles d'algorithmes présentant le plus fort potentiel d'adaptation non-linéaire : le k-NN et le SVM.

### 6.1 Optimisation du k-NN par GridSearchCV
Une recherche exhaustive sur grille avec validation croisée stratifiée à 5 plis (`cv=5`, métrique d'optimisation `scoring="f1"`) a testé les combinaisons suivantes :
* `n_neighbors` : [3, 5, 7, 9, 11]
* `weights` : ['uniform', 'distance']
* `metric` : ['euclidean', 'manhattan']

Le meilleur estimateur sélectionné sur 8 variables est :
* `n_neighbors = 7`
* `weights = 'distance'`
* `metric = 'euclidean'`

Le passage au vote pondéré par l'inverse de la distance ($w_i = 1 / d$) apporte un gain déterminant : au lieu d'accorder une voix égale à chaque voisin, un voisin très proche chimiquement influence bien plus la décision qu'un voisin situé en périphérie de voisinage.

### 6.2 Optimisation du SVM par RandomizedSearchCV
Une recherche aléatoire sur 10 itérations avec validation croisée à 5 plis a exploré les paramètres suivants :
* `C` : [0.1, 1, 5, 10, 20]
* `gamma` : ['scale', 'auto', 0.01, 0.05, 0.1]

La meilleure configuration retenue est $C = 10$ avec $\gamma = 	ext{'auto'}$. L'augmentation du paramètre de régularisation $C$ pénalise plus sévèrement les erreurs de classification de la classe minoritaire, remontant le rappel de 21.1% à 29.3%.

### 6.3 Comparatif final après optimisation (8 variables)

| Modèle Optimisé | Accuracy | Précision | Rappel | F1-score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SVM Optimisé ($C=10, \gamma=	ext{auto}$) | 0.827 | 0.630 | 0.293 | 0.400 | 0.820 |
| **k-NN Optimisé ($k=7, 	ext{distance}$)** | **0.867** | **0.683** | **0.605** | **0.642** | **0.890** |

Le modèle k-NN optimisé surclasse très largement le SVM sur l'ensemble des indicateurs.

### 6.4 Matrice de confusion du k-NN retenu ($N = 1\,300$)
* Vrais Négatifs (Vins ordinaires correctement détectés) : **972**
* Faux Positifs (Vins ordinaires pris à tort pour de bons vins) : **72**
* Faux Négatifs (Bons vins manqués) : **101**
* Vrais Positifs (Bons vins correctement détectés) : **155**

La précision atteint 68.3% ($155 / (155 + 72)$) et le rappel s'établit à 60.55% ($155 / (155 + 101)$), avec une aire sous la courbe ROC de 0.890.

---

## 7. Déploiement et Livrables Bonus

Conformément à la section 2.5 du sujet, la solution a été industrialisée selon les standards du génie logiciel :

1. **Sérialisation des modèles :**  
   Le modèle k-NN optimisé et son objet de normalisation ont été persistés sous forme de fichiers binaires à l'aide de la bibliothèque `joblib` (`best_knn_model.joblib` et `scaler.joblib`) sur la base des 8 variables.
2. **Tests unitaires automatisés (`pytest`) :**  
   Le fichier `tests/test_model.py` intègre quatre tests automatisés :
   * Vérification de la présence physique des fichiers d'artefacts.
   * Contrôle du bon chargement et de la dimension attendue (8 descripteurs).
   * Validation de la cohérence des prédictions (format binaire 0 ou 1, somme des probabilités égale à 1.0).
   * Gestion des erreurs lors d'un apport incomplet de variables.
3. **Application web interactive (`Streamlit`) :**  
   L'interface `app.py` propose 8 curseurs ajustables pour renseigner la composition chimique d'un vin et fournit instantanément la décision de classification accompagnée de la jauge de probabilité.
4. **Conteneurisation (`Docker`) :**  
   Le `Dockerfile` et le fichier `docker-compose.yml` permettent un déploiement autonome en une commande sans nécessiter l'installation préalable de l'environnement Python sur la machine hôte.

---

## 8. Conclusion

Ce projet illustre l'importance de dépasser l'analyse purement linéaire en prétraitement. L'Analyse en Composantes Principales (seuil de 90%) et l'Information Mutuelle ont permis d'isoler 8 variables prédictives clés, écartant le bruit des variables non informatives tout en réhabilitant des descripteurs non-linéaires cruciaux comme le dioxyde de soufre et l'acide citrique.

Parmi les architectures comparées, le modèle des $k$ plus proches voisins avec vote pondéré par l'inverse de la distance s'est imposé comme la méthode la plus performante, atteignant un F1-score de 0.642, un rappel de 60.55% et une aire sous la courbe ROC record de 0.890.

Comme perspective d'amélioration, l'entraînement de méthodes d'ensemble comme *Random Forest* ou le rééchantillonnage synthétique (*SMOTE*) permettrait d'explorer un gain additionnel sur la détection des cas limites.
