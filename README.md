# advanced-graph-text-ml

Description du TP

- Objectif général : explorer, manipuler et analyser des graphes, puis mettre en œuvre et comparer des méthodes de détection de communautés et de classification de graphes (approches basées sur noyaux et sur réseaux de neurones graphes).
- Préparation des données : chargement et prétraitement de jeux de données classiques (ex. MUTAG) présents dans `data/` et `datasets/`. Conversion des fichiers bruts (indices de graphe, listes d’arêtes, labels de nœuds/graphes) en formats exploitables par les bibliothèques (PyG / GraKel).
- Part 1 — Exploration : exploration des propriétés des graphes (degrés, distributions, visualisations) à l’aide de NetworkX. Vérification de la qualité et cohérence des données (détection de cas limites, graphes isolés, etc.).
- Part 2 — Détection de communautés : implémentation et comparaison de méthodes de community detection (par ex. Louvain, Girvan–Newman). Évaluation des partitions obtenues avec des métriques pertinentes (modularité, NMI si labels disponibles) et visualisation des résultats.
- Part 3 — Classification de graphes : deux approches comparées :
  - Noyaux de graphes avec GraKel : extraction de caractéristiques par kernels puis entraînement d’un classifieur classique.
  - Réseaux de neurones pour graphes (PyTorch Geometric) : préparation du dataset, définition d’un GNN simple, entraînement et évaluation (accuracy, F1, courbes d’apprentissage).
  Comparaison des performances, du temps d’entraînement et de la complexité des méthodes.
- Expérimentations et évaluation : protocoles (k-fold ou train/test), métriques rapportées (accuracy, precision, recall, F1, AUC si pertinent) et traçage des hyperparamètres testés.
- Environnement et reproductibilité : fichiers de requirements fournis dans `RequirementsFiles/` pour recréer l’environnement (conda/venv). Gestion des gros fichiers (utilisation recommandée de Git LFS si nécessaire).
