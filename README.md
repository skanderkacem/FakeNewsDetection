# 🔍 FakeNewsDetection - Système de Détection de Fausses Nouvelles Multimodal

Un projet complet d'apprentissage automatique pour détecter les fausses nouvelles en utilisant une analyse de données multimodale, combinant contenu textuel, images et métadonnées pour obtenir des performances de classification robustes.

*Note: Les mots-clés techniques et noms de bibliothèques sont conservés en anglais selon les conventions du développement logiciel.*

## 📋 Table des Matières
- [Aperçu du Projet](#-aperçu-du-projet)
- [Jeux de Données](#-jeux-de-données)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture du Pipeline](#-architecture-du-pipeline)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Résultats](#-résultats)
- [Contribution](#-contribution)

## 🎯 Aperçu du Projet

Ce projet implémente une approche multimodale de pointe pour la détection de fausses nouvelles, exploitant :
- **Analyse Textuelle** : Transformers basés sur BERT pour la compréhension sémantique
- **Analyse d'Images** : CNN ResNet-18 pour l'extraction de caractéristiques visuelles
- **Fusion Multimodale** : Analyse combinée texte-image pour une précision améliorée

Le système peut traiter trois types d'entrées :
- **Multimodal** : Articles avec texte et images
- **Texte uniquement** : Articles avec contenu textuel seulement
- **Image uniquement** : Articles avec contenu visuel seulement

## 📊 Jeux de Données

### Sources de Données Principales

#### 1. Dataset FakeNewsNet
- **Source** : Kaggle - Collection FakeNewsNet
- **Format** : Fichiers CSV multiples
- **Structure** : 
  - Fichiers `*fake*.csv` → Label 0 (Fausses Nouvelles)
  - Fichiers `*real*.csv` → Label 1 (Vraies Nouvelles)
- **Colonnes** : `title`, `text`, `image_url`, `top_img`, `images`

#### 2. Dataset IFND
- **Source** : Kaggle - Image-based Fake News Detection
- **Fichier** : `IFND.csv`
- **Encodage** : Latin-1
- **Colonnes** : `Statement`, `Web`, `Image`, `Label`
- **Labels** : `TRUE`/`REAL` → 1, `Fake`/`FALSE` → 0

### Statistiques des Datasets
- **Total d'Articles** : 80 332+ articles combinés
- **Images Téléchargées** : 3 000+ images de haute qualité
- **Dataset Équilibré** : Distribution égale fausses/vraies nouvelles
- **Articles Multimodaux** : Combinaisons texte + image
- **Articles Texte uniquement** : 500 par classe (équilibré)
- **Articles Image uniquement** : 1 500 par classe (équilibré)

## ✨ Fonctionnalités

### Traitement des Données
- **Nettoyage Intelligent du Texte** : Normalisation Unicode et validation du contenu
- **Téléchargement et Validation d'Images** : Récupération automatisée d'images avec vérifications d'intégrité
- **Équilibrage Multimodal** : Échantillonnage intelligent à travers différents types de contenu
- **Filtrage de Qualité** : Longueur minimale de texte (50 caractères) et exigences de titre

### Pipeline d'Apprentissage Automatique
- **Intégration BERT** : Modèles transformer pré-entraînés pour la compréhension textuelle
- **Architecture CNN** : ResNet-18 pour l'extraction de caractéristiques d'images
- **Stratégies de Fusion** : Approches multiples pour combiner les modalités
- **Métriques d'Évaluation** : Précision, F1-score, matrices de confusion

### Stack Technique
- **Deep Learning** : PyTorch, Transformers (Hugging Face)
- **Vision par Ordinateur** : torchvision, PIL, OpenCV
- **Data Science** : pandas, numpy, scikit-learn
- **Visualisation** : matplotlib, seaborn
- **Web Scraping** : requests, hashlib pour le traitement d'images

## 🔄 Architecture du Pipeline

### 1. Collection et Prétraitement des Données
```
Fichiers CSV Bruts → Chargement des Données → Nettoyage du Texte → Validation des URLs d'Images
```

### 2. Traitement des Images
```
URLs d'Images → Téléchargement et Validation → Stockage Local → Extraction de Caractéristiques
```

### 3. Équilibrage du Dataset
```
Données Combinées → Filtrage Multimodal → Équilibrage des Classes → Dataset Final
```

### 4. Entraînement du Modèle
```
Texte: Tokenisation BERT → Embeddings → Tête de Classification
Image: ResNet-18 → Feature Maps → Tête de Classification
Fusion: Caractéristiques Combinées → Prédiction Finale
```

### 5. Évaluation
```
Division Train/Test → Entraînement du Modèle → Métriques de Performance → Visualisation
```

## 🚀 Installation

### Prérequis
- Python 3.8+
- GPU compatible CUDA (recommandé)
- 8GB+ RAM

### Dépendances
```bash
pip install torch torchvision transformers
pip install pandas numpy matplotlib seaborn
pip install scikit-learn requests pillow tqdm
pip install datasets accelerate
```

### Configuration
1. Cloner le repository
2. Créer les répertoires de données :
   ```bash
   mkdir FakeNewsNetData image_final_ancien
   ```
3. Télécharger les datasets depuis Kaggle
4. Exécuter le pipeline de prétraitement

## 💻 Utilisation

### Exécution du Pipeline de Base
```python
# Charger et prétraiter les données
df_fakenewsnet = load_fakenewsnet_data('FakeNewsNetData')
df_ifnd = load_ifnd_data('chemin/vers/IFND.csv')

# Combiner et nettoyer les datasets
df_combined = pd.concat([df_fakenewsnet, df_ifnd])
df_clean = preprocess_dataset_complet(df_combined)

# Créer un dataset multimodal équilibré
df_equilibre = create_balanced_dataset(df_clean)

# Exécuter l'analyse exploratoire
exploratory_analysis(df_equilibre)
```

### Entraînement du Modèle
```python
# Initialiser le modèle multimodal
model = MultimodalFakeNewsDetector()

# Entraîner sur le dataset équilibré
model.train(df_equilibre)

# Évaluer les performances
results = model.evaluate(test_data)
```

## 📈 Résultats

### Composition du Dataset
- **Articles Multimodaux** : Combinaisons texte + image de haute qualité
- **Classes Équilibrées** : Distribution égale fausses/vraies nouvelles
- **Métriques de Qualité** : Images validées et contenu textuel significatif

### Métriques de Performance
- **Modèle Texte uniquement** : Performance de base sur les caractéristiques textuelles
- **Modèle Image uniquement** : Capacités de reconnaissance de motifs visuels
- **Modèle Multimodal** : Précision améliorée grâce à la fusion de caractéristiques


**Note** : Ce projet est à des fins éducatives et de recherche. Vérifiez toujours les nouvelles à travers plusieurs sources fiables.
