# Projet Pluridisciplinaire - Semestre 8 (M1 Sciences Cognitives)

HADDAD Naoures, MICK Léa, BACHEROT Jordy, GRUNBLATT Cyprien.

## 🧠 Objectif du projet

Ce projet a pour but de contrôler un robot simulé dans Webots à partir de gestes captés par une webcam. Les gestes sont détectés à l'aide de **MediaPipe**, puis classifiés avec un modèle d'IA entraîné **manuellement** en utilisant des méthodes telles que **RandomForest** et **Keras**. Le comportement du robot est ensuite piloté dans l'environnement **Webots** ou **Choregraphe** (utilisé pour les robots NAO).

---

## 🧰 Technologies utilisées

- [Webots](https://cyberbotics.com/) : simulateur de robots.
- [MediaPipe](https://mediapipe.dev/) : détection de poses et de gestes à partir d'une webcam.
- [Scikit-learn (RandomForest)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) : classification de gestes.
- [Keras (Tensorflow)](https://keras.io/) : modèle de deep learning pour classification de gestes.
- [Choregraphe](https://www.softbankrobotics.com/emea/en/support/nao-6/downloads/software) : interface de programmation du robot NAO.

---

## 🎮 Fonctionnement général

1. **Capture de la pose** via une webcam à l’aide de MediaPipe.
2. **Extraction des points clés** (landmarks) pour représenter les gestes.
3. **Classification des gestes** à l’aide :
   - d’un modèle RandomForest pour une approche rapide et interprétable.
   - d’un modèle Keras pour des performances plus robustes.
4. **Intégration avec Webots** pour simuler le comportement sur un robot NAO.
5. **Intégration avec Choregraphe** pour vérifier si ça fonctionne dans des conditions réels.

---

## 🤖 Exemple d'utilisation avec Webots
1. Aller dans le dossier IA comportant le script model_API.py ou model_API_onearm.py (model_API_ANN.py possible mais besoin d'un tensorflow spécifique)
2. Lancer dans le terminal la commande : fastapi dev model_API_one_arm.py (ou model_API.py)
3. Lancer le projet webots une fois l’API bien démarré

## 💃 Exemple d'utilisation avec Choregraph
1. Aller dans le dossier IA comportant le script model_API_mediapipe.py
2. Lancer dans le terminal la commande : fastapi dev model_API_mediapipe.py
3. Installer à la main les setups de requests sur python 2.7
4. Changer l'adresse IP
5. Lancer le script


## 🧪 Structure du projet

```bash
projet_pluri_2/
├── IA/             # Tous les scripts et modèles en python
├────── data/              # Ensemble des données enregistrées via mediapipe et regroupé par action
├────── data_regrouped_unprocessed/              # Données regroupées et séparées uniquement entre celles correspondantes à un changement de vitesse et à son mouvement 
├────── exploration_data/              # 
├────── model_exploration/              # 
├────── model_ANN/              # 
├────── model_one_arm/              # 
├────── model_temp/              # 
├────── preprocessing_data/              # 
├────── preprocessing_one_arm_data/              # Entraînement et sauvegarde des modèles (RandomForest / Keras)
├────── model_API.py              # Type de script à exécuter avec fastAPI
├────── ....py
├── partie_choregraph/      # Contient les scripts python 2.7 à exécuter sur choregraph
├── partie_webots/         # Comportements/flows pour NAO via Choregraphe
└── README.md            # Ce fichier
