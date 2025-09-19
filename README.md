# APT Classification System

Ce projet contient le code source, le script de déploiement, la configuration Docker et les instructions pour le système de classification CySecBERTMaxPerformance.

## Structure du dépôt

```
├── apt_classification_api.py
├── apt_classification_interface.py
├── deploy.sh
├── requirements_interface.txt
├── deployment/
│   └── docker-compose.yml
```

## Modèle entraîné (.pt)

⚠️ **Le modèle `best_cysecbert_max_performance.pt` n'est PAS inclus dans le dépôt GitHub (trop volumineux).**

Téléchargez-le sur HuggingFace :  
👉 [https://huggingface.co/melissachall/cysecbert-apt-classifier](https://huggingface.co/melissachall/cysecbert-apt-classifier)

Placez le fichier téléchargé à la racine du projet avant de lancer les scripts.

## Installation et lancement

1. Installez Python 3.8+ et git.
2. Clonez le dépôt :
   ```
   git clone https://github.com/melissachall/apt-classification.git
   cd apt-classification
   ```
3. Téléchargez le modèle `.pt` comme indiqué ci-dessus.
4. Installez les dépendances :
   ```
   pip install -r requirements_interface.txt
   ```
5. Lancez l'interface Streamlit :
   ```
   streamlit run apt_classification_interface.py
   ```
   ou lancez l'API :
   ```
   python apt_classification_api.py
   ```
6. (Optionnel) Déploiement avec Docker :
   ```
   cd deployment
   docker-compose up
   ```

## Contact

melissachall  
[https://github.com/melissachall](https://github.com/melissachall)