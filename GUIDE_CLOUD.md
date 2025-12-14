# ☁️ Guide de Déploiement : Streamlit Cloud

Ce guide vous explique comment mettre votre application en ligne gratuitement pour y accéder depuis votre téléphone.

## ⚠️ Prérequis Important : Base de Données
Votre application utilise une base de données locale (`finance_app.db`).
- **Sur Streamlit Cloud, les fichiers locaux sont réinitialisés à chaque redémarrage.**
- **Conséquence** : Votre portfolio et vos alertes seront remis à zéro si l'application redémarre (ce qui arrive environ une fois par jour ou à chaque mise à jour).
- **Solution pour plus tard** : Connecter une base de données cloud (Google Sheets ou Supabase). Pour l'instant, nous allons faire au plus simple.

## Étape 1 : Préparer les Fichiers (Déjà fait ✅)
J'ai déjà mis à jour votre fichier `requirements.txt` avec toutes les bibliothèques nécessaires. Assurez-vous que tout votre code est sauvegardé.

## Étape 2 : Mettre le Code sur GitHub
Streamlit Cloud se connecte directement à GitHub. Si vous n'avez pas de compte, créez-en un sur [github.com](https://github.com).

1.  **Créer un nouveau Repository** (Projet) sur GitHub.
    *   Nommez-le par exemple `finance-app`.
    *   Mettez-le en "Public" (plus simple) ou "Private".
2.  **Envoyer votre code** :
    *   Si vous utilisez GitHub Desktop ou la ligne de commande :
        ```bash
        git init
        git add .
        git commit -m "Initial commit"
        git branch -M main
        git remote add origin https://github.com/VOTRE_PSEUDO/finance-app.git
        git push -u origin main
        ```
    *   *Alternative simple* : Vous pouvez aussi "Uploader" les fichiers manuellement sur le site GitHub (glisser-déposer), mais c'est moins pratique pour les mises à jour.

## Étape 3 : Connecter Streamlit Cloud
1.  Allez sur [share.streamlit.io](https://share.streamlit.io/) et connectez-vous avec votre compte GitHub.
2.  Cliquez sur **"New app"**.
3.  Sélectionnez votre repository `finance-app`.
4.  **Configuration** :
    *   **Main file path** : `app_streamlit_v5.py`
5.  **Section "Advanced settings" (Secrets)** :
    *   C'est ici que vous devez mettre vos clés API (comme dans votre fichier `.env`).
    *   Copiez le contenu de votre fichier local `.env` (si vous en avez un avec des clés) et collez-le dans la zone de texte TOML, formaté comme ceci :
        ```toml
        ALPHAVANTAGE_API_KEY = "votre_clé"
        NEWSAPI_KEY = "votre_clé"
        ```
6.  Cliquez sur **"Deploy!"**.

## 📱 Accès Mobile
Une fois déployé (ça prend 2-3 minutes), vous aurez une URL du type `https://finance-app-votre-pseudo.streamlit.app`.
- Envoyez-vous ce lien par mail ou WhatsApp.
- Ouvrez-le sur votre téléphone.
- **Astuce Pro** : Sur iPhone (Safari) ou Android (Chrome), faites "Ajouter à l'écran d'accueil". L'app apparaîtra comme une vraie application native !

## 🔄 Mises à jour
Pour mettre à jour l'application, il suffit de modifier le code sur votre ordinateur et de faire un nouveau `push` sur GitHub. Streamlit Cloud détectera le changement et mettra à jour le site automatiquement.
