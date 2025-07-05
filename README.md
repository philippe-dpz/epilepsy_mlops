appuyer sur edit pour voir en bon format
├── .dvc                 # DVC tracking files
├── .dvcignore           # Fichiers et dossiers à ignorer par DVC
├── .gitignore           # Fichiers et dossiers à ignorer par Git
├── README.md            # Documentation de ton projet
├── requirements.txt     # Liste des dépendances Python
├── data                 # Dossier de données
│   ├── raw              # Données brutes
│   │   └── Epileptic_Seizure_Recognition.csv
│   └── processed        # Données traitées
│       ├── X_train.npy
│       ├── Y_train.npy
│       ├── X_test.npy
│       ├── Y_test.npy
│       └── processed.dvc
├── models               # Dossier des modèles
│   └── epilepsy_model.h5
├── src                  # Code source
│   ├── app.py           # API FastAPI pour les prédictions
│   ├── patient_data_pull.py # Script pour gérer les données patients
│   ├── preprocessing.py # Préprocessing des données, 
│   └── train.py         # Entraînement du modèle
└── patients_data.csv    # Fichier CSV contenant les données patients

mode de fonctionnement: céer environment, telecharger dependencies qui sont dans requirements.txt (python 3.9.21)
lancer les programmes dans l'ordre:

- preprocessing.py: prends 800 lignes aleatoires du dataset originel pour entrainer le model 
- train.py: entraine le model avec données preprocessed et sauvegarde le model
- patient_data_pull.py: prends les données non-utilisées par train.py et créer une nouvelle base de données avec des identifiants allant de 1 à 10700
- app.py: lancer l'api uvicorn src.app:app --reload dans terminal puis entrer http://127.0.0.1:8000/predict/[id du patient] (eg:http://127.0.0.1:8000/predict/35)


