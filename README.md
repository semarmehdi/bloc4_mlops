# 🪐 Demo MLOps — IBM Attrition

Pipeline MLOps complet de bout en bout : entraînement d'un modèle de prédiction
d'attrition (départ d'un employé), suivi des expériences avec **MLflow**, exposition
du modèle via une **API FastAPI**, et **pipeline ETL** qui collecte des données,
les enrichit avec les prédictions, et les stocke sur **S3** + **PostgreSQL (Neon)**.
Le tout testé et industrialisé via **GitHub Actions** et déployé sur **Hugging Face Spaces**.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │   MLflow Tracking Server      │
                    │   (HF Space - Docker)         │
                    │   Backend  : Neon Postgres    │
                    │   Artifacts: S3               │
                    └───────────┬──────────────────┘
                                │ log / register
                                │
   (1) ENTRAÎNEMENT LOCAL ──────┘
        train/train.py
                                │ load model @production
                                ▼
                    ┌──────────────────────────────┐
                    │   API Modèle (FastAPI)        │
                    │   (HF Space - Docker)         │
                    │   GET /  GET /preview         │
                    │   POST /predict               │
                    └───────────┬──────────────────┘
                                ▲ POST /predict
                                │
   (2) API DONNÉES             │
   (HF Space)                  │
   GET /current-employee ──────┤
                                │
   (3) PIPELINE ETL  ──────────┘
        etl.py + utils/
        ├── extract  → API données  + backup JSON sur S3
        ├── transform→ appelle l'API modèle ligne par ligne
        └── load     → CSV sur S3 + INSERT dans Neon Postgres
```

Trois services sont déployés sur Hugging Face Spaces (le serveur MLflow, l'API
modèle, l'API données) et un pipeline batch (l'ETL) tourne en local, dans Docker,
ou dans la CI.

---

## 🗂️ Structure du projet

```
demo-MLops/
├── .github/workflows/
│   └── ci.yaml                 # CI : tests + build Docker + run ETL
├── ibmattritionapi/            # (sous-repo HF) API qui sert les données employés
├── mlflow/                     # Serveur MLflow à déployer sur HF
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env                    # (local, ignoré par git)
├── mlflow_HF/                  # (clone du repo HF du Space MLflow)
├── model_api/                  # API du modèle à déployer sur HF
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env                    # (local, ignoré par git)
├── model_api_HF/               # (clone du repo HF du Space API modèle)
├── train/
│   ├── train.py                # entraînement local -> log vers MLflow
│   ├── requirements.txt
│   ├── temp/                   # artefacts locaux (ignoré par git)
│   └── .env
├── utils/                      # package ETL
│   ├── __init__.py
│   ├── extract.py
│   ├── transform.py
│   └── load.py
├── tests/
│   ├── test_smoke_apis.py      # tests réels des 2 APIs (skip si env absent)
│   └── test_transform.py       # tests logique pure (zéro mock)
├── conftest.py
├── etl.py                      # point d'entrée du pipeline ETL
├── Dockerfile                  # image qui exécute l'ETL une fois
├── requirements.txt
├── requirements-tests.txt
├── .gitignore
└── .env                        # (local, ignoré par git)
```

---

## ✅ Prérequis

- Un compte **Hugging Face** (pour les 3 Spaces Docker)
- Un compte **AWS** avec un **bucket S3** et une paire de clés IAM
- Une base **PostgreSQL** gratuite sur **[Neon](https://neon.tech)**
- **Python 3.11** en local (idéalement via conda)
- **Docker** installé en local

---

## 🔐 Variables d'environnement (référence)

> ⚠️ **Aucun `.env` ni secret ne doit être commité.** Ils sont tous ignorés par
> `.gitignore`. En CI, ils viennent des _GitHub Secrets/Variables_ ; sur HF, des
> _Settings → Variables and secrets_ du Space.

**Racine `.env` (ETL)**

| Variable                                   | Exemple                                             | Rôle                      |
| ------------------------------------------ | --------------------------------------------------- | ------------------------- |
| `IBM_ATTRITION_BASE_URL`                   | `https://semarmehdi-ibmattritionapi.hf.space`       | URL de l'API données      |
| `IBM_ATTRITION_ENDPOINT`                   | `/current-employee`                                 | Endpoint données          |
| `IBM_ATTRITION_BATCH_SIZE`                 | `20`                                                | Nb de lignes collectées   |
| `IBM_ATTRITION_SLEEP_SECONDS`              | `1.0`                                               | Pause entre 2 pulls       |
| `IBM_ATTRITION_MODEL_API_BASE_URL`         | `https://semarmehdi-model-api.hf.space`             | URL de l'API modèle       |
| `IBM_ATTRITION_MODEL_API_PREDICT_ENDPOINT` | `/predict`                                          | Endpoint de prédiction    |
| `IBM_ATTRITION_MODEL_API_TIMEOUT`          | `120`                                               | Timeout des appels modèle |
| `S3BucketName`                             | `mon-bucket`                                        | Bucket S3                 |
| `IBM_ATTRITION_S3_PREFIX`                  | `raw/ibm`                                           | Préfixe des backups bruts |
| `IBM_ATTRITION_S3_PRED_PREFIX`             | `clean/ibm`                                         | Préfixe des prédictions   |
| `AWS_ACCESS_KEY_ID`                        | `AKIA...`                                           | Clé IAM                   |
| `AWS_SECRET_ACCESS_KEY`                    | `...`                                               | Secret IAM                |
| `AWS_DEFAULT_REGION`                       | `eu-west-3`                                         | Région S3                 |
| `DATABASE_URL`                             | `postgresql://user:pwd@host/neondb?sslmode=require` | Connexion Neon            |
| `DB_TARGET_TABLE`                          | `ibm_attrition_predictions`                         | Table cible               |

> ⚠️ **Important : format `.env` pour Docker.** Avec `docker run --env-file .env`,
> n'entoure **jamais** les valeurs de guillemets et ne mets pas d'espace autour du
> `=`. Une URL entre guillemets devient `"https://..."` (guillemets inclus) et
> `requests` plante avec _No connection adapters were found_.

**`mlflow/.env` (serveur de tracking)**

| Variable                                                             | Rôle                                      |
| -------------------------------------------------------------------- | ----------------------------------------- |
| `BACKEND_STORE_URI`                                                  | URL Neon (stocke runs, params, métriques) |
| `ARTIFACT_ROOT`                                                      | `s3://mon-bucket/mlflow-artifacts`        |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | Accès S3                                  |

**`model_api/.env` et `train/.env`**

| Variable                                                             | Rôle                                               |
| -------------------------------------------------------------------- | -------------------------------------------------- |
| `MLFLOW_TRACKING_URI`                                                | URL du Space MLflow                                |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | Pour télécharger les artefacts du modèle depuis S3 |

---

## 🧱 Étape 0 — Préparer le stockage

### a) Base Neon (Postgres)

1. Crée deux projets sur [neon.tech](https://neon.tech).
2. Récupère les **connection string** (format `postgresql://...sslmode=require`).
3. Une servira au **backend MLflow** (`BACKEND_STORE_URI`) et l'autre à l'**ETL** (`DATABASE_URL`).

### b) Bucket S3

1. Crée un bucket S3 (ex. `mon-bucket`) dans une région (ex. `eu-west-3`).
2. Crée un utilisateur IAM avec une policy donnant l'accès à ce bucket
   (`s3:PutObject`, `s3:GetObject`, `s3:ListBucket`).
3. Note la paire `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.

---

## 🧪 Étape 1 — Déployer le serveur MLflow sur Hugging Face

Le serveur MLflow centralise tous tes entraînements. Il utilise Neon comme
**backend store** (métadonnées) et S3 comme **artifact store** (modèles, fichiers).

### 1.1 Créer le Space

1. Sur HF : **New Space** → SDK **Docker** → visibilité au choix.
2. Clone-le en local (c'est ton dossier `mlflow_HF/`).

### 1.2 Le `mlflow/Dockerfile`

Exemple type (adapte à ce que tu as déjà) :

```dockerfile
FROM python:3.11

WORKDIR /home/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends nano unzip curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN curl -fsSL https://get.deta.dev/cli.sh | sh

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install



# THIS IS SPECIFIC TO HUGGINFACE
# We create a new user named "user" with ID of 1000
RUN useradd -m -u 1000 user
# We switch from "root" (default user when creating an image) to "user"
USER user
# We set two environmnet variables
# so that we can give ownership to all files in there afterwards
# we also add /home/user/.local/bin in the $PATH environment variable
# PATH environment variable sets paths to look for installed binaries
# We update it so that Linux knows where to look for binaries if we were to install them with "user".
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# We set working directory to $HOME/app (<=> /home/user/app)
WORKDIR $HOME/app

# Copy all local files to /home/user/app with "user" as owner of these files
# Always use --chown=user when using HUGGINGFACE to avoid permission errors
COPY --chown=user . $HOME/app


COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV BACKEND_STORE_URI=$BACKEND_STORE_URI
ENV ARTIFACT_ROOT=$ARTIFACT_ROOT

CMD mlflow server -p $PORT \
    --host 0.0.0.0 \
    --backend-store-uri $BACKEND_STORE_URI \
    --default-artifact-root $ARTIFACT_ROOT \
    --allowed-hosts "*"
```

### 1.3 Renseigner les secrets du Space

Dans **Settings → Variables and secrets** du Space, ajoute (en _Secrets_) :
`BACKEND_STORE_URI`, `ARTIFACT_ROOT`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`.

### 1.4 Pousser et vérifier

```bash
cd mlflow_HF
# copie ton Dockerfile + requirements, commit, push
git add .
git commit -m "mlflow server"
git push
```

Une fois le Space _Running_, ouvre son URL : tu dois voir l'**interface MLflow**, tu as trois petits points en haut à droite. Clique sur "embbed this space".
Cette URL est ton `MLFLOW_TRACKING_URI`.

---

## 🏋️ Étape 2 — Entraîner le modèle en local

L'entraînement se fait **sur ta machine**, mais tout est **loggé vers le serveur
MLflow distant** (HF). Le modèle entraîné est stocké sur S3 via MLflow.

### 2.1 Préparer l'environnement

```bash
conda create -n demo-mlflow-train python=3.11 -y
conda activate demo-mlflow-train
pip install -r train/requirements.txt
```

### 2.2 Configurer le `.env` de l'entraînement

`train/.env` :

```dotenv
MLFLOW_TRACKING_URI=https://<ton-space-mlflow>.hf.space
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
```

### 2.3 Lancer l'entraînement

```bash
cd train
python train.py
```

**Ce qui se passe, étape par étape :**

1. **Chargement des variables** — `load_dotenv()` lit `train/.env`, ce qui pointe
   MLflow vers le serveur HF (`MLFLOW_TRACKING_URI`).
2. **Création / sélection d'une expérience** — `mlflow.set_experiment("ibm_attrition")`.
3. **Chargement des données** — le dataset IBM HR Attrition (Excel sur S3 public).
4. **Préparation** — split train/test, pipeline de preprocessing (encodage des
   variables catégorielles + scaling), puis le modèle (ex. régression logistique
   ou random forest).
5. **Démarrage d'un run** — `with mlflow.start_run():` ; on active
   `mlflow.sklearn.autolog()` (ou on logge manuellement params + métriques).
6. **Entraînement** — `model.fit(X_train, y_train)`.
7. **Évaluation** — calcul accuracy / f1 / recall sur le test, loggés dans MLflow.
8. **Log du modèle** — `mlflow.sklearn.log_model(model, name="model", registered_model_name="ibm_attrition_detector")`.
   → Le modèle est **uploadé sur S3** et **enregistré** dans le Model Registry.

### 2.4 Vérifier dans l'UI MLflow

Ouvre le Space MLflow → onglet **Experiments** : ton run apparaît avec ses
métriques. Onglet **Models** : la version du modèle `ibm_attrition_detector`
est listée.

---

## 🚀 Étape 3 — Promouvoir le modèle en "production"

L'API modèle charge le modèle via l'alias `@production`
(`models:/ibm_attrition_detector@production`). Il faut donc **attacher cet alias**
à la version que tu veux servir.

**Via l'UI MLflow :** Models → `ibm_attrition_detector` → la version voulue →
_Aliases_ → ajoute `production`.

**Ou via code :**

```python
from mlflow import MlflowClient
client = MlflowClient()  # lit MLFLOW_TRACKING_URI
client.set_registered_model_alias(
    name="ibm_attrition_detector",
    alias="production",
    version=1,            # la version à promouvoir
)
```

> 💡 Quand tu réentraînes plus tard, tu crées une **nouvelle version**. Il suffit
> de déplacer l'alias `production` dessus pour que l'API serve le nouveau modèle
> au prochain redémarrage — sans changer une ligne de code.

---

## 🤖 Étape 4 — Déployer l'API du modèle sur Hugging Face

L'API (`model_api/app.py`, FastAPI) charge le modèle `@production` au démarrage et
expose `/predict`. Elle renvoie `prediction`, `proba_0` et `proba_1`.

### 4.1 Créer le Space

New Space → SDK **Docker** → clone-le (ton dossier `model_api_HF/`).

### 4.2 Le `model_api/Dockerfile`

```dockerfile
FROM python:3.11-slim

RUN apt-get update -y
RUN apt-get install nano unzip curl -y

# # THIS IS SPECIFIC TO HUGGINFACE
# # We create a new user named "user" with ID of 1000
# RUN useradd -m -u 1000 user
# # We switch from "root" (default user when creating an image) to "user"
# USER user
# # We set two environmnet variables
# # so that we can give ownership to all files in there afterwards
# # we also add /home/user/.local/bin in the $PATH environment variable
# # PATH environment variable sets paths to look for installed binaries
# # We update it so that Linux knows where to look for binaries if we were to install them with "user".
# ENV HOME=/home/user \
#     PATH=/home/user/.local/bin:$PATH

# We set working directory to $HOME/app (<=> /home/user/app)
WORKDIR /home/app

# Leverage layer caching: copy only reqs first
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# # Copy all local files to /home/user/app with "user" as owner of these files
# # Always use --chown=user when using HUGGINGFACE to avoid permission errors
# COPY --chown=user . $HOME/app

COPY app.py /home/app/app.py

CMD ["bash","-lc","fastapi run app.py --host 0.0.0.0 --port ${PORT}"]
```

### 4.3 Secrets du Space

`MLFLOW_TRACKING_URI`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, (pour télécharger le modèle depuis S3).

### 4.4 Pousser et tester

```bash
cd model_api_HF
git add .
git commit -m "model api"
git push
```

Une fois _Running_, teste la doc interactive : `https://<space>.hf.space/docs`,
ou en ligne de commande :

```bash
curl -X POST https://<space>.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"Age":36,"BusinessTravel":"Travel_Rarely","DailyRate":852,"Department":"Research & Development","DistanceFromHome":5,"Education":4,"EducationField":"Life Sciences","EmployeeCount":1,"EmployeeNumber":51,"EnvironmentSatisfaction":2,"Gender":"Female","HourlyRate":82,"JobInvolvement":2,"JobLevel":1,"JobRole":"Research Scientist","JobSatisfaction":1,"MaritalStatus":"Married","MonthlyIncome":3419,"MonthlyRate":13072,"NumCompaniesWorked":9,"Over18":"Y","OverTime":"Yes","PercentSalaryHike":14,"PerformanceRating":3,"RelationshipSatisfaction":4,"StandardHours":80,"StockOptionLevel":1,"TotalWorkingYears":6,"TrainingTimesLastYear":3,"WorkLifeBalance":4,"YearsAtCompany":1,"YearsInCurrentRole":1,"YearsSinceLastPromotion":0,"YearsWithCurrManager":0}'
```

Réponse attendue :

```json
{ "prediction": 0, "proba_0": 0.87, "proba_1": 0.13 }
```

---

## 📡 Étape 5 — L'API de données

L'api `ibmattritionapi/` contient l'API qui sert des lignes d'employés (un
Space HF séparé). Elle expose un endpoint (ex. `/current-employee`) qui renvoie un
employé au format `{"columns": [...], "data": [[...]]}`. C'est la **source** de
l'ETL. Déploie-la de la même façon (Space Docker), puis renseigne son URL dans
`IBM_ATTRITION_BASE_URL` / `IBM_ATTRITION_ENDPOINT`.
Voici l'url de base : `https://semarmehdi-ibmattritionapi.hf.space`
Attention vous n'avez pas besoin de déployer cette api.
Contactez-moi par message si elle ne tourne plus.

---

## 🔄 Étape 6 — Le pipeline ETL

`etl.py` orchestre les 3 étapes du package `utils/` :

1. **extract** (`extract.py`) — interroge l'API données `BATCH_SIZE` fois,
   sauvegarde un backup JSON brut sur S3, et renvoie l'artefact en mémoire.
2. **transform** (`transform.py`) — reconstruit un DataFrame, appelle l'API modèle
   ligne par ligne, et ajoute `prediction` / `proba_0` / `proba_1`.
3. **load** (`load.py`) — exporte le DataFrame en CSV sur S3, crée la table Neon
   si besoin, et insère les lignes (mode `append`).

### 6.1 Tester en local (sans Docker)

```bash
conda activate demo-mlflow-train
pip install -r requirements.txt
python etl.py        # lit le .env de la racine via load_dotenv()
```

### 6.2 Tester en local avec Docker

```bash
docker build -t ibm-attrition-etl .
docker run --rm --env-file .env ibm-attrition-etl
```

> Rappel : `.env` **sans guillemets** pour `--env-file`.

### 6.3 Note Postgres importante

La table est créée avec des colonnes **entre guillemets** (`"Age"`, etc.) pour
respecter la casse exacte de pandas. Si une ancienne table existe avec des colonnes
en minuscules, supprime-la d'abord :

```sql
DROP TABLE IF EXISTS public.ibm_attrition_predictions;
```

**Ou directement dans l'interface Neon à la main !**

## 🧫 Étape 7 — Les tests

Deux niveaux :

- **`tests/test_transform.py`** — tests de **logique pure**, zéro mock, zéro réseau.
  Idéal pour comprendre pytest (motif _Arrange / Act / Assert_).
- **`tests/test_smoke_apis.py`** — _smoke tests_ qui appellent **réellement** les
  deux APIs. Ils se **skippent** si les variables d'env ne sont pas définies.

```bash
pip install -r requirements-tests.txt
pytest -v
```

---

## ⚙️ Étape 8 — CI/CD avec GitHub Actions

Le workflow `.github/workflows/ci.yaml` se déclenche à chaque push/PR sur `main` :
checkout → install → `pytest` → build Docker → run de l'ETL → listing de debug.

### 8.1 Où mettre les secrets et variables

**Repo GitHub → Settings → Secrets and variables → Actions**, deux onglets :
**Secrets** et **Variables**.

> ℹ️ **Repository secrets, pas Environment secrets.** Sur cette page, GitHub
> affiche deux blocs : _Environment secrets_ (fonctionnalité optionnelle des
> "Environments", à ignorer ici) et _Repository secrets_. Utilise le bouton vert
> **"New repository secret"** — aucun environnement à créer. Idem côté variables :
> onglet _Variables_ → _New repository variable_. C'est ce niveau "repository" que
> lisent `${{ secrets.NOM }}` et `${{ vars.NOM }}` sans configuration
> supplémentaire. (Un _Environment_ ne sert que si tu veux ajouter une approbation
> manuelle ou des règles de protection avant un déploiement — hors périmètre ici.)

- **Secrets** (chiffrés, masqués dans les logs) : `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `DATABASE_URL`,
  `IBM_ATTRITION_BASE_URL`, `IBM_ATTRITION_ENDPOINT`,
  `IBM_ATTRITION_MODEL_API_BASE_URL`, `S3BUCKETNAME`, `IBM_ATTRITION_S3_PREFIX`,
  `IBM_ATTRITION_S3_PRED_PREFIX`.
  → appelés via `${{ secrets.NOM }}`.
- **Variables** (en clair) : `IBM_ATTRITION_BATCH_SIZE`,
  `IBM_ATTRITION_SLEEP_SECONDS`, `IBM_ATTRITION_MODEL_API_PREDICT_ENDPOINT`,
  `IBM_ATTRITION_MODEL_API_TIMEOUT`, `DB_TARGET_TABLE`.
  → appelés via `${{ vars.NOM }}`.

Règle : si une fuite est un problème → **Secret**. Sinon → **Variable**.

> ⚠️ Avec cette CI, **chaque push exécute réellement l'ETL** (écriture S3 +
> insertion Neon + appels aux APIs). Pour un usage "prod", garder le `docker run`
> en _smoke test_ et déplacer le vrai run dans un workflow `workflow_dispatch` /
> `schedule` séparé.

---

## 🔒 Sécurité — à ne pas oublier

- **Aucun secret dans le code ni dans les Dockerfiles.** Tout passe par les `.env`
  (local), les _GitHub Secrets_ (CI) ou les _secrets HF_ (Spaces).
- **`load.py`** : retirer le mot de passe Neon en dur dans le `default=` de
  `DATABASE_URL`. Si ce mot de passe a déjà été poussé un jour, le **faire tourner**
  côté Neon (le considérer comme compromis).
- Avant le premier `git push`, vérifier qu'aucun secret n'est suivi :
  ```bash
  git ls-files | grep -E "\.env$|__pycache__"
  ```
  (ne doit rien renvoyer).

---

## ⚡ Démarrage rapide (TL;DR)

```bash
# 1. Stockage : créer Neon + bucket S3
# 2. Déployer le serveur MLflow (Space Docker)
# 3. Entraîner et promouvoir le modèle
conda create -n demo-mlflow-train python=3.11 -y && conda activate demo-mlflow-train
pip install -r train/requirements.txt
python train/train.py            # log vers MLflow + register
# (promouvoir la version en alias 'production' via l'UI MLflow)

# 4. Déployer l'API modèle + l'API données (Spaces Docker)

# 5. Lancer l'ETL
pip install -r requirements.txt
python etl.py                    # ou : docker run --rm --env-file .env ibm-attrition-etl

# 6. Tests
pip install -r requirements-tests.txt
pytest -v
```
