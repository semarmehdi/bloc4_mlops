# Demo MLOps — Prédiction d'attrition (IBM HR)

**Livrable de certification**
Titre AIA — Architecte en Intelligence Artificielle
Bloc 4 — MLOps (industrialisation, déploiement, monitoring et réentraînement)
Auteur : **Ahmed Mehdi SEMAR**

---

## 1. Contexte et objectif

Ce projet implémente une chaîne MLOps de bout en bout autour d'un cas métier RH :
prédire le risque de départ d'un employé (*attrition*) à partir du jeu de données
public IBM HR Attrition.

L'objectif n'est pas seulement d'entraîner un modèle, mais de couvrir l'ensemble du
cycle de vie : entraînement et suivi d'expériences (MLflow), exposition du modèle via
une API REST (FastAPI), pipeline de données batch (ETL), industrialisation par
intégration continue (GitHub Actions), monitoring de dérive (Evidently) et
réentraînement automatisé.

Le projet est réparti sur **deux dépôts complémentaires** :

| Dépôt | Rôle |
| --- | --- |
| **bloc4_mlops** (ce dépôt) | Infrastructure de service : pipeline ETL, API modèle, serveur MLflow, CI/CD, tests |
| [**train-repo**](https://github.com/semarmehdi/train-repo) | (Ré)entraînement automatisé et monitoring de dérive (Evidently) |

---

## 2. Architecture

![Architecture MLOps](docs/diagrams/architecture.svg)

> Source éditable : [`docs/diagrams/architecture.drawio`](docs/diagrams/architecture.drawio) (ouvrir avec [draw.io](https://app.diagrams.net)).

Quatre composants sont déployés en tant que Spaces Docker sur Hugging Face (serveur
MLflow, API modèle, API données) ; le pipeline ETL s'exécute en local, dans Docker
ou dans la CI. La boucle de monitoring et de réentraînement est portée par le dépôt
`train-repo`.

---

## 3. Pile technique

- **Modèle** : scikit-learn (RandomForestClassifier, pipeline de preprocessing)
- **Suivi & registry** : MLflow (backend Neon PostgreSQL, artifact store S3)
- **Service modèle** : FastAPI, conteneurisé (Docker), déployé sur Hugging Face Spaces
- **Données** : Amazon S3 (backups bruts et prédictions), PostgreSQL Neon (table cible)
- **Orchestration batch** : script ETL Python conteneurisé
- **CI/CD** : GitHub Actions
- **Monitoring** : Evidently (dérive des données et des prédictions)

---

## 4. Structure du dépôt

```
bloc4_mlops/
├── .github/workflows/
│   └── ci.yaml                 # CI : tests + build Docker + exécution ETL
├── mlflow/                     # Serveur MLflow (image à déployer sur HF)
│   ├── Dockerfile
│   └── requirements.txt
├── model_api/                  # API du modèle (à déployer sur HF)
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── train/                      # Entraînement initial de référence (bootstrap)
│   ├── train.py
│   └── requirements.txt
├── utils/                      # Package ETL
│   ├── __init__.py
│   ├── extract.py
│   ├── transform.py
│   └── load.py
├── tests/
│   ├── test_transform.py       # Tests de logique pure (sans réseau ni mock)
│   └── test_smoke_apis.py      # Smoke tests des APIs (skip si env absent)
├── conftest.py
├── etl.py                      # Point d'entrée du pipeline ETL
├── Dockerfile                  # Image d'exécution de l'ETL
├── requirements.txt
├── requirements-tests.txt
└── .env.example
```

> L'entraînement de référence sous `train/` sert à amorcer le premier modèle. Le
> **(ré)entraînement automatisé** (split stratifié, gestion du déséquilibre, tuning,
> métriques) et le **monitoring** sont industrialisés dans le dépôt `train-repo`.

---

## 5. Prérequis

- Un compte **Hugging Face** (pour les Spaces Docker)
- Un compte **AWS** avec un bucket **S3** et une paire de clés IAM
- Une base **PostgreSQL** (offre gratuite **Neon**)
- **Python 3.11** et **Docker** en local

---

## 6. Variables d'environnement (référence)

Aucun fichier `.env` ni secret n'est versionné (tous ignorés par `.gitignore`).
En CI, les valeurs proviennent des *GitHub Secrets/Variables* ; sur Hugging Face,
des *Settings → Variables and secrets* du Space.

**Racine `.env` (ETL)**

| Variable | Exemple | Rôle |
| --- | --- | --- |
| `IBM_ATTRITION_BASE_URL` | `https://semarmehdi-ibmattritionapi.hf.space` | URL de l'API données |
| `IBM_ATTRITION_ENDPOINT` | `/current-employee` | Endpoint données |
| `IBM_ATTRITION_BATCH_SIZE` | `20` | Nombre de lignes collectées |
| `IBM_ATTRITION_SLEEP_SECONDS` | `1.0` | Pause entre deux appels |
| `IBM_ATTRITION_MODEL_API_BASE_URL` | `https://semarmehdi-model-api.hf.space` | URL de l'API modèle |
| `IBM_ATTRITION_MODEL_API_PREDICT_ENDPOINT` | `/predict` | Endpoint de prédiction |
| `IBM_ATTRITION_MODEL_API_TIMEOUT` | `120` | Timeout des appels modèle |
| `S3BucketName` | `mon-bucket` | Bucket S3 |
| `IBM_ATTRITION_S3_PREFIX` | `raw/ibm` | Préfixe des backups bruts |
| `IBM_ATTRITION_S3_PRED_PREFIX` | `clean/ibm` | Préfixe des prédictions |
| `AWS_ACCESS_KEY_ID` | `AKIA...` | Clé IAM |
| `AWS_SECRET_ACCESS_KEY` | `...` | Secret IAM |
| `AWS_DEFAULT_REGION` | `eu-west-3` | Région S3 |
| `DATABASE_URL` | `postgresql://user:pwd@host/neondb?sslmode=require` | Connexion Neon |
| `DB_TARGET_TABLE` | `ibm_attrition_predictions` | Table cible |

Variables optionnelles de l'ordonnanceur ETL (voir section 9.6) :
`ETL_LOOP_ITERATIONS`, `ETL_LOOP_INTERVAL_SECONDS`, `ETL_FAIL_FAST`.

> **Format `.env` pour Docker.** Avec `docker run --env-file .env`, ne jamais entourer
> les valeurs de guillemets ni mettre d'espace autour du `=`. Une URL entre guillemets
> casse `requests` (*No connection adapters were found*).

**`mlflow/.env` (serveur de tracking)**

| Variable | Rôle |
| --- | --- |
| `BACKEND_STORE_URI` | URL Neon (runs, paramètres, métriques) |
| `ARTIFACT_ROOT` | `s3://mon-bucket/mlflow-artifacts` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | Accès S3 |

**`model_api/.env` et `train/.env`**

| Variable | Rôle |
| --- | --- |
| `MLFLOW_TRACKING_URI` | URL du Space MLflow |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | Téléchargement des artefacts modèle depuis S3 |

---

## 7. Mise en route

### 7.0 Préparer le stockage

1. **Neon** : créer deux bases — l'une pour le backend MLflow (`BACKEND_STORE_URI`),
   l'autre pour l'ETL (`DATABASE_URL`). Récupérer les chaînes de connexion
   (`postgresql://...sslmode=require`).
2. **S3** : créer un bucket dans une région, puis un utilisateur IAM avec une policy
   `s3:PutObject` / `s3:GetObject` / `s3:ListBucket` ; noter la paire de clés.

### 7.1 Déployer le serveur MLflow (Hugging Face, SDK Docker)

Le serveur centralise les entraînements ; backend Neon (métadonnées) et S3
(artefacts). Renseigner dans les *Secrets* du Space : `BACKEND_STORE_URI`,
`ARTIFACT_ROOT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`.
Une fois le Space *Running*, son URL est le `MLFLOW_TRACKING_URI`.

### 7.2 Entraîner le modèle

Pour l'amorçage, l'entraînement de référence est sous `train/`. Pour le pipeline
d'entraînement complet (split stratifié, `class_weight="balanced"`, `GridSearchCV`,
métriques precision/recall/F1/ROC-AUC et matrice de confusion loggées dans MLflow),
voir le dépôt **train-repo**.

```bash
conda create -n demo-mlflow-train python=3.11 -y && conda activate demo-mlflow-train
pip install -r train/requirements.txt
python train/train.py
```

Le modèle est enregistré dans le Model Registry sous le nom `ibm_attrition_detector`.

### 7.3 Promouvoir le modèle en production

L'API modèle charge `models:/ibm_attrition_detector@production`. Il faut attacher
l'alias `production` à la version voulue, via l'UI MLflow ou par code :

```python
from mlflow import MlflowClient
MlflowClient().set_registered_model_alias(
    name="ibm_attrition_detector", alias="production", version=1
)
```

Au réentraînement, une nouvelle version est créée ; déplacer l'alias `production`
suffit à servir le nouveau modèle au prochain redémarrage de l'API, sans modifier
le code (mécanisme de **rollback** : on repointe l'alias sur une version antérieure).

### 7.4 Déployer l'API du modèle (Hugging Face, SDK Docker)

`model_api/app.py` (FastAPI) charge le modèle `@production` au démarrage et expose
`POST /predict`, qui renvoie `prediction`, `proba_0` et `proba_1`. Secrets du Space :
`MLFLOW_TRACKING_URI`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.

Exemple d'appel :

```bash
curl -X POST https://<space>.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"Age":36,"BusinessTravel":"Travel_Rarely", ...}'
# -> { "prediction": 0, "proba_0": 0.87, "proba_1": 0.13 }
```

La documentation interactive (Swagger) est disponible sur `https://<space>.hf.space/docs`.

### 7.5 L'API de données

Le Space `ibmattritionapi` sert des lignes d'employés via `GET /current-employee`,
au format `{"columns": [...], "data": [[...]]}`. C'est la source de l'ETL.
URL de base : `https://semarmehdi-ibmattritionapi.hf.space` (déjà déployée).

### 7.6 Le pipeline ETL

![Pipeline ETL](docs/diagrams/etl_flow.svg)

> Source éditable : [`docs/diagrams/etl_flow.drawio`](docs/diagrams/etl_flow.drawio).

`etl.py` orchestre le package `utils/` :

1. **extract** — interroge l'API données `BATCH_SIZE` fois, sauvegarde un backup
   JSON brut sur S3, renvoie l'artefact en mémoire.
2. **transform** — reconstruit un DataFrame, appelle l'API modèle ligne par ligne,
   ajoute `prediction` / `proba_0` / `proba_1`.
3. **load** — exporte un CSV sur S3 et insère les lignes dans Neon (mode append).

Exécution en local ou dans Docker :

```bash
pip install -r requirements.txt
python etl.py
# ou
docker build -t ibm-attrition-etl .
docker run --rm --env-file .env ibm-attrition-etl
```

L'ETL embarque un **ordonnanceur applicatif borné** qui simule un cadencement de
production sans dépendance externe. Il est piloté par trois variables :

| Variable | Prod (défaut image) | CI |
| --- | --- | --- |
| `ETL_LOOP_ITERATIONS` | `0` (boucle infinie) | `3` (le job se termine seul) |
| `ETL_LOOP_INTERVAL_SECONDS` | `30` | `30` |
| `ETL_FAIL_FAST` | `false` (résilient) | `true` (échoue au premier incident) |

L'arrêt sur signal (`SIGTERM` / `SIGINT`) est géré proprement : le cycle en cours se
termine avant la sortie du conteneur.

### 7.7 Les tests

```bash
pip install -r requirements-tests.txt
pytest -v
```

- `tests/test_transform.py` : tests de logique pure (motif Arrange / Act / Assert),
  sans réseau ni mock.
- `tests/test_smoke_apis.py` : smoke tests appelant réellement les deux APIs, qui se
  *skippent* automatiquement si les variables d'environnement sont absentes.

### 7.8 CI/CD avec GitHub Actions

Le workflow `.github/workflows/ci.yaml` se déclenche sur push et pull request vers
`main` : checkout → installation → `pytest` → build Docker → exécution bornée de l'ETL.

Les secrets et variables se définissent dans **Settings → Secrets and variables →
Actions**, au niveau *Repository* (et non *Environment*). Règle de répartition : une
valeur dont la fuite est un risque va en *Secret* (`AWS_*`, `DATABASE_URL`, URLs
d'API) ; les paramètres non sensibles vont en *Variable* (`*_BATCH_SIZE`,
`*_TIMEOUT`, `DB_TARGET_TABLE`).

> **Recommandation prod.** Cette CI exécute réellement l'ETL à chaque push (écriture
> S3, insertion Neon, appels API). Pour un usage industriel, conserver le `docker run`
> en *smoke test* borné et déplacer le run réel dans un workflow `schedule` /
> `workflow_dispatch` dédié.

---

## 8. Monitoring et réentraînement

Le monitoring et le réentraînement sont automatisés dans le dépôt **train-repo** et
ferment la boucle MLOps :

- **Détection de dérive (Evidently).** Un workflow planifié (`monitoring.yaml`)
  agrège, sur une fenêtre glissante, les prédictions de production stockées sur S3,
  les compare à la distribution de référence (jeu d'entraînement) et produit un
  rapport HTML de *data drift* archivé en artefact et sur S3.
- **Réentraînement conditionnel.** Si la part de colonnes dérivées dépasse un seuil,
  le workflow déclenche automatiquement le pipeline d'entraînement
  (`train.yaml`, via `gh workflow run`), qui produit une nouvelle version du modèle
  enregistrée sous l'alias `challenger`.
- **Périmètre de mesure.** En l'absence de labels réels en production, le monitoring
  porte sur la **dérive des entrées** et la **dérive des prédictions**, et non sur
  l'accuracy en ligne, qui nécessiterait la collecte a posteriori des départs réels.

---

## 9. Versioning et rollback

Le Model Registry MLflow assure le versioning des modèles. Les **alias** matérialisent
les états de service :

- `challenger` : dernier modèle entraîné, non encore promu.
- `production` : modèle effectivement servi par l'API.

La promotion comme le **rollback** consistent à déplacer l'alias `production` vers la
version cible — sans aucune modification de code, l'API rechargeant l'alias au
redémarrage.

---

## 10. Sécurité

- Aucun secret n'est présent dans le code ni dans les Dockerfiles ; tout passe par
  les `.env` (local), les *GitHub Secrets* (CI) ou les *secrets HF* (Spaces).
- `load.py` ne doit contenir **aucun mot de passe en dur** dans la valeur par défaut
  de `DATABASE_URL`. Tout identifiant ayant pu être poussé doit être considéré comme
  compromis et **renouvelé** côté Neon.
- Avant tout `git push`, vérifier qu'aucun secret n'est suivi :
  `git ls-files | grep -E "\.env$|__pycache__"` (ne doit rien retourner).

---

## 11. Couverture des attendus du Bloc 4

| Attendu | Réalisation |
| --- | --- |
| Préparation des données et entraînement | Pipeline scikit-learn, split stratifié, gestion du déséquilibre, tuning, métriques (cf. train-repo) |
| Déploiement (REST API, conteneurisé) | API FastAPI conteneurisée, déployée sur Hugging Face Spaces |
| CI/CD | GitHub Actions : tests, build Docker, exécution ETL, pipeline de réentraînement |
| Monitoring et alertes | Détection de dérive Evidently planifiée, rapports archivés |
| Réentraînement automatisé | Déclenchement sur dérive (`monitoring.yaml` → `train.yaml`) |
| Versioning et rollback | Model Registry MLflow + alias `challenger` / `production` |
| Documentation API | Swagger interactif (`/docs`) + exemples |

---

## 12. Évolutions identifiées

Pistes d'industrialisation, hors périmètre de la version remise (contraintes de temps) :

- **Gate champion/challenger.** Comparer automatiquement le `challenger` au modèle
  `production` sur un jeu de validation figé, et ne promouvoir que s'il est meilleur
  (rollback automatique sinon). La promotion est aujourd'hui manuelle.
- **Alerte active sur dérive.** Notification (e-mail SMTP ou webhook) en complément
  des rapports archivés.
- **CI à deux niveaux.** Pull request en *smoke test*, exécution réelle réservée à
  `main` et au déclenchement manuel.
- **Versioning des données.** DVC pour tracer explicitement les jeux de référence.
- **Passage à l'échelle.** Migration du service depuis Hugging Face vers une cible
  orchestrée (ECS / Kubernetes) pour la montée en charge et l'autoscaling.

---

## 13. Démarrage rapide

```bash
# 1. Créer le stockage : base Neon + bucket S3
# 2. Déployer le serveur MLflow (Space Docker)
# 3. Entraîner et promouvoir le modèle
pip install -r train/requirements.txt
python train/train.py                 # log + register dans MLflow
# (promouvoir la version en alias 'production' via l'UI MLflow)

# 4. Déployer l'API modèle et l'API données (Spaces Docker)

# 5. Lancer l'ETL
pip install -r requirements.txt
python etl.py                         # ou docker run --rm --env-file .env ibm-attrition-etl

# 6. Tests
pip install -r requirements-tests.txt
pytest -v
```

---

## Auteur

**Ahmed Mehdi SEMAR** — Livrable Bloc 4 (MLOps), certification AIA — Architecte en
Intelligence Artificielle.
Dépôt associé : [train-repo](https://github.com/semarmehdi/train-repo)
(entraînement automatisé et monitoring).
