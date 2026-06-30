import os
import time

import matplotlib

matplotlib.use("Agg")  # backend headless (CI, pas d'affichage)
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

EXPERIMENT_NAME = "ibm_attrition_detector"
REGISTERED_MODEL_NAME = "ibm_attrition_detector"
DATA_URL = (
    "https://full-stack-assets.s3.eu-west-3.amazonaws.com/"
    "Deployment/ibm_hr_attrition.xlsx"
)
TARGET = "Attrition"
RANDOM_STATE = 42

# Colonnes constantes du dataset IBM HR Attrition (variance nulle).
# On les GARDE dans le schéma d'entrée pour que le modèle reste un
# remplaçant "drop-in" de l'API en prod (même contrat JSON) ; le
# OneHotEncoder/StandardScaler les neutralise de toute façon.
CONSTANT_COLS = ["EmployeeCount", "StandardHours", "Over18"]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Charge le dataset et sépare features / cible binaire."""
    df = pd.read_excel(DATA_URL, index_col=0)
    y = df[TARGET].map({"No": 0, "Yes": 1})
    X = df.drop(columns=[TARGET])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Encodage des catégorielles + standardisation des numériques."""
    categorical = X.select_dtypes("object").columns
    numerical = X.columns[~X.columns.isin(categorical)]
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                # handle_unknown='ignore' : robustesse en prod si une
                # modalité inconnue arrive au moment de l'inférence.
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                categorical,
            ),
            ("numerical", StandardScaler(), numerical),
        ]
    )


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)
    # autolog capture params/métriques d'entraînement et la recherche CV.
    # log_models=False : on logge le modèle nous-mêmes (signature + registry).
    mlflow.sklearn.autolog(log_models=False, max_tuning_runs=5)

    # --- Pilotage par variables d'environnement (cf. train.yaml) ---
    alias_name = os.getenv("REGISTER_ALIAS", "challenger")
    tune = os.getenv("TUNE", "true").lower() == "true"
    n_estimators = int(os.getenv("N_ESTIMATORS", "300"))
    min_samples_split = int(os.getenv("MIN_SAMPLES_SPLIT", "2"))

    # --- Données : split stratifié (l'attrition est déséquilibrée ~16%) ---
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor(X_train)),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced",  # compense le déséquilibre
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    with mlflow.start_run(run_name=f"rf_{'tuned' if tune else 'fixed'}") as run:
        start = time.time()

        if tune:
            # Recherche d'hyperparamètres optimisée sur le F1 (déséquilibre).
            param_grid = {
                "classifier__n_estimators": [150, 300],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 8],
            }
            search = GridSearchCV(
                pipe, param_grid, scoring="f1", cv=5, n_jobs=-1, refit=True
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
        else:
            # Mode rapide (CI) : entraînement direct avec HP fournis.
            pipe.set_params(
                classifier__n_estimators=n_estimators,
                classifier__min_samples_split=min_samples_split,
            )
            pipe.fit(X_train, y_train)
            model = pipe
            best_params = {
                "classifier__n_estimators": n_estimators,
                "classifier__min_samples_split": min_samples_split,
            }

        # --- Évaluation sur le test (jamais vu pendant l'entraînement) ---
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params(
            {k.replace("classifier__", "best_"): v for k, v in best_params.items()}
        )

        report = classification_report(
            y_test, y_pred, target_names=["Stay (0)", "Leave (1)"]
        )
        mlflow.log_text(report, "classification_report.txt")
        print(report)
        print("Best params :", best_params)
        print("Test metrics:", metrics)

        # Matrice de confusion en artefact
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["Stay", "Leave"], ax=ax, colorbar=False
        )
        ax.set_title("Confusion matrix (test)")
        fig.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # --- Log + enregistrement du modèle dans le registry ---
        signature = infer_signature(X_test, y_pred)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=X_test.head(5),
        )

        version = model_info.registered_model_version
        MlflowClient().set_registered_model_alias(
            name=REGISTERED_MODEL_NAME, alias=alias_name, version=version
        )

        mlflow.set_tag("tuned", tune)
        mlflow.set_tag("alias", alias_name)

        print(
            f"[INFO] Modèle enregistré v{version}, alias '{alias_name}' -> v{version}"
        )
        print(f"[INFO] Temps total : {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
