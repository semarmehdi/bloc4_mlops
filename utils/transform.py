from datetime import datetime
import json
import logging
import os
import pandas as pd
import requests


def item_to_row(it, rows):
    payload = it.get("data")

    if isinstance(payload, str):
        payload = json.loads(payload)

    cols = payload.get("columns")
    data = payload.get("data")

    if not cols or not data:
        return

    values = data[0] if isinstance(data, list) and len(data) > 0 else []
    row = {c: v for c, v in zip(cols, values)}

    if "current_time" in row:
        row.pop("current_time")

    rows.append(row)


def predict_from_api(predict_url, payload, request_timeout):
    """Appelle l'API de prédiction une seule fois et renvoie le tuple de résultats."""
    response = requests.post(
        predict_url,
        json=payload,
        timeout=request_timeout,
    )
    response.raise_for_status()
    result = response.json()

    prediction = result.get("prediction")
    proba_0 = result.get("proba_0")
    proba_1 = result.get("proba_1")

    # Gestion si l'API renvoie des listes au lieu de scalaires
    if isinstance(prediction, list) and len(prediction) > 0:
        prediction = prediction[0]
    if isinstance(proba_0, list) and len(proba_0) > 0:
        proba_0 = proba_0[0]
    if isinstance(proba_1, list) and len(proba_1) > 0:
        proba_1 = proba_1[0]

    return prediction, proba_0, proba_1


def transform_employees(raw_batch):
    """Prend l'artefact brut en entrée, reconstruit les lignes,

    enrichit avec les prédictions du modèle, et renvoie un DataFrame Pandas.
    """
    # 1) Récupération de la configuration via les variables d'environnement
    request_timeout = int(os.getenv("IBM_ATTRITION_MODEL_API_TIMEOUT", "120"))
    model_api_base_url = os.getenv("IBM_ATTRITION_MODEL_API_BASE_URL")
    model_api_predict_endpoint = os.getenv(
        "IBM_ATTRITION_MODEL_API_PREDICT_ENDPOINT", "/predict"
    )
    predict_url = f"{model_api_base_url}{model_api_predict_endpoint}"

    # 2) Extraction des items depuis l'artefact reçu de l'extract
    items = raw_batch.get("items", [])
    if not items:
        raise ValueError("Le batch brut ne contient aucun item.")

    # 3) Reconstruction des lignes (Format split -> dictionnaire)
    rows = []
    for it in items:
        item_to_row(it, rows)

    if not rows:
        raise ValueError("Aucune ligne exploitable n'a pu être reconstruite.")

    features = pd.DataFrame(rows)
    logging.info(f"DataFrame initial créé : shape={features.shape}")

    # 4) Appels de l'API de prédiction ligne par ligne
    predictions = []
    proba_0_list = []
    proba_1_list = []

    for idx, row in features.iterrows():
        payload = json.loads(row.to_json())  # Conversion propre en JSON natif

        try:
            # UN SEUL appel API ici qui récupère les 3 éléments d'un coup
            pred, p0, p1 = predict_from_api(predict_url, payload, request_timeout)

            predictions.append(pred)
            proba_0_list.append(p0)
            proba_1_list.append(p1)
            logging.info(f"Prédiction OK pour la ligne {idx}")

        except Exception as e:
            logging.error(f"Échec de la prédiction pour la ligne {idx} : {e}")
            predictions.append(None)
            proba_0_list.append(None)
            proba_1_list.append(None)

    # 5) Construction du DataFrame de résultats final
    result_df = features.copy()
    result_df["prediction"] = predictions
    result_df["proba_0"] = proba_0_list
    result_df["proba_1"] = proba_1_list

    logging.info(f"Transformation terminée avec succès (rows={len(result_df)})")

    # On retourne le DataFrame prêt à être chargé (Load)
    return result_df
