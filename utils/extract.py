from datetime import datetime
import json
import logging
import os
import time
import boto3
import requests


def get_data(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = json.loads(r.text)
    return json.loads(payload) if isinstance(payload, str) else payload


def extract_employees():
    """Appelle l'API, sauvegarde le JSON sur S3 et retourne la donnée brute."""
    base_url = os.getenv("IBM_ATTRITION_BASE_URL")
    endpoint = os.getenv("IBM_ATTRITION_ENDPOINT")
    batch_size = int(os.getenv("IBM_ATTRITION_BATCH_SIZE", "20"))
    sleep_seconds = float(os.getenv("IBM_ATTRITION_SLEEP_SECONDS", "1.0"))

    bucket = os.getenv("S3BucketName")
    s3_prefix = os.getenv("IBM_ATTRITION_S3_PREFIX")

    url = f"{base_url}{endpoint}"
    items = []

    # 1. Collecte des données
    for i in range(batch_size):
        try:
            data = get_data(url)
            items.append(
                {
                    "pulled_at_utc": datetime.utcnow().isoformat(),
                    "data": data,
                }
            )
        except Exception as e:
            logging.warning(f"Erreur lors du pull {i+1}: {e}")

        if i < batch_size - 1 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    # Structure finale du JSON à stocker
    artifact = {
        "meta": {
            "records_collected": len(items),
            "created_at_utc": datetime.utcnow().isoformat(),
        },
        "items": items,
    }

    # 2. Envoi direct sur S3 (en mémoire, pas de fichier local /tmp)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3_key = f"{s3_prefix}/{timestamp}_ibm_employees.json"

    s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=s3_key, Body=json.dumps(artifact, indent=4))

    logging.info(f"Backup brut envoyé sur S3 : s3://{bucket}/{s3_key}")

    # 3. On renvoie directement l'artefact pour le pipeline (etl.py)
    return artifact
