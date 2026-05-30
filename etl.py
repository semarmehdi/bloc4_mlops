import logging
import sys
from dotenv import load_dotenv

# Grâce au __init__.py, l'importation est centralisée et ultra propre
from utils import extract_employees, load_employees, transform_employees

# Configuration globale des logs pour voir défiler les étapes
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def run_pipeline():
    logging.info("==================================================")
    logging.info("🚀 DÉBUT DU PIPELINE ETL (SANS AIRFLOW)")
    logging.info("==================================================")

    try:
        # CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
        # Idéal de le faire au tout début du point d'entrée unique
        load_dotenv()
        logging.info("Variables d'environnement chargées avec succès.")

        # 1. EXTRACT
        logging.info("--- Étape 1 : Extraction des données API & Backup S3 ---")
        raw_data = extract_employees()

        # 2. TRANSFORM
        logging.info("--- Étape 2 : Transformation & Prédiction du modèle ---")
        processed_df = transform_employees(raw_data)

        # 3. LOAD
        logging.info("--- Étape 3 : Chargement final (S3 Clean & Postgres) ---")
        load_summary = load_employees(processed_df)

        # FIN ET RÉSUMÉ
        logging.info("==================================================")
        logging.info("✅ PIPELINE EXÉCUTÉ AVEC SUCCÈS")
        logging.info(f"Résumé du chargement : {load_summary}")
        logging.info("==================================================")

    except Exception as e:
        logging.error("==================================================")
        logging.error(f"❌ LE PIPELINE A ÉCHOUÉ À UNE ÉTAPE CRITIQUE")
        logging.error(f"Détail de l'erreur : {e}", exc_info=True)
        logging.error("==================================================")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
