from datetime import datetime
import json
import logging
import os
import boto3
import pandas as pd
from sqlalchemy import create_engine, text


def load_employees(df_processed):
    """Prend le DataFrame transformé, l'exporte en CSV sur S3,

    crée la table Postgres si nécessaire, et y insère les données.
    """
    # ==========================================
    # 1) CONFIGURATION VIA ENV
    # ==========================================
    # Configuration S3
    bucket = os.getenv("S3BucketName")
    s3_prefix_predictions = os.getenv("IBM_ATTRITION_S3_PRED_PREFIX")

    # Configuration Postgres / Neon
    db_url = os.getenv(
        "DATABASE_URL",
        default="postgresql://neondb_owner:npg_0HLv6RGbCcPn@ep-cold-cherry-agr3ixjk-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require",
    )
    table_name = os.getenv("DB_TARGET_TABLE", default="ibm_attrition_predictions")

    # Génération du nom de fichier unique basé sur le timestamp actuel
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{ts}_ibm_predictions.csv"

    # ==========================================
    # 2) SAUVEGARDE DU CSV SUR S3 (Comme l'opérateur)
    # ==========================================
    s3_key_predictions = f"{s3_prefix_predictions}/{filename}"

    # Conversion du DataFrame en string CSV en mémoire (évite le fichier local /tmp)
    csv_buffer = df_processed.to_csv(index=False)

    s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=s3_key_predictions, Body=csv_buffer)
    logging.info(
        f"Predictions sauvegardées sur S3 : s3://{bucket}/{s3_key_predictions}"
    )

    # ==========================================
    # 3) CRÉATION DE LA TABLE POSTGRES (DDL)
    # ==========================================
    engine = create_engine(db_url)

    ddl_predictions = f"""
    CREATE TABLE IF NOT EXISTS public.{table_name} (
        id SERIAL PRIMARY KEY,
        "Age" TEXT,
        "BusinessTravel" TEXT,
        "DailyRate" TEXT,
        "Department" TEXT,
        "DistanceFromHome" TEXT,
        "Education" TEXT,
        "EducationField" TEXT,
        "EmployeeCount" TEXT,
        "EmployeeNumber" TEXT,
        "EnvironmentSatisfaction" TEXT,
        "Gender" TEXT,
        "HourlyRate" TEXT,
        "JobInvolvement" TEXT,
        "JobLevel" TEXT,
        "JobRole" TEXT,
        "JobSatisfaction" TEXT,
        "MaritalStatus" TEXT,
        "MonthlyIncome" TEXT,
        "MonthlyRate" TEXT,
        "NumCompaniesWorked" TEXT,
        "Over18" TEXT,
        "OverTime" TEXT,
        "PercentSalaryHike" TEXT,
        "PerformanceRating" TEXT,
        "RelationshipSatisfaction" TEXT,
        "StandardHours" TEXT,
        "StockOptionLevel" TEXT,
        "TotalWorkingYears" TEXT,
        "TrainingTimesLastYear" TEXT,
        "WorkLifeBalance" TEXT,
        "YearsAtCompany" TEXT,
        "YearsInCurrentRole" TEXT,
        "YearsSinceLastPromotion" TEXT,
        "YearsWithCurrManager" TEXT,
        prediction TEXT,
        proba_0 TEXT,
        proba_1 TEXT
    );
    """

    with engine.begin() as conn:
        conn.execute(text(ddl_predictions))
    logging.info(f"Table '{table_name}' vérifiée/créée avec succès.")

    # ==========================================
    # 4) INSERTION DANS POSTGRES VIA PANDAS
    # ==========================================
    # Utilisation du mode 'append' pour ajouter les lots successifs sans écraser l'historique
    df_processed.to_sql(table_name, engine, if_exists="append", index=False)
    logging.info(
        f"Insertion réussie dans Postgres ({len(df_processed)} lignes ajoutées)."
    )

    return {"s3_key": s3_key_predictions, "rows_loaded": len(df_processed)}
