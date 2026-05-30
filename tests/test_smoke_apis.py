import os
import pytest
import requests

from utils import extract

DATA_API_READY = bool(
    os.getenv("IBM_ATTRITION_BASE_URL") and os.getenv("IBM_ATTRITION_ENDPOINT")
)
MODEL_API_READY = bool(os.getenv("IBM_ATTRITION_MODEL_API_BASE_URL"))

# Un employé d'exemple conforme au schéma PredictionFeatures de ton app.py
SAMPLE_EMPLOYEE = {
    "Age": 36,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 852,
    "Department": "Research & Development",
    "DistanceFromHome": 5,
    "Education": 4,
    "EducationField": "Life Sciences",
    "EmployeeCount": 1,
    "EmployeeNumber": 51,
    "EnvironmentSatisfaction": 2,
    "Gender": "Female",
    "HourlyRate": 82,
    "JobInvolvement": 2,
    "JobLevel": 1,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 1,
    "MaritalStatus": "Married",
    "MonthlyIncome": 3419,
    "MonthlyRate": 13072,
    "NumCompaniesWorked": 9,
    "Over18": "Y",
    "OverTime": "Yes",
    "PercentSalaryHike": 14,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 4,
    "StandardHours": 80,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 6,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 4,
    "YearsAtCompany": 1,
    "YearsInCurrentRole": 1,
    "YearsSinceLastPromotion": 0,
    "YearsWithCurrManager": 0,
}


@pytest.mark.skipif(not DATA_API_READY, reason="Variables d'env API données absentes")
def test_smoke_data_api_repond_et_parse():
    """L'API source répond et renvoie une structure colonnes/data exploitable."""
    url = os.getenv("IBM_ATTRITION_BASE_URL") + os.getenv("IBM_ATTRITION_ENDPOINT")
    payload = extract.get_data(url)  # on teste l'API ET le helper de parsing
    assert isinstance(payload, dict)
    assert "columns" in payload and "data" in payload
    assert len(payload["columns"]) > 0


@pytest.mark.skipif(not MODEL_API_READY, reason="Variables d'env API modèle absentes")
def test_smoke_model_api_predict():
    """L'API modèle répond 200 et renvoie au moins une prédiction."""
    base = os.getenv("IBM_ATTRITION_MODEL_API_BASE_URL")
    endpoint = os.getenv("IBM_ATTRITION_MODEL_API_PREDICT_ENDPOINT", "/predict")
    timeout = int(os.getenv("IBM_ATTRITION_MODEL_API_TIMEOUT", "60"))

    r = requests.post(f"{base}{endpoint}", json=SAMPLE_EMPLOYEE, timeout=timeout)
    assert r.status_code == 200

    body = r.json()
    assert "prediction" in body
    # Si tu as bien ajouté predict_proba dans /predict, ces clés doivent être là :
    assert "proba_0" in body and "proba_1" in body
