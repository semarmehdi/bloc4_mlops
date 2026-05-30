import mlflow
import uvicorn
import pandas as pd
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile
import joblib
from dotenv import load_dotenv
import os

description = """
Welcome to Mehdi MLops demo API. This app is made for you to understand how FastAPI works! Try it out 🕹️

## Introduction Endpoints

Here are two endpoints you can try:
* `/`: **GET** request that display a simple default message.

## Data Preview

Just a quick look at the train data. 

## Machine Learning

This is a Machine Learning endpoint that predict attrition given all the data of the employee. Here is the endpoint:

* `/predict` that accepts `floats`


Check out documentation below 👇 for more information on each endpoint. 
"""

tags_metadata = [
    {
        "name": "Introduction Endpoints",
        "description": "Simple endpoints to try out!",
    },
    {
        "name": "Data Preview",
        "description": "An endpoint to look at a sample of the original dataset.",
    },
    {"name": "Machine Learning", "description": "Prediction Endpoint."},
]


load_dotenv()
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
REGISTERED_MODEL_NAME = "ibm_attrition_detector"
MODEL_ALIAS = "production"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
MODEL = mlflow.sklearn.load_model(MODEL_URI)


app = FastAPI(
    title="🪐 Mehdi Demo MLops API",
    description=description,
    version="0.1",
    contact={
        "name": "Mehdi",
        "url": "semarmehdi1@gmail.com",
    },
    openapi_tags=tags_metadata,
)


class PredictionFeatures(BaseModel):
    Age: Union[int, float]
    BusinessTravel: str
    DailyRate: Union[int, float]
    Department: str
    DistanceFromHome: Union[int, float]
    Education: Union[int, float]
    EducationField: str
    EmployeeCount: Union[int, float]
    EmployeeNumber: Union[int, float]
    EnvironmentSatisfaction: Union[int, float]
    Gender: str
    HourlyRate: Union[int, float]
    JobInvolvement: Union[int, float]
    JobLevel: Union[int, float]
    JobRole: str
    JobSatisfaction: Union[int, float]
    MaritalStatus: str
    MonthlyIncome: Union[int, float]
    MonthlyRate: Union[int, float]
    NumCompaniesWorked: Union[int, float]
    Over18: str
    OverTime: str
    PercentSalaryHike: Union[int, float]
    PerformanceRating: Union[int, float]
    RelationshipSatisfaction: Union[int, float]
    StandardHours: Union[int, float]
    StockOptionLevel: Union[int, float]
    TotalWorkingYears: Union[int, float]
    TrainingTimesLastYear: Union[int, float]
    WorkLifeBalance: Union[int, float]
    YearsAtCompany: Union[int, float]
    YearsInCurrentRole: Union[int, float]
    YearsSinceLastPromotion: Union[int, float]
    YearsWithCurrManager: Union[int, float]


@app.get("/", tags=["Introduction Endpoints"])
async def index():
    """
    Simply returns a welcome message!
    """
    message = "Hello world! This `/` is the most simple and default endpoint. If you want to learn more, check out documentation of the api at `/docs`"
    return message


@app.get("/preview", tags=["Data Preview"])
async def preview(ligne: int):
    """
    Append a new blog article into the database which is a `.csv` file.
    """
    df = pd.read_excel(
        "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/ibm_hr_attrition.xlsx",
        index_col=0,
    )
    df_preview = df.sample()

    return df_preview.head(ligne).to_json()


@app.post("/predict", tags=["Machine Learning"])
async def predict(predictionFeatures: PredictionFeatures):
    """
    Prediction of attrition!
    """

    ####### version de Mehdi #######

    employee_info = pd.DataFrame([predictionFeatures.dict()])

    prediction = MODEL.predict(employee_info)

    response = {"prediction": prediction.tolist()[0]}
    return response
