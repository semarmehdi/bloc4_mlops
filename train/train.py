import argparse
import pandas as pd
import time
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os
import boto3

load_dotenv()

# Set tracking URI to your hf application
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

if __name__ == "__main__":

    ### MLFLOW Experiment setup
    experiment_name = "ibm_attrition_detector"

    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)

    print("training model...")

    # Time execution
    start_time = time.time()

    # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=False)  # We won't log models right away

    # # Parse arguments given in shell script
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--n_estimators")
    # parser.add_argument("--min_samples_split")
    # args = parser.parse_args()

    # Import dataset
    filename = "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/ibm_hr_attrition.xlsx"
    df = pd.read_excel(
        filename,
        index_col=0,
    )

    # X, y split
    target_col_name = "Attrition"
    X_train = df.loc[:, df.columns != target_col_name]
    y_train = df.loc[:, target_col_name].apply(lambda x: 0 if x == "No" else 1)

    # # Train / test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Preprocessing
    categorical_features = X_train.select_dtypes(
        "object"
    ).columns  # Select all the columns containing strings
    # categorical_transformer = OneHotEncoder(
    #     drop="first", handle_unknown="error", sparse_output=False
    # )
    categorical_transformer = OneHotEncoder(
        drop="first", handle_unknown="ignore", sparse_output=False  # 👈 important
    )

    numerical_feature_mask = ~X_train.columns.isin(
        X_train.select_dtypes("object").columns
    )  # Select all the columns containing anything else than strings
    numerical_features = X_train.columns[numerical_feature_mask]
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical_transformer", categorical_transformer, categorical_features),
            ("numerical_transformer", numerical_transformer, numerical_features),
        ]
    )

    # Pipeline
    # n_estimators = int(args.n_estimators)
    # min_samples_split = int(args.min_samples_split)
    n_estimators = 100
    min_samples_split = 2

    model = Pipeline(
        steps=[
            ("Preprocessing", preprocessor),
            (
                "Classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators, min_samples_split=min_samples_split
                ),
            ),
        ],
        verbose=True,
    )

    # Log experiment to MLFlow
    with mlflow.start_run() as run:

        model.fit(X_train, y_train)

        predictions = model.predict(X_train)

        # Log model seperately to have more flexibility on setup

        signature = infer_signature(X_train, predictions)
        input_example = X_train.head(5)
        registered_model_name = "ibm_attrition_detector"

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
        )

        alias_name = "challenger"

        model_version = model_info.registered_model_version
        print(f"[INFO] Model logged as version {model_version}")

        client.set_registered_model_alias(
            name=registered_model_name,
            alias=alias_name,
            version=model_version,
        )

        print(f"[INFO] Alias '{alias_name}' now points to version {model_version}")

    print("...Done!")
    print(f"---Total training time: {time.time()-start_time}")
