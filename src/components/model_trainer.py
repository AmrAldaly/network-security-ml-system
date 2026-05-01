import os
import sys

from src.exception.exception import CustomException 
from src.logging.logger import logging

from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.ml_utils.model.estimator import NetworkModel
from src.utils.main_utils.utils import save_object, load_object
from src.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from src.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

import mlflow

import dagshub
dagshub.init(repo_owner='AmrAldaly', repo_name='network-security-ml-system', mlflow=True)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def track_mlflow(self, best_model, best_model_name, train_metric, test_metric):
        # FIX 1: try/except now wraps the entire `with` block so the run
        #         context closes cleanly before any exception propagates.
        try:
            with mlflow.start_run():
                # Train metrics
                mlflow.log_metric("train_f1_score", train_metric.f1_score)
                mlflow.log_metric("train_precision", train_metric.precision_score)
                mlflow.log_metric("train_recall", train_metric.recall_score)

                # Test metrics
                mlflow.log_metric("test_f1_score", test_metric.f1_score)
                mlflow.log_metric("test_precision", test_metric.precision_score)
                mlflow.log_metric("test_recall", test_metric.recall_score)

                # FIX 2: Use best_model_name instead of hardcoded "model"
                # FIX 3: registered_model_name makes the model appear in the
                #         Model Registry tab on DagsHub (and uploads the artifact)
                mlflow.sklearn.log_model(
                    best_model,
                    best_model_name,
                    registered_model_name=best_model_name
                )

        except Exception as e:
            raise CustomException(e, sys)

    def train_model(self, X_train, y_train, x_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 128, 256]
            },
            "Gradient Boosting": {
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {},
            "AdaBoost": {
                'learning_rate': [.1, .01, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

        model_report: dict = evaluate_models(
            X_train=X_train, y_train=y_train,
            X_test=x_test, y_test=y_test,
            models=models, param=params
        )

        # Get best model score and name
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]

        # Train metrics
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        # Test metrics
        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        # FIX: Pass best_model_name so MLflow logs the real model name
        self.track_mlflow(best_model, best_model_name, classification_train_metric, classification_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)

        # Save the final raw model to final_model/model.pkl.
        # NOTE: preprocessor.pkl is already saved to final_model/ by
        # data_transformation.py — no need to duplicate it here.
        os.makedirs("final_model", exist_ok=True)
        save_object("final_model/model.pkl", best_model)

        # Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Load training and testing arrays
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
