import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from src.constant.training_pipeline import TARGET_COLUMN
from src.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Initialises a KNNImputer with the parameters from training_pipeline constants
        and wraps it in a sklearn Pipeline.

        The preprocessor is intentionally lightweight:
            - KNNImputer: fills any missing values using k-nearest neighbours
            - No scaler is applied — features are already encoded as {-1, 0, 1}
              so scaling would distort the discrete encoding.

        Returns:
            Pipeline: fitted preprocessor ready for .transform()
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initialised KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor: Pipeline = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # ── Training split ────────────────────────────────────────────────
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            # ── Testing split ─────────────────────────────────────────────────
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]

            # ── IMPORTANT: Do NOT remap target labels ─────────────────────────
            # The original dataset uses {-1, 0, 1} for {phishing, suspicious, legitimate}.
            # A previous version incorrectly applied .replace(-1, 0), which collapsed
            # phishing (-1) and suspicious (0) into the same class, making the model
            # unable to distinguish between them or ever output -1.
            #
            # The labels must stay as-is:
            #   -1 → phishing
            #    0 → suspicious
            #    1 → legitimate
            #
            # prediction_pipeline.py and the API label map both rely on this encoding.

            # ── Fit preprocessor on training features only ────────────────────
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            # ── Transform features (impute missing values) ────────────────────
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # ── Concatenate features + target into final arrays ───────────────
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # ── Save outputs ──────────────────────────────────────────────────
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )
            # Save preprocessor to Artifacts/ (used by model_trainer.py)
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object,
            )
            # Save preprocessor to final_model/ (used by prediction_pipeline.py at inference)
            # This is the single source of truth — model_trainer.py does NOT duplicate this.
            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            logging.info("Data transformation completed successfully")

            # ── Build and return artifact ─────────────────────────────────────
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)
