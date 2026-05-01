import os
import sys
from pathlib import Path

from src.exception.exception import CustomException
from src.logging.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

# FIX 1: Removed duplicate `import sys` that appeared after the artifact imports.


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info("Starting Data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config,
            )
            logging.info("Starting Data Validation")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data Validation completed. Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config,
            )
            logging.info("Starting Data Transformation")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation completed. Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            self.model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            logging.info("Starting Model Training")
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model Training completed. Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self) -> ModelTrainerArtifact:
        """
        Executes the full training pipeline in order:
            Data Ingestion → Data Validation → Data Transformation → Model Trainer

        FIX 2: Added pipeline-level start/completion logging so the top-level
                execution is traceable without having to read per-stage logs.
        FIX 3: Added current stage tracking so if the pipeline fails, the log
                clearly shows which stage raised the exception.
        FIX 4: Return type annotation added — run_pipeline returns the
                ModelTrainerArtifact so callers (e.g. main.py) can inspect it.
        """
        current_stage = "Initialisation"
        try:
            logging.info("=" * 60)
            logging.info("Training Pipeline started")
            logging.info("=" * 60)

            current_stage = "Data Ingestion"
            data_ingestion_artifact = self.start_data_ingestion()

            current_stage = "Data Validation"
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            current_stage = "Data Transformation"
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )

            current_stage = "Model Training"
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            logging.info("=" * 60)
            logging.info("Training Pipeline completed successfully")
            logging.info("=" * 60)

            return model_trainer_artifact

        except Exception as e:
            # FIX 3: Log which stage failed before re-raising, so it's visible
            # in the log file without needing to trace back through the stack.
            logging.error(f"Training Pipeline failed at stage: [{current_stage}]")
            raise CustomException(e, sys)


# ── Entry point ───────────────────────────────────────────────────────────────
# This block only runs when the file is executed directly:
#     python src/pipeline/training_pipeline.py
#
# When training_pipeline.py is imported by other modules (e.g. a scheduler or
# a FastAPI background task), this block is skipped — the class is just imported.
#
# Without this block, running the file directly only defines the class and
# triggers any module-level side effects (like dagshub.init from model_trainer
# imports), but never actually starts the pipeline.

if __name__ == "__main__":
    # ── Working directory fix ─────────────────────────────────────────────────
    # When this file is run directly (python src/pipeline/training_pipeline.py),
    # Python sets CWD to src/pipeline/ — so relative paths like
    # 'data_schema/schema.yaml' or 'final_model/' fail with FileNotFoundError.
    #
    # This block walks up from this file's location (__file__) until it finds
    # the project root (identified by setup.py or requirements.txt), then sets
    # CWD there — exactly the same working directory that main.py uses.
    _file_path = Path(__file__).resolve()
    _project_root = next(
        p for p in _file_path.parents
        if (p / "setup.py").exists() or (p / "requirements.txt").exists()
    )
    os.chdir(_project_root)

    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        raise CustomException(e, sys)
