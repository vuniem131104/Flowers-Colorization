from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline
STAGE_1_NAME = "Data Ingestion stage"    
STAGE_2_NAME = "Data Transformation stage"
STAGE_3_NAME = "Model Trainer stage"
STAGE_4_NAME = "Model Evaluation stage"
if __name__ == "__main__":
    # print(f">>>>>> stage {STAGE_1_NAME} started <<<<<<")
    # data_ingestion_pipeline = DataIngestionPipeline()
    # data_ingestion_pipeline.initiate_data_ingestion_pipeline()
    # print(f">>>>>> stage {STAGE_1_NAME} completed <<<<<<")
    # print(f">>>>>> stage {STAGE_2_NAME} started <<<<<<")
    # data_transformation_pipeline = DataTransformationPipeline()
    # data_transformation_pipeline.initiate_data_transformation_pipeline()
    # print(f">>>>>> stage {STAGE_2_NAME} completed <<<<<<")
    # print(f">>>>>> stage {STAGE_3_NAME} started <<<<<<")
    model_trainer_pipeline = ModelTrainerPipeline()
    model_trainer_pipeline.initiate_model_trainer_pipeline()
    print(f">>>>>> stage {STAGE_3_NAME} completed <<<<<<")
    print(f">>>>>> stage {STAGE_4_NAME} started <<<<<<")
    model_evaluation_pipeline = ModelEvaluationPipeline()
    model_evaluation_pipeline.initiate_model_evaluation_pipeline()
    print(f">>>>>> stage {STAGE_4_NAME} completed <<<<<<")