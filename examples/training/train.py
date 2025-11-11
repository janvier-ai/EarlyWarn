import logging
from typer import Typer
from typer_config import use_yaml_config
from typing import List, Dict

from aeon.classification.convolution_based import MiniRocketClassifier

from src.earlywarn.datasets._data_loaders import load_dataset
from src.earlywarn.utils.build import build_from_cfg
from src.earlywarn.cost import Cost

app = Typer(pretty_exceptions_enable=False)

@app.command()
@use_yaml_config(param_name="config")
def main(dataset: str, 
         training_data_paths: List[str], 
         validation_data_paths: List[str], 
         alphas: List[float],
         anticipation: Dict,
         results_save_path: str, 
         models_save_path: str, 
         context_length: int, 
         prediction_horizon: int,
         cost_cfg: Dict,
         triggers_cfg: List[Dict],
         classifiers_cfg: List[Dict],  
         extractors_cfg: List[Dict] = [{}], 
         classifier_metrics: List[str] = ["accuracy"],
         random_seed: int = 42):
    logger.info("Starting training with the following configuration:")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Training data paths: {training_data_paths}")
    logger.info(f"Validation data paths: {validation_data_paths}")
    logger.info(f"Anticipation: {anticipation}")
    logger.info(f"Results save path: {results_save_path}")
    logger.info(f"Models save path: {models_save_path}")
    logger.info(f"Context length: {context_length}")
    logger.info(f"Prediction horizon: {prediction_horizon}")
    logger.info(f"Classifiers: {classifiers_cfg}")
    logger.info(f"Extractors: {extractors_cfg}")
    logger.info(f"Extractors: {triggers_cfg}")
    
    # Train
    # Loop through combinations of extractors, classifiers, triggers, and alphas for each training sub-dataset

    # Read and preprocess training and validation data
    logger.info(f"Loading dataset: {dataset}")
    for training_path, validation_path in zip(training_data_paths, validation_data_paths):
        logger.info(f"Loading training data from: {training_path}")
        logger.info(f"Loading validation data from: {validation_path}")
        X_train, y_train = load_dataset(training_path)
        X_val, y_val = load_dataset(validation_path)

        for extr_cfg in extractors_cfg:
            logger.info(f"Using extractor: {extr_cfg}")
            # Initialize extractor
            extractor = build_from_cfg(extr_cfg)
            # Extract features from training and validation data
            logger.info(f"Extracting training features (i.e. fit_transform) for data: {training_path}")
            # Extract features
            X_train_feat = extractor.fit_transform(X_train)
            logger.info(f"Extracting validation features (i.e. transform) for data: {validation_path}")
            X_val_feat = extractor.transform(X_val)
            for clsf_cfg in classifiers_cfg:
                logger.info(f"Using classifier: {clsf_cfg}")
                # Initialize classifier
                classifier = build_from_cfg(clsf_cfg)
                # Train classifier on extracted training data
                logger.info(f"Training classifier on extracted training features.")
                classifier.fit(X_train_feat, y_train)
                # Evaluate classifier on extracted validation data
                val_scores = {classifier.score(X_val_feat, y_val, metric=metric) for metric in classifier_metrics}
                logger.info(f"Validation score of classifier: {val_scores}")
                for trig_cfg in triggers_cfg:
                    logger.info(f"Using trigger: {trig_cfg}")
                    for alpha in alphas:
                        logger.info(f"Training with alpha: {alpha}")
                        # Initialize cost matrix
                        cost = Cost(
                            misclassification=cost_cfg["misclassification"],
                            delay=cost_cfg["delay"],
                            combination=cost_cfg["combination"]
                        )
                        # Initialize Trigger, add to EarlyWarner
                        trigger = build_from_cfg(trig_cfg)
                        # Train EarlyWarner with extractor, classifier, trigger, cost, alpha
                        logger.info(f"Model trained and saved for alpha: {alpha}")

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    app()