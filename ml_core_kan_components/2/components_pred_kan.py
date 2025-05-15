import os
# Set up logging to file
import logging
import datetime
from analysis_functions import opener, KANModelTrainTest, logger


def main():
    # Print project directory tree
    print("Project directory structure:")
    for root, dirs, files in os.walk('.', topdown=True):
        level = root.count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")
    print("="*50)

    # Load data
    X_stress_components_new = opener(
        "X_stress_components_new_components", path_import="./new_components_resources/"
    )
    y_stress_components_new = opener(
        "y_stress_components_new_components", path_import="./new_components_resources/"
    )
    print(X_stress_components_new.shape)

    component_num = 2
    n_trials = 100
    
    # Init KAN model
    kan_model = KANModelTrainTest(use_gpu=True) # type: ignore

    kan_model.create_train_val_test(
        X=X_stress_components_new[component_num],
        y=y_stress_components_new[component_num],
        n_splits=1,
    )

    ds = kan_model.create_dataset(
        kan_model.train_set_X[0],
        kan_model.train_set_y[0],
        kan_model.val_set_X[0],
        kan_model.val_set_y[0],
    )

    print("Train data shape: {}".format(ds["train_input"].shape))
    print("Train target shape: {}".format(ds["train_label"].shape))
    print("Test data shape: {}".format(ds["test_input"].shape))
    print("Test target shape: {}".format(ds["test_label"].shape))
    print("====================================")


    
    # Create output directory
    # os.makedirs("/output", exist_ok=True)
    artifact_file = "MY_ARTIFACT"
    
    # Configure logging
    # logger = logging.getLogger("kan_training")
    logger.setLevel(logging.INFO)
    
    # Create file handler for our artifact file
    file_handler = logging.FileHandler(artifact_file)
    file_handler.setLevel(logging.INFO)
    
    # Format for log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to the logger
    logger.addHandler(file_handler)
    
    # Log basic information
    logger.info(f"KAN Model Training Log - {datetime.datetime.now()}")
    logger.info("="*50)
    logger.info(f"Component Number: {component_num}")
    logger.info(f"Number of Trials: {n_trials}")
    logger.info("Training Process:")
    
    # Create a custom writer that redirects stdout to our logger
    class LoggerWriter:
        def __init__(self, logger):
            self.logger = logger
            self.buffer = ""
            
        def write(self, message):
            if message.strip() != "":
                self.buffer += message
                if "\n" in message:
                    lines = self.buffer.split("\n")
                    for line in lines[:-1]:
                        if line.strip():
                            self.logger.info(line.rstrip())
                    self.buffer = lines[-1] if lines[-1] else ""
                    
        def flush(self):
            if self.buffer:
                self.logger.info(self.buffer.rstrip())
                self.buffer = ""
    
    # Redirect stdout to our logger
    import sys
    original_stdout = sys.stdout
    sys.stdout = LoggerWriter(logger)
    
    # Run optimization with direct logging
    best_params = kan_model.optimize_hyperparams(
        n_trials=n_trials,
        max_n_layers=15,
        max_n_units=100,
        max_steps=500,
        max_grid=6,
        max_k=6,
        n_jobs=30,
    )
    
    # Restore original stdout and stderr
    sys.stdout = original_stdout
    
    # Log best parameters
    logger.info(f"Best Parameters:\n{best_params}")
    
    # Add model evaluation metrics if available
    logger.info("Model Evaluation:")
    try:
        kan_model.calc_test_metric()
        if hasattr(kan_model, 'test_rmse'):
            logger.info(f"Test RMSE: {kan_model.test_rmse}")
        if hasattr(kan_model, 'test_r2'):
            logger.info(f"Test R²: {kan_model.test_r2}")
    except Exception as e:
        logger.error(f"Could not calculate test metrics: {str(e)}")
    
    # Print to console as well
    print("Best parameters:", best_params)
    print(f"Training logs and best parameters written to {artifact_file}")
    
    
if __name__ == "__main__":
    main()
