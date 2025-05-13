import os

# Set up logging to file
import logging
import datetime
import torch
from analysis_functions import opener

# Импортируем новую GPU-ускоренную версию оптимизатора гиперпараметров
from gpu_optuna import do_optuna_pytorch


# Класс для перенаправления вывода в логгер
class LoggerWriter:
    def __init__(self, logger):
        self.logger = logger
        self.line_buffer = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.info(line.rstrip())

    def flush(self):
        pass


def main():
    # Print project directory tree
    print("Project directory structure:")
    for root, dirs, files in os.walk(".", topdown=True):
        level = root.count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")
    print("=" * 50)

    # Load data
    X_stress_components_new = opener(
        "X_stress_components_new_components", path_import="./new_components_resources/"
    )
    y_stress_components_new = opener(
        "y_stress_components_new_components", path_import="./new_components_resources/"
    )

    print(X_stress_components_new.shape)

    component_num = 0
    n_trials = 200

    # Проверяем доступность GPU
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    print(f"Используется устройство: {device}")

    # Данные для текущего компонента
    X_current = X_stress_components_new[component_num]
    y_current = y_stress_components_new[component_num]

    # Вывод размерностей данных
    print(f"Форма входных данных X: {X_current.shape}")
    print(f"Форма целевых данных y: {y_current.shape}")
    print("====================================")

    # Обучение GPU-MLP модели с использованием новой реализации
    print("\n## Запуск GPU-ускоренного MLP с оптимизацией гиперпараметров ##")

    # Параметры для GPU-MLP
    mlp_n_trials = n_trials
    mlp_n_splits = 5
    mlp_n_layers = 15  # Максимальное количество слоев для поиска
    mlp_n_neurons = 200  # Максимальное количество нейронов в слое для поиска

    # Запуск оптимизации гиперпараметров MLP на GPU
    print(f"Запуск оптимизации гиперпараметров MLP с {mlp_n_trials} trials...")
    best_mlp_params, X_test_mlp, y_test_mlp, best_mlp_val = do_optuna_pytorch(
        X=X_current,
        y=y_current,
        n_trials=mlp_n_trials,
        n_splits=mlp_n_splits,
        n_layers=mlp_n_layers,
        n_neurons=mlp_n_neurons,
    )
    print("Оптимизация гиперпараметров MLP завершена!")
    print(f"Лучшие параметры MLP: {best_mlp_params}")
    print(f"Лучшее значение метрики: {best_mlp_val}")
    print("====================================")

    # Create output directory
    os.makedirs("/output", exist_ok=True)
    artifact_file = "/output/MY_ARTIFACT"

    # Configure logging
    logger = logging.getLogger("mlp_gpu_training")
    logger.setLevel(logging.INFO)

    # Create file handler for our artifact file
    file_handler = logging.FileHandler(artifact_file)
    file_handler.setLevel(logging.INFO)

    # Format for log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(file_handler)

    # Log basic information
    logger.info(f"GPU-MLP Model Training Log - {datetime.datetime.now()}")
    logger.info("=" * 50)
    logger.info(f"Component Number: {component_num}")
    logger.info(f"Number of Trials: {n_trials}")
    logger.info(f"Device: {device}")

    # Log MLP GPU results
    logger.info("## GPU-MLP MODEL RESULTS ##")
    logger.info(f"Best MLP Parameters: {best_mlp_params}")
    logger.info(f"Best MLP Validation RMSE: {best_mlp_val}")
    logger.info("=" * 50)

    # Redirect stdout to our logger for any additional information
    import sys

    original_stdout = sys.stdout
    sys.stdout = LoggerWriter(logger)

    # Запись дополнительной информации о тестовых данных
    logger.info("Тестовая выборка:")
    logger.info(f"  X_test shape: {X_test_mlp.shape}")
    logger.info(f"  y_test shape: {y_test_mlp.shape}")

    # Restore original stdout
    sys.stdout = original_stdout

    # Print to console as well
    print(f"Best GPU-MLP parameters: {best_mlp_params}")
    print(f"Best GPU-MLP validation RMSE: {best_mlp_val}")
    print(f"Training logs and best parameters written to {artifact_file}")


if __name__ == "__main__":
    main()
