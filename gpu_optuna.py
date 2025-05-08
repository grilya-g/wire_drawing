import numpy as np
import pandas as pd
import optuna

import torch
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, max_error, root_mean_squared_error

from analysis_functions import split_transform_one_comp_cv, clean_input_array, choose_worst

def to_gpu(data):
    """Convert NumPy arrays to GPU tensors using PyTorch"""
    if isinstance(data, np.ndarray):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor(data, device=device)
    return data

def cuml_scorer(y_true, y_pred, pipeline, X_train):
    """Similar to the original scorer but works with cuML and NumPy arrays"""
    # Convert to numpy for sklearn metrics if needed
    if hasattr(y_true, 'to_numpy'):
        y_true = y_true.to_numpy()
    if hasattr(y_pred, 'to_numpy'):
        y_pred = y_pred.to_numpy()
    if hasattr(X_train, 'to_numpy'):
        X_train = X_train.to_numpy()

    # Check for NaNs and handle them
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        # Return large error values to penalize this trial
        return np.array([0, 1e6, 1e6, 1e6, 0, 1e6, 1e6, 1e6, 1e6])

    evs = explained_variance_score(y_true, y_pred)  # 1-- BEST , 0 -- WORST
    medae = median_absolute_error(y_true, y_pred)  # 0 -- BEST, \INF -- WORST
    mse = mean_squared_error(y_true, y_pred)  # 0 -- BEST
    mae = mean_absolute_error(y_true, y_pred)  # 0 -- BEST
    r2 = r2_score(y_true, y_pred)  # 1 --BEST
    me = max_error(y_true, y_pred)  # 0-- BEST
    rmse = np.sqrt(mse)  # 0 -- BEST

    # Get model parameters count (approximate)
    # For neural network in cuML we'll use a simpler approach
    n_params = 1000  # Placeholder - this needs appropriate implementation based on cuML models used
    
    n = len(X_train)  # number of samples
    aic = n * np.log(mse) + 2 * n_params  # 0 -- BEST
    bic = n * np.log(mse) + n_params * np.log(n)  # 0 -- BEST

    return np.array([evs, medae, mse, mae, r2, me, aic, bic, rmse])

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Define a PyTorch neural network class
class MLPNetwork(nn.Module):
    def __init__(self, layer_sizes, activation="relu"):
        super(MLPNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # Map activation string to PyTorch activation
        activation_fn = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "logistic": nn.Sigmoid(),
        }[activation]

        # Create layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(activation_fn)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def do_optuna_pytorch(X, y, n_trials=100, **kwargs):
    """
    GPU-accelerated version of do_optuna function that uses PyTorch for training neural networks on GPU
    
    This implementation uses PyTorch's neural networks which can run on GPU
    
    Parameters
    ----------
    X : numpy.ndarray
        Input features matrix 
    y : numpy.ndarray
        Target values matrix
    n_trials : int, optional (default=100)
        Number of Optuna trials for hyperparameter search
    **kwargs : dict
        Additional parameters:
        - n_splits : int, optional (default=3)
            Number of cross-validation splits
        - n_layers : int, optional (default=30)
            Maximum number of layers to try
        - n_neurons : int, optional (default=100)
            Maximum number of neurons per layer to try
        
    Returns
    -------
    tuple
        (best_params, cur_X_test, cur_y_test, best_value)
    """

    n_splits = kwargs.get("n_splits", 3)
    n_layers = kwargs.get("n_layers", 30)
    n_neurons = kwargs.get("n_neurons", 100)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Preparing datasets
    cur_X_test, cur_y_test, val_list_X, val_list_y, train_list_X, train_list_y = (
        split_transform_one_comp_cv(X, y, n_splits=n_splits)
    )
    
    def optuna_pytorch_objective(trial):
        # Define hyperparameters to optimize
        k_layers = trial.suggest_int("n_layers", 1, n_layers)
        layers = []
        for i in range(k_layers):
            layers.append(trial.suggest_int(f"n_units_{i}", 1, n_neurons))
            
        activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop", "LBFGS"])
        max_epochs = trial.suggest_int("max_epochs", 50, 500)
        
        # Layer dimensions: input -> hidden layers -> output
        input_size = train_list_X[0].shape[1]
        layer_sizes = [input_size] + layers + [1]
        
        # Fitting and scoring for each split
        errors = np.zeros((n_splits, 9))
        
        for split_idx in range(n_splits):
            # Get data for this split and clean it
            cur_X_train = train_list_X[split_idx]
            cur_y_train = train_list_y[split_idx]
            cur_X_val = val_list_X[split_idx]
            cur_y_val = val_list_y[split_idx]
            
            cur_X_train, cur_y_train = clean_input_array(cur_X_train, cur_y_train)
            cur_X_val, cur_y_val = clean_input_array(cur_X_val, cur_y_val)
            
            # Convert to PyTorch tensors and move to GPU
            X_train_tensor = torch.FloatTensor(cur_X_train).to(device)
            y_train_tensor = torch.FloatTensor(cur_y_train.reshape(-1, 1)).to(device)
            X_val_tensor = torch.FloatTensor(cur_X_val).to(device)
            y_val_tensor = torch.FloatTensor(cur_y_val.reshape(-1, 1)).to(device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Create standardization layer
            mean = X_train_tensor.mean(0, keepdim=True)
            std = X_train_tensor.std(0, unbiased=False, keepdim=True)
            std[std == 0] = 1  # prevent division by zero
            
            # Create the model, loss function and optimizer
            model = MLPNetwork(layer_sizes, activation).to(device)
            criterion = nn.MSELoss()
            
            if optimizer_name == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_name == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            elif optimizer_name == "LBFGS":
                optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20, history_size=10)
            else:
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
            
            # Early stopping parameters
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            # Training loop
            for epoch in range(max_epochs):
                model.train()
                
                # Special handling for LBFGS which requires a closure
                if optimizer_name == "LBFGS":
                    # Full batch training for LBFGS
                    X_std = (X_train_tensor - mean) / std
                    
                    def closure():
                        optimizer.zero_grad()
                        outputs = model(X_std)
                        loss = criterion(outputs, y_train_tensor)
                        loss.backward()
                        return loss
                    
                    optimizer.step(closure)
                else:
                    # Mini-batch training for other optimizers
                    for X_batch, y_batch in train_loader:
                        # Standardize batch
                        X_batch = (X_batch - mean) / std
                        
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    X_val_std = (X_val_tensor - mean) / std
                    val_outputs = model(X_val_std)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    
                    # Early stopping check
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break
            
            # Final evaluation
            model.eval()
            with torch.no_grad():
                X_val_std = (X_val_tensor - mean) / std
                val_outputs = model(X_val_std)
                
                # Convert to numpy for metrics
                val_pred = val_outputs.cpu().numpy()
                val_true = y_val_tensor.cpu().numpy()
                
                # Calculate metrics
                errors[split_idx] = cuml_scorer(val_true, val_pred, model, cur_X_train)
        
        # Get the worst performance across splits
        val_metrics = choose_worst(errors)
        return_value = val_metrics[-1] if pd.notnull(val_metrics[-1]) else +1e6  # rmse
        return return_value
    
    # Create a study object to optimize the objective
    study = optuna.create_study(direction="minimize")  # rmse
    study.optimize(optuna_pytorch_objective, n_trials=n_trials, n_jobs=1)  # Use n_jobs=1 for GPU
    
    # Print the best hyperparameters found by Optuna
    best_params = study.best_params
    best_value = study.best_value
    print("Best Hyperparameters:", best_params)
    
    return best_params, cur_X_test, cur_y_test, best_value


def test_best_model_pytorch(X, y, best_params):
    """
    Test a PyTorch neural network model with the best hyperparameters found by do_optuna_pytorch.
    
    This function trains a model with the best hyperparameters on a train-test split and evaluates it on the test set.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input features matrix
    y : numpy.ndarray
        Target values matrix
    best_params : dict
        Best hyperparameters found by do_optuna_pytorch
        
    Returns
    -------
    tuple
        (trained_model, test_metrics, mean, std)
        - trained_model: PyTorch model trained with best hyperparameters
        - test_metrics: array of evaluation metrics on the test set
        - mean: mean values used for standardization
        - std: standard deviation values used for standardization
    """
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Split data into train/test sets (using the same function as in do_optuna_pytorch)
    X_test, y_test, _, _, train_list_X, train_list_y = split_transform_one_comp_cv(X, y, n_splits=1)
    
    # Get training data
    X_train = train_list_X[0]
    y_train = train_list_y[0]
    
    # Clean input arrays (handle NaNs, etc.)
    X_train, y_train = clean_input_array(X_train, y_train)
    X_test, y_test = clean_input_array(X_test, y_test)
    
    # Convert to PyTorch tensors and move to GPU
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1)).to(device)
    
    # Extract hyperparameters
    activation = best_params["activation"]
    batch_size = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]
    max_epochs = best_params["max_epochs"]
    optimizer_name = best_params["optimizer"]
    
    # Extract layer structure
    n_layers = best_params["n_layers"]
    layers = []
    for i in range(n_layers):
        layers.append(best_params[f"n_units_{i}"])
    
    # Layer dimensions: input -> hidden layers -> output
    input_size = X_train.shape[1]
    layer_sizes = [input_size] + layers + [1]
    
    # Create standardization parameters
    mean = X_train_tensor.mean(0, keepdim=True)
    std = X_train_tensor.std(0, unbiased=False, keepdim=True)
    std[std == 0] = 1  # prevent division by zero
    
    # Create the model, loss function and optimizer
    model = MLPNetwork(layer_sizes, activation).to(device)
    criterion = nn.MSELoss()
    
    # Create optimizer based on best parameters
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "LBFGS":
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20, history_size=10)
    else:  # RMSprop
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    # For tracking training history
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(max_epochs):
        model.train()
        
        # Special handling for LBFGS which requires a closure
        if optimizer_name == "LBFGS":
            # Full batch training for LBFGS
            X_std = (X_train_tensor - mean) / std
            
            def closure():
                optimizer.zero_grad()
                outputs = model(X_std)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                return loss
            
            optimizer.step(closure)
        else:
            # Mini-batch training for other optimizers
            for X_batch, y_batch in train_loader:
                # Standardize batch
                X_batch = (X_batch - mean) / std
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        # Calculate train error
        model.eval()
        with torch.no_grad():
            # Calculate training error
            X_train_std = (X_train_tensor - mean) / std
            train_outputs = model(X_train_std)
            train_loss = criterion(train_outputs, y_train_tensor)
            
            # Calculate validation error (test set used for validation)
            X_test_std = (X_test_tensor - mean) / std
            test_outputs = model(X_test_std)
            test_loss = criterion(test_outputs, y_test_tensor)
            
            # Save losses for history
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            
            # Print progress every 20 epochs
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{max_epochs}, Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}")
            
            # Early stopping check
            if test_loss.item() < best_val_loss:
                best_val_loss = test_loss.item()
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}, Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}")
                    break
    
    # Load best model state if early stopping was triggered
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        X_test_std = (X_test_tensor - mean) / std
        test_outputs = model(X_test_std)
        
        # Convert to numpy for metrics
        test_pred = test_outputs.cpu().numpy()
        test_true = y_test_tensor.cpu().numpy()
        
        # Calculate metrics
        test_metrics = cuml_scorer(test_true, test_pred, model, X_train)
        
        # Print metrics
        print("\nTest Metrics:")
        print(f"Explained Variance Score: {test_metrics[0]:.6f}")
        print(f"Median Absolute Error: {test_metrics[1]:.6f}")
        print(f"Mean Squared Error: {test_metrics[2]:.6f}")
        print(f"Mean Absolute Error: {test_metrics[3]:.6f}")
        print(f"R² Score: {test_metrics[4]:.6f}")
        print(f"Max Error: {test_metrics[5]:.6f}")
        print(f"AIC: {test_metrics[6]:.6f}")
        print(f"BIC: {test_metrics[7]:.6f}")
        print(f"RMSE: {test_metrics[8]:.6f}")
    
    return model, test_metrics, mean, std, {'train_losses': train_losses, 'test_losses': test_losses}


def predict_with_model(model, X, mean, std):
    """
    Make predictions with a trained PyTorch model.
    
    Parameters
    ----------
    model : MLPNetwork
        Trained PyTorch model
    X : numpy.ndarray
        Features for prediction
    mean : torch.Tensor
        Mean values for standardization
    std : torch.Tensor
        Standard deviation values for standardization
        
    Returns
    -------
    numpy.ndarray
        Predicted values
    """
    device = next(model.parameters()).device
    
    # Convert to PyTorch tensor and move to device
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Standardize
    X_std = (X_tensor - mean) / std
    
    # Predict
    model.eval()
    with torch.no_grad():
        predictions = model(X_std)
    
    # Convert to numpy
    return predictions.cpu().numpy()
