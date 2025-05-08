"""
This module provides enhanced visualization for gradient boosting learning curves.
It adds validation metrics to the standard train and test metrics visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def plot_learning_curve_with_validation(model, cur_X_train, cur_y_train, cur_X_val, cur_y_val, cur_X_test, cur_y_test):
    """
    Plot learning curves for gradient boosting model with training, validation, and test metrics.
    
    Parameters:
    -----------
    model : GradientBoostingRegressor
        Trained gradient boosting model
    cur_X_train : array-like
        Training features
    cur_y_train : array-like
        Training targets
    cur_X_val : array-like
        Validation features
    cur_y_val : array-like
        Validation targets
    cur_X_test : array-like
        Test features
    cur_y_test : array-like
        Test targets
    """
    # Извлечение ошибок обучения на каждой итерации
    train_scores = model.train_score_
    
    # Преобразование значений train_score в MSE
    # В GradientBoostingRegressor train_score_ содержит значения функции потерь со знаком минус
    # для squared_error это половина MSE, поэтому умножаем на -2
    train_mse_scores = -2.0 * np.array(train_scores)
    
    # Вычисление RMSE для тренировочных данных
    train_rmse_scores = np.sqrt(train_mse_scores)
    
    # Вычисляем тестовую ошибку на каждой итерации
    test_mse_scores = []
    test_rmse_scores = []
    
    # Используем staged_predict для получения предсказаний на каждой итерации для тестовых данных
    for y_pred in model.staged_predict(cur_X_test):
        test_mse = mean_squared_error(cur_y_test, y_pred)
        test_mse_scores.append(test_mse)
        test_rmse_scores.append(np.sqrt(test_mse))
    
    # Вычисляем ошибку на валидационных данных на каждой итерации
    val_mse_scores = []
    val_rmse_scores = []
    
    # Используем staged_predict для получения предсказаний на каждой итерации для валидационных данных
    for y_pred in model.staged_predict(cur_X_val):
        val_mse = mean_squared_error(cur_y_val, y_pred)
        val_mse_scores.append(val_mse)
        val_rmse_scores.append(np.sqrt(val_mse))
    
    # Создание массива с номерами итераций
    iterations = np.arange(len(train_scores)) + 1
    
    # Находим оптимальное количество итераций (минимум на валидационной выборке)
    best_val_iteration = np.argmin(val_rmse_scores) + 1
    best_val_rmse = val_rmse_scores[best_val_iteration - 1]
    best_train_rmse = train_rmse_scores[best_val_iteration - 1]
    best_test_rmse = test_rmse_scores[best_val_iteration - 1]
    
    print(f"Оптимальное количество итераций: {best_val_iteration}")
    print(f"При оптимальном количестве итераций:")
    print(f"RMSE на обучающей выборке: {best_train_rmse:.4f}")
    print(f"RMSE на валидационной выборке: {best_val_rmse:.4f}")
    print(f"RMSE на тестовой выборке: {best_test_rmse:.4f}")
    
    # Построение графика
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, train_rmse_scores, 'b-', label='Train RMSE', alpha=0.7)
    plt.plot(iterations, val_rmse_scores, 'g-', label='Validation RMSE', alpha=0.7)
    plt.plot(iterations, test_rmse_scores, 'r-', label='Test RMSE', alpha=0.7)
    
    # Отмечаем точку с оптимальным количеством итераций
    plt.axvline(x=best_val_iteration, color='k', linestyle='--', alpha=0.5, label=f'Best iteration: {best_val_iteration}')
    plt.scatter([best_val_iteration], [best_val_rmse], color='g', s=100, edgecolor='k', zorder=5, label=f'Best val RMSE: {best_val_rmse:.4f}')
    plt.scatter([best_val_iteration], [best_test_rmse], color='r', s=100, edgecolor='k', zorder=5)
    plt.scatter([best_val_iteration], [best_train_rmse], color='b', s=100, edgecolor='k', zorder=5)
    
    plt.xlabel('Boosting Iterations')
    plt.ylabel('RMSE')
    plt.title('Gradient Boosting Learning Curve - RMSE Comparison')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Добавление аннотаций с значениями RMSE для оптимального числа итераций
    plt.annotate(f'Train: {best_train_rmse:.4f}', 
                 xy=(best_val_iteration, best_train_rmse), 
                 xytext=(best_val_iteration + 10, best_train_rmse),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7))
    
    plt.annotate(f'Val: {best_val_rmse:.4f}', 
                 xy=(best_val_iteration, best_val_rmse), 
                 xytext=(best_val_iteration + 10, best_val_rmse),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7))
    
    plt.annotate(f'Test: {best_test_rmse:.4f}', 
                 xy=(best_val_iteration, best_test_rmse), 
                 xytext=(best_val_iteration + 10, best_test_rmse),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, alpha=0.7))
    
    # Ограничиваем ось Y для лучшей видимости
    plt.ylim(0.8 * min(min(train_rmse_scores), min(val_rmse_scores), min(test_rmse_scores)),
             1.2 * max(test_rmse_scores[:best_val_iteration*2]))
    
    plt.show()
    
    # Создаем еще одну визуализацию с фокусом на области оптимального числа итераций
    plt.figure(figsize=(12, 7))
    window_start = max(0, best_val_iteration - 50)
    window_end = min(len(iterations), best_val_iteration + 50)
    window_iterations = iterations[window_start:window_end]
    window_train = train_rmse_scores[window_start:window_end]
    window_val = val_rmse_scores[window_start:window_end]
    window_test = test_rmse_scores[window_start:window_end]
    
    plt.plot(window_iterations, window_train, 'b-', label='Train RMSE', alpha=0.7)
    plt.plot(window_iterations, window_val, 'g-', label='Validation RMSE', alpha=0.7)
    plt.plot(window_iterations, window_test, 'r-', label='Test RMSE', alpha=0.7)
    
    # Отмечаем точку с оптимальным количеством итераций
    plt.axvline(x=best_val_iteration, color='k', linestyle='--', alpha=0.5, label=f'Best iteration: {best_val_iteration}')
    plt.scatter([best_val_iteration], [best_val_rmse], color='g', s=100, edgecolor='k', zorder=5)
    plt.scatter([best_val_iteration], [best_test_rmse], color='r', s=100, edgecolor='k', zorder=5)
    plt.scatter([best_val_iteration], [best_train_rmse], color='b', s=100, edgecolor='k', zorder=5)
    
    plt.xlabel('Boosting Iterations')
    plt.ylabel('RMSE')
    plt.title('Zoomed Gradient Boosting Learning Curve - Near Optimal Iterations')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Return metrics for further analysis
    return {
        'best_iteration': best_val_iteration,
        'best_val_rmse': best_val_rmse,
        'best_test_rmse': best_test_rmse,
        'best_train_rmse': best_train_rmse
    }
