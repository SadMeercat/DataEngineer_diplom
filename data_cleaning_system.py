import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Шаг 1: Загрузка данных
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Шаг 2: Очистка данных с использованием Isolation Forest
def clean_data(data, feature_col):
    # Инициализация модели Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)  # 10% данных считается выбросами
    # Обучение модели на выбранном признаке
    data['anomaly'] = iso_forest.fit_predict(data[[feature_col]])

    # Отделение нормальных и аномальных данных
    normal_data = data[data['anomaly'] == 1]
    anomaly_data = data[data['anomaly'] == -1]

    return normal_data, anomaly_data

# Шаг 3: Визуализация результатов
def plot_results(normal_data, anomaly_data, feature_col):
    plt.figure(figsize=(10, 6))
    plt.scatter(normal_data.index, normal_data[feature_col], color='blue', label='Normal data', s=10)
    plt.scatter(anomaly_data.index, anomaly_data[feature_col], color='red', label='Anomaly data', s=10)
    plt.title('Data Cleaning with Isolation Forest')
    plt.xlabel('Index')
    plt.ylabel(feature_col)
    plt.legend()
    plt.show()

# Шаг 4: Основной процесс
def main():
    data = load_data('AB_NYC_2019.csv')
    feature_col = 'price'

    # Очистка данных
    normal_data, anomaly_data = clean_data(data, feature_col)

    # Визуализация результатов
    plot_results(normal_data, anomaly_data, feature_col)

    # Сохранение очищенных данных
    normal_data.to_csv('cleaned_data.csv', index=False)

if __name__ == "__main__":
    main()
