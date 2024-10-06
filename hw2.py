import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем данные
red_wine = pd.read_csv('winequality-red.csv', sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')

# 1. Проверка на наличие пропущенных данных
print("Пропуски в данных красного вина:\n", red_wine.isnull().sum())
print("\nПропуски в данных белого вина:\n", white_wine.isnull().sum())

# 2. Преобразуем метки качества к бинарным классам (0 - плохое вино, 1 - хорошее вино)
threshold = 6
red_wine['good_wine'] = (red_wine['quality'] >= threshold).astype(int)
white_wine['good_wine'] = (white_wine['quality'] >= threshold).astype(int)

# 3. Выбросы по столбцу 'quality'
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

red_outliers = detect_outliers(red_wine, 'quality')
white_outliers = detect_outliers(white_wine, 'quality')

print(f"Количество выбросов по качеству в красном вине: {len(red_outliers)}")
print(f"Количество выбросов по качеству в белом вине: {len(white_outliers)}")

# Удаляем выбросы
red_wine_cleaned = red_wine.drop(red_outliers.index)
white_wine_cleaned = white_wine.drop(white_outliers.index)

# Построим график распределения по качеству
plt.figure(figsize=(10, 5))
sns.distplot(red_wine_cleaned['quality'], label='Red Wine Quality', kde=False, bins=10)
sns.distplot(white_wine_cleaned['quality'], label='White Wine Quality', kde=False, bins=10)
plt.legend()
plt.title("Распределение по качеству")
plt.show()

# Построим график распределения бинарных меток (баланс классов)
plt.figure(figsize=(10, 5))
sns.countplot(x='good_wine', data=red_wine_cleaned, label='Red Wine')
sns.countplot(x='good_wine', data=white_wine_cleaned, label='White Wine')
plt.title("Баланс бинарных классов")
plt.legend()
plt.show()

# Найдем медианы по каждому признаку
print("Медианы по признакам для красного вина:\n", red_wine_cleaned.median())
print("\nМедианы по признакам для белого вина:\n", white_wine_cleaned.median())

# Построим график "ящик с усами" по качеству
plt.figure(figsize=(10, 5))
sns.boxplot(data=red_wine_cleaned, x='quality')
plt.title("Ящик с усами по показателю качества (Красное вино)")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=white_wine_cleaned, x='quality')
plt.title("Ящик с усами по показателю качества (Белое вино)")
plt.show()

# Построим графики распределений значений каждого из 12 признаков
features = red_wine.columns[:-2]  # Все столбцы кроме 'quality' и 'good_wine'

for feature in features:
    plt.figure(figsize=(10, 5))
    sns.histplot(red_wine_cleaned[feature], kde=True, color='red', label='Red Wine')
    sns.histplot(white_wine_cleaned[feature], kde=True, color='green', label='White Wine')
    plt.title(f"Распределение признака {feature}")
    plt.legend()
    plt.show()

# Построим матрицу корреляции между признаками
plt.figure(figsize=(12, 8))
sns.heatmap(red_wine_cleaned.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Матрица корреляции признаков (Красное вино)")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(white_wine_cleaned.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Матрица корреляции признаков (Белое вино)")
plt.show()
