# Titanic Survival Prediction (Random Forest)

Учебный ML-проект на классическом датасете Titanic (OpenML). По характеристикам пассажира предсказывается вероятность выживания. Основной акцент сделан на построении воспроизводимого ML pipeline: feature engineering, кросс-валидация, сравнение моделей, подбор гиперпараметров и интерпретация результатов.

## Задача

Предсказать признак `survived`.

Условия задания:

* финальная модель — `RandomForestClassifier`;
* train/test split 80/20 (`random_state=42`);
* способ обработки пропусков и кодирования категориальных признаков выбирается самостоятельно.

---

## Данные

```python
from sklearn.datasets import fetch_openml

df = fetch_openml("titanic", version=1, as_frame=True).frame
```

Размер датасета:

* 1309 объектов;
* 14 исходных признаков.

Из датасета были исключены признаки, содержащие утечку информации:

* `boat`
* `body`

Также после извлечения признаков были удалены сырые текстовые поля:

* `name`
* `ticket`
* `cabin`

---

## Feature Engineering

На основе исходных признаков было создано около 17 дополнительных признаков.

### Семья

* `FamilySize`
* `IsAlone`
* `FamilyGroup`
* `FarePerPerson`
* `FemaleWithChildren`

### Имя пассажира

* `Title`

### Билет

* `TicketCount`
* `SharedTicket`
* `TicketPrefix`

### Каюта

* `HasCabin`
* `CabinLetter`

### Возраст

* `AgeGroup`
* `Child`

### Взаимодействия признаков

* `SexPclass`
* `TitlePclass`
* `ChildPclass`

Итоговый набор признаков содержит 23 колонки.

---

## Препроцессинг

Вся подготовка данных реализована через `sklearn.Pipeline`.

### Числовые признаки

* медианное заполнение пропусков.

### Категориальные признаки

* заполнение наиболее частым значением;
* `OneHotEncoder(handle_unknown="ignore")`.

---

## Сравнение моделей

Были протестированы несколько алгоритмов:

* Logistic Regression
* Random Forest
* HistGradientBoosting
* CatBoost
* LightGBM
* XGBoost
* ExtraTrees

После сравнения и подбора гиперпараметров были получены следующие результаты:

| Модель                | CV accuracy | Test accuracy |
| --------------------- | ----------: | ------------: |
| HistGradientBoosting  |  **0.8176** |        0.8321 |
| RandomForest (manual) |      0.8137 |    **0.8511** |
| RandomForest (Optuna) |      0.8137 |        0.8435 |
| CatBoost              |      0.8070 |    **0.8511** |

Несмотря на то, что HistGradientBoosting показал лучший результат на кросс-валидации, на отложенной тестовой выборке лучшие результаты продемонстрировали RandomForest и CatBoost.

---

## Выбор финальной модели

Параметры RandomForest были предварительно подобраны вручную на основе серии экспериментов и общих рекомендаций по настройке ансамблей.

Дополнительно был проведён автоматический подбор гиперпараметров с помощью Optuna. Полученные параметры дали очень близкий результат, однако не улучшили качество модели на тестовой выборке.

Поскольку по условию задания финальной моделью должен быть RandomForest, был выбран вариант с ручной настройкой:

```python
RandomForestClassifier(
    n_estimators=1000,
    max_depth=8,
    min_samples_leaf=3,
    min_samples_split=10,
    max_features="sqrt",
    random_state=42
)
```

### Финальные метрики

* Test Accuracy: **0.8511**
* 5-fold CV Accuracy: **0.814 ± 0.023**

---

## Интерпретация модели

Для анализа модели были использованы:

* Feature Importance;
* SHAP;
* Confusion Matrix;
* Classification Report;
* ROC-AUC;
* анализ ошибок модели.

Наиболее важными признаками оказались:

* пол пассажира;
* титул (`Title`);
* класс билета (`Pclass`);
* стоимость билета (`Fare`);
* размер семьи (`FamilySize`);
* наличие информации о каюте (`HasCabin`).

---

## Структура проекта

```text
notebooks/
│
├── 01_EDA.ipynb
├── 02_Feature_Engineering.ipynb
├── 03_Model_Comparison.ipynb
├── 04_Model_Tuning.ipynb
└── 05_Model_Interpretation.ipynb

data/
│
└── titanic_fe.parquet

models/
│
├── best_model.pkl
├── study_rf.pkl
├── study_hgb.pkl
└── study_cat.pkl

requirements.txt
README.md
```

---

## Используемый стек

* Python 3.13
* Pandas
* NumPy
* Scikit-learn
* Optuna
* SHAP
* CatBoost
* LightGBM
* XGBoost
* Matplotlib
* Seaborn

---

## Итог

Проект охватывает полный цикл решения задачи машинного обучения:

> **EDA → Feature Engineering → Pipeline → Model Comparison → Hyperparameter Tuning → Model Interpretation**
