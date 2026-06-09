# Titanic Survival Prediction (Random Forest)

Учебный ML-проект на классическом датасете Titanic (OpenML): по характеристикам пассажира предсказывается вероятность выживания. Проект демонстрирует полный воспроизводимый цикл решения задачи — от EDA до интерпретации модели — с акцентом на корректную методологию: препроцессинг внутри `Pipeline`, исключение утечек, честное сравнение моделей по кросс-валидации.

**Ключевой результат:** 5-fold CV accuracy **0.814 ± 0.023**, hold-out test accuracy **0.851** (RandomForest, финальная проверка на отложенной выборке).

> **EDA → Feature Engineering → Pipeline → Model Comparison → Hyperparameter Tuning → Interpretation**

---

## Задача

Предсказать признак `survived`.

Условия задания:

- финальная модель — `RandomForestClassifier`;
- train/test split 80/20 (`random_state=42`, стратификация по таргету);
- способ обработки пропусков и кодирования категориальных признаков выбирается самостоятельно.

---

## Данные

```python
from sklearn.datasets import fetch_openml

df = fetch_openml("titanic", version=1, as_frame=True).frame
```

1309 объектов, 14 исходных признаков.

Исключены признаки с утечкой целевой информации: `boat`, `body` (известны только после катастрофы). После извлечения признаков удалены сырые текстовые поля `name`, `ticket`, `cabin` и шумный `home.dest`.

---

## Feature Engineering

На основе исходных признаков создано ~17 новых; итоговый набор — 23 колонки.

| Группа | Признаки |
| --- | --- |
| Семья | `FamilySize`, `IsAlone`, `FamilyGroup`, `FarePerPerson`, `FemaleNotAlone` |
| Имя | `Title` (с группировкой редких титулов) |
| Билет | `TicketCount`, `SharedTicket`, `TicketPrefix` |
| Каюта | `HasCabin`, `CabinLetter` |
| Возраст | `AgeGroup`, `Child` |
| Взаимодействия | `SexPclass`, `TitlePclass`, `ChildPclass` |

Baseline на сырых признаках без FE построен для сравнения — feature engineering даёт заметный прирост над ним.

---

## Препроцессинг

Вся подготовка данных — внутри `sklearn.Pipeline` + `ColumnTransformer`, поэтому импутация и кодирование выполняются только на train-фолдах внутри кросс-валидации, без утечки статистик из валидационных данных.

- Числовые признаки: медианное заполнение пропусков.
- Категориальные: заполнение модой + `OneHotEncoder(handle_unknown="ignore")`.

---

## Сравнение моделей

На одинаковом сплите и одинаковом препроцессинге протестированы: Logistic Regression, Random Forest, ExtraTrees, HistGradientBoosting, CatBoost, LightGBM, XGBoost.

Результаты после подбора гиперпараметров (основная метрика — CV):

| Модель | CV accuracy | Test accuracy |
| --- | --- | --- |
| HistGradientBoosting | **0.8176** | 0.8321 |
| RandomForest (manual) | 0.8137 | 0.8511 |
| RandomForest (Optuna) | 0.8137 | 0.8435 |
| CatBoost | 0.8070 | 0.8511 |

Разница между моделями укладывается в std кросс-валидации (±0.02) — на датасете такого размера они статистически неразличимы, а ранжирование по одиночному hold-out нестабильно (1 объект теста ≈ 0.4% accuracy). Поэтому выбор модели делается по CV, а test используется один раз — как финальная проверка.

---

## Финальная модель

По условию задания финальная модель — RandomForest. Гиперпараметры подбирались двумя способами: вручную (серия экспериментов по CV) и автоматически через Optuna (50 trials, TPE). Optuna дала результат, идентичный ручной настройке — потолок качества здесь определяется данными и признаками, а не гиперпараметрами.

```python
RandomForestClassifier(
    n_estimators=1000,
    max_depth=8,
    min_samples_leaf=3,
    min_samples_split=10,
    max_features="sqrt",
    random_state=42,
)
```

**Метрики:** 5-fold CV accuracy 0.814 ± 0.023, test accuracy 0.851.

---

## Интерпретация модели

Feature Importance, SHAP, Confusion Matrix, Classification Report, ROC-AUC и анализ ошибок.

Наиболее важные признаки: пол, `Title`, `Pclass`, `Fare`, `FamilySize`, `HasCabin` — что согласуется с историческим контекстом («женщины и дети, первый класс»).

---

## Структура проекта

```
notebooks/
├── 01_EDA.ipynb                    # обзор данных, пропуски, таргет
├── 02_Feature_Engineering.ipynb    # FE, пайплайн, baseline vs FE
├── 03_Model_Comparison.ipynb       # CV-сравнение 7 моделей
├── 04_Model_Tuning.ipynb           # Optuna + кэширование studies
└── 05_Model_Interpretation.ipynb   # SHAP, ошибки, метрики

data/        # генерируется ноутбуком 02 (titanic_fe.parquet)
models/      # генерируется ноутбуком 04 (модели и optuna studies)

requirements.txt
README.md
```

Папки `data/` и `models/` не хранятся в репозитории — все артефакты воспроизводятся запуском ноутбуков по порядку. Optuna studies кэшируются на диск, поэтому повторный запуск ноутбука 04 не перезапускает тюнинг.

## Запуск

```bash
pip install -r requirements.txt
jupyter lab
# выполнить ноутбуки 01 → 05 по порядку
```

Тестировалось на Python 3.13, версии зависимостей зафиксированы в `requirements.txt`.

---

## Стек

Python 3.13 · Pandas · NumPy · Scikit-learn · Optuna · SHAP · CatBoost · LightGBM · XGBoost · Matplotlib · Seaborn

---

## Выводы

- Воспроизводимый пайплайн с препроцессингом внутри CV и исключёнными утечками.
- Feature engineering важнее выбора алгоритма: семь моделей сошлись к одному уровню качества, а прирост дали именно новые признаки.
- Тюнинг гиперпараметров на маленьком датасете упирается в потолок данных — Optuna не улучшила ручную настройку.
- Оценивать модели на малых данных корректно по кросс-валидации; одиночный hold-out годится только для финальной проверки.
