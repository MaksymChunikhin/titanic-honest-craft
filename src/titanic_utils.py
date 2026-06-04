"""
titanic_utils.py — общий код проекта (загрузка, фичи, препроцессинг, сохранение).
Shared project code (loading, features, preprocessing, persistence).

Зачем модуль: не дублировать код в ноутбуках + честная воспроизводимость.
Why a module: avoid copy-paste across notebooks + reproducibility.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# --- Пути и константы / Paths & constants -----------------------------------
ROOT = Path(__file__).resolve().parents[1]          # корень проекта / project root
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OPTUNA_DIR = ROOT / "optuna"
REPORTS_DIR = ROOT / "reports"
for _d in (DATA_DIR, MODELS_DIR, OPTUNA_DIR, REPORTS_DIR):
    _d.mkdir(exist_ok=True)

RANDOM_STATE = 42                                   # фикс для воспроизводимости
RAW_CACHE = DATA_DIR / "titanic_raw.parquet"        # локальный кэш датасета

# Колонки-утечки: известны только ПОСЛЕ катастрофы -> выбрасываем.
# Leakage columns: known only AFTER the event -> drop for honesty.
LEAKS = ["boat", "body", "home.dest"]

# Сводим редкие титулы в крупные группы / collapse rare titles into buckets.
TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Ms": "Miss", "Mlle": "Miss", "Mme": "Mrs",
    "Dr": "Officer", "Rev": "Officer", "Col": "Officer", "Major": "Officer",
    "Capt": "Officer", "Don": "Royalty", "Dona": "Royalty", "Lady": "Royalty",
    "Sir": "Royalty", "the Countess": "Royalty", "Jonkheer": "Royalty",
}


# --- Загрузка данных / Data loading -----------------------------------------
def load_data(use_cache: bool = True) -> pd.DataFrame:
    """Грузим Титаник как в задании; кэшируем локально, чтобы не ходить в сеть.
    Load Titanic exactly as in the task; cache locally to avoid network calls."""
    if use_cache and RAW_CACHE.exists():
        return pd.read_parquet(RAW_CACHE)
    from sklearn.datasets import fetch_openml
    df = fetch_openml("titanic", version=1, as_frame=True).frame
    df.to_parquet(RAW_CACHE)                         # сохраняем кэш / save cache
    return df


def get_target(df: pd.DataFrame) -> pd.Series:
    """survived приходит как category -> приводим к int 0/1."""
    return df["survived"].astype(int)


# --- Списки признаков / Feature lists ---------------------------------------
# A = классика, B = наши идеи (см. ноутбук 02). Финальный набор отбираем по важности.
NUM_FEATURES = [
    "age", "fare", "sibsp", "parch",                 # A: сырые / raw
    "FamilySize", "IsAlone",                          # A: семья / family
    "TicketGroupSize", "FarePerPerson",              # B: группа по билету / ticket group
    "HasCabin", "CabinCount",                         # B: каюта / cabin
    "AgeMissing", "IsChild", "IsMother", "NameLength",  # B/Г
]
CAT_FEATURES = [
    "pclass", "sex", "embarked", "Title", "Deck",    # A
    "SexPclass", "TicketPrefix",                      # B: сочетания / crosses
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


# --- Создание признаков / Feature engineering -------------------------------
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Признаки считаются по всему датафрейму, но БЕЗ обращения к таргету (утечки ответа нет).
    Групповые счётчики (TicketGroupSize / FarePerPerson / TicketPrefix) используют состав группы
    по билету — для строгой оценки на «новых» пассажирах см. GroupKFold в ноутбуке 03.
    Features use no target (no label leakage); group counters use ticket-group composition,
    so see the GroupKFold check in notebook 03 for the strict 'unseen passengers' evaluation."""
    d = data.copy()

    # A. Титул из имени / title from name -> сильнейший признак.
    raw_title = d["name"].str.extract(r",\s*([^.]+)\.")[0].str.strip()
    d["Title"] = raw_title.map(TITLE_MAP).fillna("Rare")

    # A. Размер семьи и «едет один» / family size & travelling alone.
    d["FamilySize"] = d["sibsp"] + d["parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)

    # B. Сколько людей по тому же билету / people sharing the same ticket.
    d["TicketGroupSize"] = d.groupby("ticket")["ticket"].transform("count")
    # B. Цена НА ЧЕЛОВЕКА: fare — за весь билет / fare is per ticket, not per person.
    d["FarePerPerson"] = d["fare"] / d["TicketGroupSize"]

    # B. Каюта / cabin: палуба, есть ли каюта, сколько кают.
    d["Deck"] = d["cabin"].str[0].fillna("U")
    d["HasCabin"] = d["cabin"].notna().astype(int)
    d["CabinCount"] = d["cabin"].str.split().str.len().fillna(0).astype(int)

    # B. Приставка билета / ticket prefix (PC, STON/O ...) -> тип билета.
    prefix = d["ticket"].astype(str).str.extract(r"^([A-Za-z./]+)")[0]
    prefix = prefix.str.replace(r"[^A-Za-z]", "", regex=True).str.upper()
    prefix = prefix.replace("", np.nan).fillna("NONE")
    rare = prefix.value_counts()[lambda s: s < 10].index     # редкие -> RARE
    d["TicketPrefix"] = prefix.where(~prefix.isin(rare), "RARE")

    # B. Сочетание пол×класс / sex×class — знаменитый сюжет Титаника.
    d["SexPclass"] = d["sex"].astype(str) + "_" + d["pclass"].astype(str)

    # B/Г. Возрастные флаги / age flags (NaN -> считаем «не ребёнок»).
    d["AgeMissing"] = d["age"].isna().astype(int)
    d["IsChild"] = (d["age"] < 16).astype(int)   # NaN < 16 -> False -> 0
    d["IsMother"] = (
        (d["sex"].astype(str) == "female")
        & (d["parch"] > 0)
        & (d["age"].fillna(0) > 18)
        & (d["Title"] == "Mrs")
    ).astype(int)

    # Г. Длина имени / name length — скорее шум, проверим и, вероятно, выкинем.
    d["NameLength"] = d["name"].str.len().astype(int)
    return d


# --- Препроцессинг / Preprocessing pipeline ---------------------------------
def build_preprocessor(num_features=None, cat_features=None):
    """Импьютинг + OneHot, фитятся ТОЛЬКО на train (через Pipeline).
    Imputation + OneHot, fit on train only (via Pipeline) -> leak-free."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    num_features = NUM_FEATURES if num_features is None else num_features
    cat_features = CAT_FEATURES if cat_features is None else cat_features

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])


# --- Сохранение/загрузка моделей / Model persistence ------------------------
def save_model(obj, name: str) -> Path:
    """Сохранить модель/объект в models/<name>.joblib."""
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(obj, path)
    return path


def load_model(name: str):
    """Загрузить, если есть, иначе None — для паттерна «есть сохранённая → берём».
    Load if exists else None — for the 'use cached model if present' pattern."""
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path) if path.exists() else None


def model_exists(name: str) -> bool:
    return (MODELS_DIR / f"{name}.joblib").exists()
