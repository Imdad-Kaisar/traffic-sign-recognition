import json, os, tensorflow as tf
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(__file__))
ASSETS = os.path.join(ROOT, "assets")

MODEL_PATH = os.path.join(ASSETS, "best_model.h5")
CLASS_NAMES_PATH = os.path.join(ASSETS, "class_names.json")
METRICS_PATH = os.path.join(ASSETS, "metrics.json")

@lru_cache(maxsize=1)
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

@lru_cache(maxsize=1)
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@lru_cache(maxsize=1)
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
