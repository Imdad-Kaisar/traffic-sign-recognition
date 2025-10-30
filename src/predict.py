import numpy as np

def top_k_predictions(probs, class_names, k=5):
    p = probs[0]
    idxs = np.argsort(p)[::-1][:k]
    return [(class_names[i], float(p[i])) for i in idxs]
