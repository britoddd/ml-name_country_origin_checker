"""
Loads the sklearn models (.pkl + vectorizer.pkl), and returns top-k (country, probability).
Preprocessing is imported from common.py, so it is identical to training by construction.

CLI:
  python 05_predict.py --name "Joko Widodo" --top_k 5
  python 05_predict.py --name "Tanaka" --model baseline_linsvm.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from pipeline.helper import force_utf8_stdout, normalize_name, name_to_indices

force_utf8_stdout()

# Anchor default I/O dirs to the project root (parent of pipeline/) so paths always
# resolve to the project's real structure, regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DISCLAIMER = ("⚠️  Statistical guess from name patterns only — NOT a statement of a "
              "person's actual nationality. Do not use for profiling or any "
              "high-stakes decision.")


class SklearnPredictor:
    def __init__(self, model_path: str, model_dir: str, label_map_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f:
            self.vec = pickle.load(f)
        with open(label_map_path, encoding="utf-8") as f:
            self.label_map = {int(k): v for k, v in json.load(f).items()}

    def predict(self, name: str, top_k: int = 5):
        X = self.vec.transform([normalize_name(name)])
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            classes = self.model.classes_
            idx = np.argsort(-probs)[:top_k]
            return [(self.label_map[int(classes[i])], float(probs[i])) for i in idx]
        # LinearSVC: use decision_function -> softmax-ish ranking
        scores = self.model.decision_function(X)[0]
        classes = self.model.classes_
        e = np.exp(scores - scores.max())
        probs = e / e.sum()
        idx = np.argsort(-probs)[:top_k]
        return [(self.label_map[int(classes[i])], float(probs[i])) for i in idx]


def load_country_names(data_dir: str):
    path = os.path.join(data_dir, "..", "archive", "country_codes.csv")
    alt = os.path.join(PROJECT_ROOT, "archive", "country_codes.csv")
    for p in (path, alt):
        if os.path.exists(p):
            import pandas as pd
            df = pd.read_csv(p)
            return dict(zip(df["country_code"].astype(str), df["country_name"].astype(str)))
    return {}


def main():
    ap = argparse.ArgumentParser(description="Predict nationality from a name.")
    ap.add_argument("--name")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--model")
    ap.add_argument("--model_dir", default=os.path.join(PROJECT_ROOT, "models"))
    ap.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data"))
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()
    
    if args.name:
        name = args.name
    else:
        name = input('Input name: ')
    
    norm = normalize_name(name)
    if not norm:
        print(json.dumps({"error": "name normalizes to empty (no a-z chars)"}))
        return

    if not args.model:
        while True:
            print('Choose a model (1-3):')
            print('1. Linear SVM')
            print('2. Logistic Regression')
            print('3. Stochastic Gradient Descent')
            model_choice = int(input('>> '))

            if model_choice == 1:
                model = 'baseline_linsvm.pkl'
                break
            elif model_choice == 2:
                model = 'baseline_logreg.pkl'
                break
            elif model_choice == 3:
                model = 'baseline_sgd.pkl'
                break
            else:
                continue
    else:
        model = args.model

    model_path = os.path.join(PROJECT_ROOT, "models", model)

    predictor = SklearnPredictor(
        model_path, args.model_dir, os.path.join(args.data_dir, "label_map.json")
    )

    preds = predictor.predict(name, args.top_k)
    cnames = load_country_names(args.data_dir)
    result = {
        "input": name,
        "normalized": norm,
        "predictions": [
            {"country_code": cc, "country": cnames.get(cc, cc), "probability": round(p, 4)}
            for cc, p in preds
        ],
        "disclaimer": DISCLAIMER,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"\n🔎 '{name}'  (normalized: '{norm}')")
    for i, pr in enumerate(result["predictions"], 1):
        bar = "█" * int(pr["probability"] * 30)
        print(f"  {i}. {pr['country_code']:<3} {pr['country']:<22} {pr['probability']*100:5.1f}%  {bar}")
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()