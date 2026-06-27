import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

from pipeline.helper import force_utf8_stdout, tqdm, normalize_name

force_utf8_stdout()

# Anchor default I/O dirs to the project root (parent of pipeline/) so outputs always
# land in the project's real structure, regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_split(data_dir, split, max_rows=None, seed=42):
    df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    names = [normalize_name(n) for n in df["name"].astype(str).tolist()]
    y = df["label"].to_numpy()
    return names, y


def evaluate(model, X, y, name):
    pred = model.predict(X)
    macro = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    weighted = f1_score(y, pred, average="weighted")
    print(f"   [{name}] macroF1={macro:.4f}  acc={acc:.4f}  weightedF1={weighted:.4f}")
    return {"macro_f1": macro, "accuracy": acc, "weighted_f1": weighted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data"))
    ap.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "models"))
    ap.add_argument("--max_train", type=int, default=400_000,
                    help="subsample train for CPU baselines")
    ap.add_argument("--max_eval", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("📥 Loading splits ...")
    Xtr_raw, ytr = load_split(args.data_dir, "train", args.max_train, args.seed)
    Xva_raw, yva = load_split(args.data_dir, "val", args.max_eval, args.seed)
    Xte_raw, yte = load_split(args.data_dir, "test", args.max_eval, args.seed)
    print(f"   train={len(ytr):,}  val={len(yva):,}  test={len(yte):,}")

    print("🔤 Fitting char n-gram TF-IDF (char_wb, 2-4) ...")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          max_features=50_000, sublinear_tf=True)
    Xtr = vec.fit_transform(tqdm(Xtr_raw, desc="tfidf-fit"))
    Xva = vec.transform(Xva_raw)
    Xte = vec.transform(Xte_raw)

    models = {
        "linsvm": LinearSVC(C=1.0, dual=False, class_weight="balanced"),
        "logreg": LogisticRegression(C=5.0, solver="saga", multi_class="multinomial",
                                     class_weight="balanced", max_iter=200, n_jobs=-1),
        "sgd": SGDClassifier(loss="modified_huber", alpha=1e-5,
                             class_weight="balanced", max_iter=30, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        print(f"\n🤖 Training {name} ...")
        t0 = time.time()
        model.fit(Xtr, ytr)
        dt = time.time() - t0
        print(f"   trained in {dt:.1f}s")
        results[name] = {
            "val": evaluate(model, Xva, yva, f"{name}/val"),
            "test": evaluate(model, Xte, yte, f"{name}/test"),
            "train_seconds": round(dt, 1),
        }
        with open(os.path.join(args.out_dir, f"baseline_{name}.pkl"), "wb") as f:
            pickle.dump(model, f)

    with open(os.path.join(args.out_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)
    with open(os.path.join(args.out_dir, "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    best = max(results, key=lambda k: results[k]["val"]["macro_f1"])
    print(f"\n✅ Baselines done. Best val macro F1: {best} "
          f"({results[best]['val']['macro_f1']:.4f}). Saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
