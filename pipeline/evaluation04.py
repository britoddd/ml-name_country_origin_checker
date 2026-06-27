import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from pipeline.helper import force_utf8_stdout, tqdm, normalize_name

force_utf8_stdout()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Anchor default I/O dirs to the project root (parent of pipeline/) so outputs always
# land in the project's real structure, regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINES = ["logreg", "linsvm", "sgd"]


def full_probs(model, X, num_classes):
    """Return an (n, num_classes) prob matrix aligned to label index 0..num_classes-1."""
    classes = model.classes_.astype(int)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
    else:  # LinearSVC -> softmax over decision_function margins
        s = model.decision_function(X)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = e / e.sum(axis=1, keepdims=True)
    out = np.zeros((X.shape[0], num_classes), dtype=np.float64)
    out[:, classes] = p
    return out


def lenient_topk(order, yte, names, name_country_map, code_to_label, k):
    hits = n_multi = 0
    for i in range(len(yte)):
        nm = names[i]
        true_cc = None  # resolved via label below
        allowed_codes = set(name_country_map.get(nm, []))
        if nm in name_country_map:
            n_multi += 1
        allowed_labels = {code_to_label[c] for c in allowed_codes if c in code_to_label}
        allowed_labels.add(int(yte[i]))  # the true label is always allowed
        if any(p in allowed_labels for p in order[i, :k]):
            hits += 1
    return hits / len(yte), n_multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data"))
    ap.add_argument("--model_dir", default=os.path.join(PROJECT_ROOT, "models"))
    ap.add_argument("--max_eval", type=int, default=0, help="0 = full test set")
    ap.add_argument("--top_n_confusion", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    plots_dir = os.path.join(args.model_dir, "plots", "baselines")
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "label_map.json"), encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    num_classes = len(label_map)
    code_to_label = {v: k for k, v in label_map.items()}
    with open(os.path.join(args.data_dir, "name_country_map.json"), encoding="utf-8") as f:
        name_country_map = json.load(f)

    df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    if args.max_eval and len(df) > args.max_eval:
        df = df.sample(n=args.max_eval, random_state=args.seed).reset_index(drop=True)
    names = df["name"].astype(str).tolist()
    yte = df["label"].to_numpy()
    Xraw = [normalize_name(n) for n in names]

    with open(os.path.join(args.model_dir, "vectorizer.pkl"), "rb") as f:
        vec = pickle.load(f)
    X = vec.transform(tqdm(Xraw, desc="tfidf-transform"))
    print(f"📥 test rows={len(yte):,}  classes={num_classes}")

    results = {}
    per_country_best = None
    preds_best = None
    best_name = None
    best_macro = -1.0

    for name in BASELINES:
        path = os.path.join(args.model_dir, f"baseline_{name}.pkl")
        if not os.path.exists(path):
            print(f"   (skip {name}: {path} not found)")
            continue
        with open(path, "rb") as f:
            model = pickle.load(f)
        probs = full_probs(model, X, num_classes)
        order = np.argsort(-probs, axis=1)
        preds = order[:, 0]

        macro = f1_score(yte, preds, average="macro", labels=list(range(num_classes)), zero_division=0)
        weighted = f1_score(yte, preds, average="weighted", zero_division=0)
        acc = accuracy_score(yte, preds)
        top3 = np.mean([yte[i] in order[i, :3] for i in range(len(yte))])
        top5 = np.mean([yte[i] in order[i, :5] for i in range(len(yte))])
        ranks = np.array([np.where(order[i] == yte[i])[0][0] + 1 for i in range(len(yte))])
        mrr = float(np.mean(1.0 / ranks))
        nll = float(-np.mean(np.log(probs[np.arange(len(yte)), yte] + 1e-12)))
        len1, n_multi = lenient_topk(order, yte, names, name_country_map, code_to_label, 1)
        len3, _ = lenient_topk(order, yte, names, name_country_map, code_to_label, 3)
        per_class = f1_score(yte, preds, average=None, labels=list(range(num_classes)), zero_division=0)

        results[name] = {
            "strict": {"macro_f1": float(macro), "accuracy": float(acc),
                       "weighted_f1": float(weighted), "top3": float(top3),
                       "top5": float(top5), "mrr": mrr, "nll": nll},
            "ambiguity_aware": {"set_lenient_top1": float(len1), "set_lenient_top3": float(len3),
                                "gap_top1_vs_strict_acc": float(len1 - acc)},
            "per_country_f1": {label_map[i]: float(per_class[i]) for i in range(num_classes)},
        }
        print(f"   [{name:6}] macroF1={macro:.4f}  acc={acc:.4f}  top3={top3:.4f}  "
              f"top5={top5:.4f}  MRR={mrr:.4f}  NLL={nll:.4f}  lenient@1={len1:.4f}")
        if macro > best_macro:
            best_macro, best_name = macro, name
            per_country_best = {label_map[i]: float(per_class[i]) for i in range(num_classes)}
            preds_best = preds

    with open(os.path.join(args.model_dir, "baseline_eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- comparison bar: macro F1 across baselines (+ deep if available) -----------
    bars = {n: results[n]["strict"]["macro_f1"] for n in results}
    deep_path = os.path.join(args.model_dir, "deep", "eval_metrics.json")
    if os.path.exists(deep_path):
        with open(deep_path, encoding="utf-8") as f:
            bars["deep"] = json.load(f)["strict"]["macro_f1"]
    items = sorted(bars.items(), key=lambda kv: kv[1])
    plt.figure(figsize=(7, 4))
    colors = ["seagreen" if k == "deep" else "steelblue" for k, _ in items]
    plt.barh([k for k, _ in items], [v for _, v in items], color=colors)
    for i, (_, v) in enumerate(items):
        plt.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    plt.xlabel("test macro F1"); plt.xlim(0, max(bars.values()) * 1.15)
    plt.title("Model comparison — test macro F1 (deep vs. baselines)")
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "model_comparison_macro_f1.png"), dpi=130)
    plt.close()

    # ---- best baseline: per-country F1 + confusion matrix --------------------------
    if per_country_best is not None:
        items2 = sorted(per_country_best.items(), key=lambda kv: kv[1])
        plt.figure(figsize=(10, max(8, len(items2) * 0.18)))
        plt.barh([k for k, _ in items2], [v for _, v in items2], color="steelblue")
        plt.axvline(best_macro, color="red", ls="--", label=f"macro F1={best_macro:.3f}")
        plt.xlabel("F1"); plt.title(f"Per-country F1 — best baseline ({best_name})"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"per_country_f1_{best_name}.png"), dpi=130)
        plt.close()

        counts = np.bincount(yte, minlength=num_classes)
        top_n = min(args.top_n_confusion, num_classes)
        top_classes = list(np.argsort(-counts)[:top_n])
        cm = confusion_matrix(yte, preds_best, labels=top_classes)
        cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        ticks = [label_map[c] for c in top_classes]
        plt.figure(figsize=(max(8, top_n * 0.5), max(7, top_n * 0.45)))
        sns.heatmap(cm_norm, xticklabels=ticks, yticklabels=ticks, cmap="viridis",
                    square=True, cbar_kws={"label": "row-normalized"})
        plt.xlabel("predicted"); plt.ylabel("true")
        plt.title(f"Confusion matrix — best baseline ({best_name}), top-{top_n} classes")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"confusion_matrix_{best_name}.png"), dpi=130)
        plt.close()

    print(f"\n✅ Baseline evaluation complete. Best baseline: {best_name} "
          f"(macroF1={best_macro:.4f}). Metrics -> {args.model_dir}/baseline_eval_metrics.json, "
          f"plots -> {plots_dir}/")


if __name__ == "__main__":
    main()
