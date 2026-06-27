import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from pipeline.helper.common import (
    force_utf8_stdout,
    tqdm,
    normalize_name,
    name_to_indices,
    MAX_LEN,
)

force_utf8_stdout()

# Anchor default I/O dirs to the project root (parent of pipeline/) so outputs always
# land in the project's real structure, regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Leading particles that can be stripped to make a realistic single-token / short variant.
PARTICLES = {
    "van", "von", "de", "del", "della", "di", "da", "dos", "das", "du", "le", "la",
    "mc", "mac", "al", "el", "bin", "ben", "abu", "ibn", "st", "san", "santa", "ter",
    "ten", "op", "och", "bar",
}


# --------------------------------------------------------------------------------------
# Step 1: load + accumulate per-country token counts (chunked, RAM-bounded)
# --------------------------------------------------------------------------------------
def accumulate_tokens(csv_path: str, name_col: str, chunksize: int, max_tokens: int):
    """Return {country: {normalized_token: count}}, pruned to top `max_tokens` per country."""
    per_country: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    reader = pd.read_csv(
        csv_path,
        usecols=[name_col, "country", "count"],
        dtype={name_col: "string", "country": "string", "count": "int64"},
        chunksize=chunksize,
    )
    bar = tqdm(desc=f"reading {os.path.basename(csv_path)}", unit="rows", unit_scale=True)
    for chunk in reader:
        # collapse within-chunk by raw token first to cut python normalize calls a bit
        names = chunk[name_col].fillna("").astype(str).tolist()
        countries = chunk["country"].fillna("").astype(str).tolist()
        counts = chunk["count"].tolist()
        for nm, cc, ct in zip(names, countries, counts):
            norm = normalize_name(nm)
            if not norm or len(norm) < 2 or not cc:
                continue
            per_country[cc][norm] += int(ct)
        bar.update(len(chunk))
    bar.close()

    # prune to top-N per country by count
    pruned: dict[str, dict[str, int]] = {}
    for cc, d in per_country.items():
        if len(d) > max_tokens:
            top = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:max_tokens]
            pruned[cc] = dict(top)
        else:
            pruned[cc] = dict(d)
    return pruned


def make_sampler(token_counts: dict[str, int]):
    """Return (tokens array, probability array) for count-weighted sampling."""
    tokens = np.array(list(token_counts.keys()), dtype=object)
    counts = np.array(list(token_counts.values()), dtype=np.float64)
    probs = counts / counts.sum()
    return tokens, probs


def generate_full_names(fore, sur, target, rng: np.random.Generator):
    """Sample `target` unique 'forename surname' strings weighted by token counts."""
    results: set[str] = set()
    if fore is None and sur is None:
        return results
    # single-token fallback if one side is missing
    if fore is None or sur is None:
        toks, probs = (sur if fore is None else fore)
        n_unique = len(toks)
        take = min(target, n_unique)
        # sample without replacement-ish by oversampling then dedup
        draws = rng.choice(toks, size=min(take * 4, n_unique * 4 + take), p=probs)
        for t in draws:
            results.add(str(t))
            if len(results) >= take:
                break
        return results

    f_tok, f_p = fore
    s_tok, s_p = sur
    max_unique = len(f_tok) * len(s_tok)
    take = min(target, max_unique)
    attempts = 0
    batch = max(take * 2, 4096)
    max_attempts = take * 20 + batch
    while len(results) < take and attempts < max_attempts:
        fi = rng.choice(f_tok, size=batch, p=f_p)
        si = rng.choice(s_tok, size=batch, p=s_p)
        for a, b in zip(fi, si):
            results.add(f"{a} {b}")
            if len(results) >= take:
                break
        attempts += batch
    return results


# --------------------------------------------------------------------------------------
# Step 5: augmentation transforms (train split only)
# --------------------------------------------------------------------------------------
def augment_variants(name: str):
    """Yield realistic variants of a full name. All already normalized (a-z + space)."""
    toks = name.split()
    if len(toks) >= 2:
        # first <-> last swap
        yield " ".join([toks[-1]] + toks[1:-1] + [toks[0]])
        # single-token forms
        yield toks[0]
        yield toks[-1]
        # drop middle (3+ tokens) -> first + last
        if len(toks) >= 3:
            yield f"{toks[0]} {toks[-1]}"
        # strip leading particle
        if toks[0] in PARTICLES and len(toks) >= 2:
            yield " ".join(toks[1:])
    else:
        # already a single token: nothing realistic to add
        return


def main():
    ap = argparse.ArgumentParser()
    
    # Main argument
    ap.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "archive"), help="dir with forenames/surnames/country_codes.csv")
    ap.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "data"))

    # Optional argument
    ap.add_argument("--max_per_country", type=int, default=50000,
                    help="synthetic full names generated per country (the key balancing knob)")
    ap.add_argument("--min_per_country", type=int, default=500,
                    help="drop countries that cannot produce this many unique names")
    ap.add_argument("--max_tokens", type=int, default=200000,
                    help="top-N tokens kept per country per table (RAM bound)")
    ap.add_argument("--aug_threshold", type=int, default=10000,
                    help="train-split size below which a country is augmented")
    ap.add_argument("--max_aug_factor", type=float, default=3.0)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- country codes -------------------------------------------------------------
    cc_path = os.path.join(args.data_dir, "country_codes.csv")
    cc_df = pd.read_csv(cc_path)
    valid_codes = set(cc_df["country_code"].astype(str))
    print(f"📋 {len(valid_codes)} country codes loaded.")

    # ---- step 1: accumulate token counts -------------------------------------------
    print("📥 Accumulating forename counts ...")
    fore_counts = accumulate_tokens(
        os.path.join(args.data_dir, "forenames.csv"), "forename", args.chunksize, args.max_tokens
    )
    print("📥 Accumulating surname counts ...")
    sur_counts = accumulate_tokens(
        os.path.join(args.data_dir, "surnames.csv"), "surname", args.chunksize, args.max_tokens
    )

    countries = sorted(c for c in valid_codes if c in fore_counts or c in sur_counts)

    # ---- step 2: generate synthetic full names per country -------------------------
    print(f"🧬 Generating up to {args.max_per_country:,} synthetic full names per country ...")
    country_names: dict[str, list[str]] = {}
    for cc in tqdm(countries, desc="countries"):
        fore = make_sampler(fore_counts[cc]) if fore_counts.get(cc) else None
        sur = make_sampler(sur_counts[cc]) if sur_counts.get(cc) else None
        names = generate_full_names(fore, sur, args.max_per_country, rng)
        if len(names) >= args.min_per_country:
            country_names[cc] = sorted(names)

    kept = sorted(country_names.keys())
    dropped = [c for c in countries if c not in country_names]
    print(f"✅ Kept {len(kept)} countries; dropped {len(dropped)} below min "
          f"({args.min_per_country}): {dropped}")

    # ---- label map -----------------------------------------------------------------
    label_map = {i: cc for i, cc in enumerate(kept)}
    code_to_label = {cc: i for i, cc in label_map.items()}

    # ---- step 3: stratified 70/15/15 split -----------------------------------------
    train_rows, val_rows, test_rows = [], [], []  # each row: (name, label, cc)
    for cc in tqdm(kept, desc="splitting"):
        names = country_names[cc][:]
        rng.shuffle(names)
        n = len(names)
        n_tr = int(n * 0.70)
        n_va = int(n * 0.15)
        lbl = code_to_label[cc]
        for nm in names[:n_tr]:
            train_rows.append((nm, lbl, cc))
        for nm in names[n_tr:n_tr + n_va]:
            val_rows.append((nm, lbl, cc))
        for nm in names[n_tr + n_va:]:
            test_rows.append((nm, lbl, cc))

    # ---- step 4: name_country_map from PRE-AUGMENTATION names -----------------------
    print("🗺️  Building name->country membership map (pre-augmentation) ...")
    name_to_countries: dict[str, set] = defaultdict(set)
    for nm, _lbl, cc in (train_rows + val_rows + test_rows):
        name_to_countries[nm].add(cc)
    name_country_map = {nm: sorted(cs) for nm, cs in name_to_countries.items() if len(cs) > 1}
    print(f"   {len(name_country_map):,} multi-country names "
          f"({100*len(name_country_map)/max(1,len(name_to_countries)):.2f}% of unique names).")

    # ---- step 5: augment TRAIN only (small countries, <=max_aug_factor) -------------
    print("🔧 Augmenting train split (small countries only) ...")
    # pairs present in val/test must never be produced by augmentation (leak guard)
    valtest_pairs = {(nm, cc) for nm, _l, cc in (val_rows + test_rows)}
    train_by_cc: dict[str, set] = defaultdict(set)
    for nm, _l, cc in train_rows:
        train_by_cc[cc].add(nm)

    aug_rows = []
    for cc in tqdm(kept, desc="augmenting"):
        cur = train_by_cc[cc]
        if len(cur) >= args.aug_threshold:
            continue
        lbl = code_to_label[cc]
        cap = int(len(cur) * args.max_aug_factor)  # total train size ceiling (incl. originals)
        budget = cap - len(cur)
        if budget <= 0:
            continue
        new_for_cc: set[str] = set()
        base = list(cur)
        rng.shuffle(base)
        for nm in base:
            if len(new_for_cc) >= budget:
                break
            for var in augment_variants(nm):
                if not var or len(var) < 2:
                    continue
                if var in cur or var in new_for_cc:
                    continue
                if (var, cc) in valtest_pairs:
                    continue
                new_for_cc.add(var)
                if len(new_for_cc) >= budget:
                    break
        for var in new_for_cc:
            aug_rows.append((var, lbl, cc))
    print(f"   +{len(aug_rows):,} augmented train rows.")
    train_rows.extend(aug_rows)

    # ---- step 6: featurize + save ---------------------------------------------------
    rng.shuffle(train_rows)

    def save_split(rows, split):
        df = pd.DataFrame(rows, columns=["name", "label", "country_code"])
        df.to_csv(os.path.join(args.out_dir, f"{split}.csv"), index=False, encoding="utf-8")
        X = np.zeros((len(rows), MAX_LEN), dtype=np.int16)
        for i, nm in enumerate(tqdm(df["name"].tolist(), desc=f"featurize {split}")):
            X[i] = name_to_indices(nm)
        y = df["label"].to_numpy(dtype=np.int64)
        np.save(os.path.join(args.out_dir, f"{split}_X.npy"), X)
        np.save(os.path.join(args.out_dir, f"{split}_y.npy"), y)
        return len(rows)

    n_tr = save_split(train_rows, "train")
    n_va = save_split(val_rows, "val")
    n_te = save_split(test_rows, "test")

    with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "name_country_map.json"), "w", encoding="utf-8") as f:
        json.dump(name_country_map, f, ensure_ascii=False)

    # per-class train counts + imbalance report
    train_counts = np.bincount(
        np.array([lbl for _n, lbl, _c in train_rows]), minlength=len(kept)
    ).tolist()
    ratio = max(train_counts) / max(1, min(c for c in train_counts if c > 0))
    stats = {
        "num_classes": len(kept),
        "n_train": n_tr, "n_val": n_va, "n_test": n_te,
        "max_per_country": args.max_per_country,
        "post_split_train_imbalance_ratio": round(ratio, 1),
        "n_augmented": len(aug_rows),
        "n_multi_country_names": len(name_country_map),
        "dropped_countries": dropped,
    }
    with open(os.path.join(args.out_dir, "preprocess_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n✅ Preprocessing complete.")
    print(f"   classes={len(kept)}  train={n_tr:,}  val={n_va:,}  test={n_te:,}")
    print(f"   train imbalance ratio (max/min) = {ratio:.1f}")
    print(f"   outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
