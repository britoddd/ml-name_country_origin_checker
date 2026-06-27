import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from pipeline.helper import force_utf8_stdout, tqdm

force_utf8_stdout()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline.helper import eda_style as st

st.apply_style()

# Anchor default I/O dirs to the project root (parent of pipeline/) so outputs always
# land in the project's real structure, regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scan_table(path, token_col, chunksize, count_sample_cap):
    """Chunked pass over an aggregated table -> per-country stats + a sample of counts."""
    rows = Counter()                  # rows per country
    mass = defaultdict(float)         # sum of `count` per country
    distinct = defaultdict(set)       # distinct lowercased tokens per country
    count_sample = []                 # sampled raw count values (for the histogram)
    n_rows = 0

    reader = pd.read_csv(path, chunksize=chunksize,
                         usecols=[token_col, "country", "count"],
                         dtype={token_col: "string", "country": "string", "count": "int64"})
    for chunk in tqdm(reader, desc=f"scan {os.path.basename(path)}"):
        chunk = chunk.dropna(subset=[token_col, "country"])
        n_rows += len(chunk)
        rows.update(chunk["country"].value_counts().to_dict())
        for cc, s in chunk.groupby("country")["count"].sum().items():
            mass[cc] += float(s)
        toks = chunk[token_col].str.lower()
        for cc, grp in chunk.assign(_t=toks).groupby("country")["_t"]:
            distinct[cc].update(grp.values)
        if len(count_sample) < count_sample_cap:
            count_sample.extend(chunk["count"].values[: count_sample_cap - len(count_sample)])

    distinct_counts = {cc: len(s) for cc, s in distinct.items()}
    return {"n_rows": n_rows, "rows_per_country": dict(rows),
            "mass_per_country": dict(mass), "distinct_per_country": distinct_counts,
            "count_sample": np.asarray(count_sample, dtype=np.int64)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "data"))
    ap.add_argument("--archive_dir", default=os.path.join(PROJECT_ROOT, "archive"))
    ap.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "eda"))
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--count_sample", type=int, default=3_000_000,
                    help="cap on sampled `count` values for the log-distribution plot")
    ap.add_argument("--name_sample", type=int, default=300_000,
                    help="rows sampled from train.csv for length/char distributions")
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--skip_raw", action="store_true",
                    help="skip the 33M-row archive scan (splits-only EDA)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cc_path = os.path.join(args.archive_dir, "country_codes.csv")
    cc_names = {}
    if os.path.exists(cc_path):
        cdf = pd.read_csv(cc_path)
        cc_names = dict(zip(cdf["country_code"].astype(str), cdf["country_name"].astype(str)))

    summary = {}

    # ---- (A) raw aggregated tables -------------------------------------------------
    if not args.skip_raw:
        fore = scan_table(os.path.join(args.archive_dir, "forenames.csv"),
                          "forename", args.chunksize, args.count_sample)
        sur = scan_table(os.path.join(args.archive_dir, "surnames.csv"),
                         "surname", args.chunksize, args.count_sample)
        summary["raw"] = {
            "forenames_rows": fore["n_rows"], "surnames_rows": sur["n_rows"],
            "countries_in_forenames": len(fore["distinct_per_country"]),
            "countries_in_surnames": len(sur["distinct_per_country"]),
            "distinct_forenames_min": min(fore["distinct_per_country"].items(), key=lambda kv: kv[1]),
            "distinct_forenames_max": max(fore["distinct_per_country"].items(), key=lambda kv: kv[1]),
            "distinct_surnames_min": min(sur["distinct_per_country"].items(), key=lambda kv: kv[1]),
            "distinct_surnames_max": max(sur["distinct_per_country"].items(), key=lambda kv: kv[1]),
        }

        # plot: distinct tokens per country (sorted), log scale
        ccs = sorted(fore["distinct_per_country"],
                     key=lambda c: fore["distinct_per_country"][c], reverse=True)
        fvals = [fore["distinct_per_country"][c] for c in ccs]
        svals = [sur["distinct_per_country"].get(c, 0) for c in ccs]
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(ccs))
        ax.bar(x - 0.2, fvals, width=0.4, label="forenames", color=st.PRIMARY)
        ax.bar(x + 0.2, svals, width=0.4, label="surnames", color=st.SECONDARY)
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(ccs, rotation=90, fontsize=6)
        ax.legend()
        st.finalize(ax, "Distinct forename/surname tokens per country (raw)",
                    ylabel="distinct tokens (log)")
        fig.savefig(os.path.join(args.out_dir, "eda_tokens_per_country.png")); plt.close(fig)

        # plot: distribution of `count` values (log10), both tables
        fig, ax = plt.subplots(figsize=(9, 5))
        for sample, lab, col in [(fore["count_sample"], "forenames", st.PRIMARY),
                                 (sur["count_sample"], "surnames", st.SECONDARY)]:
            v = sample[sample > 0]
            ax.hist(np.log10(v), bins=60, alpha=0.55, label=f"{lab} (n={len(v):,})", color=col)
        ax.legend()
        st.finalize(ax, "Token frequency distribution — many orders of magnitude",
                    xlabel="log₁₀(count)", ylabel="frequency")
        fig.savefig(os.path.join(args.out_dir, "eda_count_distribution.png")); plt.close(fig)
        print(f"   raw: forenames {fore['n_rows']:,} rows, surnames {sur['n_rows']:,} rows")
    else:
        print("   (skipped raw archive scan)")

    # ---- (B) generated splits ------------------------------------------------------
    with open(os.path.join(args.data_dir, "label_map.json"), encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    num_classes = len(label_map)

    split_counts = {}
    for split in ("train", "val", "test"):
        y = np.load(os.path.join(args.data_dir, f"{split}_y.npy"))
        split_counts[split] = np.bincount(y, minlength=num_classes)
    tr = split_counts["train"]
    imbalance = float(tr.max() / max(tr.min(), 1))

    # class balance bar (train), country codes sorted by train size
    order = np.argsort(-tr)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(num_classes), tr[order], color=st.PRIMARY, width=0.8)
    ax.axhline(tr.mean(), color=st.CORAL, ls="--", lw=1.2, label=f"mean = {tr.mean():.0f}")
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([label_map[i] for i in order], rotation=90, fontsize=6)
    ax.legend()
    st.finalize(ax, f"Per-country train samples  ·  imbalance {imbalance:.2f}×",
                ylabel="train samples")
    fig.savefig(os.path.join(args.out_dir, "eda_class_balance.png")); plt.close(fig)

    # name length / token count / char frequency from a train sample
    tdf = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    if len(tdf) > args.name_sample:
        tdf = tdf.sample(n=args.name_sample, random_state=42)
    names = tdf["name"].astype(str)
    lengths = names.str.len().to_numpy()
    n_tokens = names.str.split().map(len).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=range(0, 33), color=st.PRIMARY, edgecolor="white", linewidth=0.4)
    st.finalize(ax, f"Name length distribution  (sample n={len(names):,})",
                xlabel="normalized name length (chars, max_len=30)", ylabel="frequency")
    fig.savefig(os.path.join(args.out_dir, "eda_name_length.png")); plt.close(fig)

    tok_dist = Counter(n_tokens.tolist())
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = sorted(tok_dist)
    bars = ax.bar([str(k) for k in ks], [tok_dist[k] for k in ks], color=st.SECONDARY, width=0.7)
    ax.bar_label(bars, fmt="%d", padding=3, color=st.MUTED, fontsize=8); ax.margins(y=0.15)
    st.finalize(ax, "Tokens per generated name", xlabel="tokens per name", ylabel="frequency")
    fig.savefig(os.path.join(args.out_dir, "eda_tokens_per_name.png")); plt.close(fig)

    char_freq = Counter("".join(names.tolist()))
    chars = list("abcdefghijklmnopqrstuvwxyz ")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([("␣" if c == " " else c) for c in chars],
           [char_freq.get(c, 0) for c in chars], color=st.PRIMARY, width=0.8)
    st.finalize(ax, "Character frequency over normalized names (a–z + space)", ylabel="frequency")
    fig.savefig(os.path.join(args.out_dir, "eda_char_frequency.png")); plt.close(fig)

    # ---- (C) multi-country ambiguity -----------------------------------------------
    with open(os.path.join(args.data_dir, "name_country_map.json"), encoding="utf-8") as f:
        ncm = json.load(f)
    breadth = np.array([len(v) for v in ncm.values()])
    top_broad = sorted(ncm.items(), key=lambda kv: len(kv[1]), reverse=True)[: args.top_n]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(2, int(breadth.max()) + 2) if len(breadth) else [2, 3]
    ax.hist(breadth, bins=bins, color=st.ACCENT, edgecolor="white", linewidth=0.5, align="left")
    st.finalize(ax, f"Multi-country name breadth  (total = {len(ncm):,} names)",
                xlabel="# countries a name spans", ylabel="# names")
    fig.savefig(os.path.join(args.out_dir, "eda_multicountry.png")); plt.close(fig)

    summary["splits"] = {
        "num_classes": num_classes,
        "n_train": int(tr.sum()), "n_val": int(split_counts["val"].sum()),
        "n_test": int(split_counts["test"].sum()),
        "train_imbalance_ratio": round(imbalance, 3),
        "train_per_country_min": [label_map[int(order[-1])], int(tr.min())],
        "train_per_country_max": [label_map[int(order[0])], int(tr.max())],
        "name_length_mean": round(float(lengths.mean()), 2),
        "name_length_p95": int(np.percentile(lengths, 95)),
        "tokens_per_name": {str(k): int(v) for k, v in sorted(tok_dist.items())},
        "n_multi_country_names": len(ncm),
        "multi_country_breadth_mean": round(float(breadth.mean()), 3) if len(breadth) else 0,
        "multi_country_breadth_max": int(breadth.max()) if len(breadth) else 0,
        "broadest_names": [{"name": n, "n_countries": len(v), "countries": v}
                           for n, v in top_broad],
    }

    with open(os.path.join(args.out_dir, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # human-readable report
    s = summary["splits"]
    lines = ["# EDA report — name-to-nationality dataset\n"]
    if "raw" in summary:
        r = summary["raw"]
        lines += [
            "## Raw aggregated tables\n",
            f"- forenames.csv: **{r['forenames_rows']:,}** rows across {r['countries_in_forenames']} countries\n",
            f"- surnames.csv: **{r['surnames_rows']:,}** rows across {r['countries_in_surnames']} countries\n",
            f"- distinct forenames/country: min {r['distinct_forenames_min'][1]:,} "
            f"({r['distinct_forenames_min'][0]}) … max {r['distinct_forenames_max'][1]:,} "
            f"({r['distinct_forenames_max'][0]})\n",
            f"- distinct surnames/country: min {r['distinct_surnames_min'][1]:,} "
            f"({r['distinct_surnames_min'][0]}) … max {r['distinct_surnames_max'][1]:,} "
            f"({r['distinct_surnames_max'][0]})\n",
        ]
    lines += [
        "\n## Generated splits\n",
        f"- classes: **{s['num_classes']}**  |  train/val/test = "
        f"{s['n_train']:,} / {s['n_val']:,} / {s['n_test']:,}\n",
        f"- train imbalance (max/min): **{s['train_imbalance_ratio']}x** "
        f"(max {s['train_per_country_max'][0]}={s['train_per_country_max'][1]:,}, "
        f"min {s['train_per_country_min'][0]}={s['train_per_country_min'][1]:,})\n",
        f"- name length: mean {s['name_length_mean']} chars, p95 {s['name_length_p95']} "
        f"(cap is max_len=30)\n",
        f"- tokens per name: {s['tokens_per_name']}\n",
        f"\n## Multi-country ambiguity\n",
        f"- names spanning >1 country: **{s['n_multi_country_names']:,}** "
        f"(mean breadth {s['multi_country_breadth_mean']}, max {s['multi_country_breadth_max']})\n",
        f"- broadest example: `{s['broadest_names'][0]['name']}` "
        f"→ {s['broadest_names'][0]['n_countries']} countries\n",
    ]
    with open(os.path.join(args.out_dir, "eda_report.md"), "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n✅ EDA complete. {summary['splits']['num_classes']} classes, "
          f"imbalance {imbalance:.2f}x, {len(ncm):,} multi-country names. "
          f"Plots + eda_summary.json + eda_report.md -> {args.out_dir}/")


if __name__ == "__main__":
    main()
