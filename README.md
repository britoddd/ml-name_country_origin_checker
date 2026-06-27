# 🌍 Name → Country-of-Origin Checker

Predict the most likely **countries of origin** for a romanized personal name, across
**~104 countries**, using character-level patterns. The repo is a complete, reproducible
ML pipeline — from raw name-frequency tables to a demo hosted on
[Hugging Face Spaces](https://huggingface.co/spaces) (built with [Gradio](https://gradio.app))
that doubles as a guided walkthrough of every pipeline stage.

**Repository:** https://github.com/britoddd/ml-name_country_origin_checker

> ## ⚠️ Ethics & limitations — read first
> A name is a **weak, probabilistic** signal. This tool reports a **statistical guess** over
> the character patterns of a romanized name — it is **not** a statement about any individual's
> nationality, ethnicity, or origin. Many names legitimately span multiple countries
> (e.g. *"ali ali"* appears in **30** of the 104 countries in this data), and the training data
> is romanized and uneven across countries.
>
> **Do not use for profiling, hiring, credit, immigration, or law-enforcement decisions.**
> Names entered into the demo are not logged or stored.

---

## What it does

Given a full name like `Joko Widodo`, the model returns a ranked list of countries with
confidence scores. Internally a name is:

1. **Normalized** — NFD-decomposed, accents stripped, lowercased, reduced to `a–z` + single
   spaces (identical in training and inference, guaranteed by a shared `common.py`).
2. **Vectorized** — character **n-gram TF-IDF**.
3. **Classified** — by a linear model over the 104 countries.

## The pipeline

The project is organized as a five-stage pipeline under [`pipeline/`](pipeline/). Each stage is
runnable on its own (`python -m pipeline.<module>`), and [`main.py`](main.py) runs them in order.

| # | Stage | Module | What it produces |
|---|-------|--------|------------------|
| 1 | **EDA** | `pipeline/eda01.py` | 8 plots + `eda/eda_report.md` + `eda_summary.json` — class balance, name length, character/token distributions, multi-country ambiguity |
| 2 | **Preprocess** | `pipeline/preprocess02.py` | Synthesizes realistic full names from the aggregated tables, stratified 70/15/15 split, featurizes to index arrays → `data/` |
| 3 | **Train** | `pipeline/baseline03.py` | Trains 3 linear models on char-TF-IDF → `models/*.pkl` + `baseline_metrics.json` |
| 4 | **Evaluate** | `pipeline/evaluation04.py` | Held-out test metrics (strict + ambiguity-aware) + plots → `models/baseline_eval_metrics.json` |
| 5 | **Predict** | `pipeline/predict05.py` | CLI inference: top-k countries for a name |

Shared helpers live in [`pipeline/helper/`](pipeline/helper/): `common.py` (featurization),
`eda_style.py` (plot styling), `deploy_space.py` (one-shot Space deploy).

## Models & results

Three scikit-learn linear classifiers over char n-gram TF-IDF. **Logistic Regression** is the
default served model — best macro-F1, trains in under 3 minutes on CPU.

| Model | Test macro-F1 | Accuracy | Top-3 | Top-5 | MRR | Set-lenient top-1¹ |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **0.500** | 51.1% | 69.3% | 77.1% | 0.629 | 55.1% |
| Linear SVM | 0.493 | 51.3% | 67.9% | 74.9% | 0.622 | 55.4% |
| SGD (modified Huber) | 0.478 | 50.0% | 67.3% | 74.6% | 0.611 | 53.9% |

¹ *Set-lenient* credits a prediction that matches **any** country a name legitimately belongs to
(via the multi-country map), isolating how much error is just name ambiguity (~+4 points).

Easiest countries (high F1): Iceland, South Korea, Cambodia, Azerbaijan, Lithuania.
Hardest: high-immigration / shared-script countries — US, Canada, and the Gulf states — where
names overlap heavily across borders.

## The demo app

[`app/`](app/) is a self-contained **Hugging Face Space** (Gradio UI). It presents the project
as **one page per pipeline stage** (EDA → Preprocess → Train → Evaluate → **Predict**): each stage page replays
its console log and reveals the **pre-generated** plots and metrics, while **Predict** runs live
(single name or batch CSV). Run it locally:

```bash
cd app
pip install -r requirements.txt
python app.py
```

## Project structure

```
.
├── main.py                 # run the full pipeline end-to-end
├── pipeline/
│   ├── eda01.py            # 1 · exploratory data analysis
│   ├── preprocess02.py     # 2 · build dataset + splits
│   ├── baseline03.py       # 3 · train linear models
│   ├── evaluation04.py     # 4 · evaluate on held-out test
│   ├── predict05.py        # 5 · CLI inference
│   └── helper/
│       ├── common.py       # shared normalization + featurization
│       ├── eda_style.py    # plot styling
│       └── deploy_space.py # deploy app/ to a Hugging Face Space
├── app/                    # Gradio demo / walkthrough (Hugging Face Space)
├── archive/                # raw input tables (forenames/surnames/country_codes)
├── data/                   # generated splits + featurized arrays + maps
├── eda/                    # EDA plots + report
└── models/                 # trained .pkl models, vectorizer, metrics, plots
```

## Data

Source tables in [`archive/`](archive/) are **aggregated frequency counts** of name tokens per
country (not pre-joined full names):

- `forenames.csv` — ~12.4M rows across 104 countries
- `surnames.csv` — ~21.1M rows across 104 countries
- `country_codes.csv` — country code → display name

Because there are no full-name rows, `preprocess02.py` **synthesizes** full names by sampling a
forename and surname weighted by their real per-country counts, reproducing realistic name
popularity while matching the inference contract (`"First Last"`).

## Setup & usage

```bash
# 1. install dependencies (full environment)
pip install -r requirements.txt

# 2. run the whole pipeline (EDA → preprocess → train → evaluate → predict)
python main.py

# …or run a single stage
python -m pipeline.eda01
python -m pipeline.baseline03

# 3. predict from the CLI
python -m pipeline.predict05 --name "Joko Widodo" --top_k 5
python -m pipeline.predict05 --name "Tanaka" --model baseline_linsvm.pkl --json
```

> **Note:** run pipeline modules from the repo root as `python -m pipeline.<module>` (not
> `python pipeline/<module>.py`) so the `pipeline` package resolves on `sys.path`.

### Deploy the demo

```bash
python -m pipeline.helper.deploy_space   # uploads app/ to the configured HF Space
```

## License

See [`archive/LICENSE.txt`](archive/LICENSE.txt) for the dataset license. The Space is published
under MIT (see [`app/README.md`](app/README.md)).