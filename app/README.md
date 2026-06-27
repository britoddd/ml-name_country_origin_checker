---
title: Name to Nationality Classifier
emoji: 🌍
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
license: mit
---

# 🌍 Name → Nationality classifier (prototype)

Predicts likely **countries** from the character patterns of a romanized name, across
~104 countries. Character n-gram **TF-IDF** + linear classifiers (Logistic Regression /
Linear SVM / SGD), optimized for **macro-F1** (every country weighted equally).

**Source code:** [github.com/britoddd/ml-name_country_origin_checker](https://github.com/britoddd/ml-name_country_origin_checker)

## A guided pipeline tour
The Space is organized as **one page per pipeline stage**, mirroring `pipeline/`:
**1 · EDA → 2 · Preprocess → 3 · Baselines → 4 · Evaluation → 5 · Predict**. Each stage
page replays that stage's console log (▶ Run this step) and then shows the **pre-generated**
plots and metrics it produced. Only **Predict** runs live — the heavy stages ran offline
over millions of rows.

## Features
- **Pipeline walkthrough** → EDA plots, dataset stats, baseline/eval metrics per stage
- **Single name** → top-k countries with confidence bars
- **Batch CSV** → upload a column of names, download predictions
- **Backend toggle** → Logistic Regression / Linear SVM / SGD

## ⚠️ Ethics & limitations
A name is a *weak, probabilistic* signal. This tool reports statistical guesses over
romanized character patterns — **not** facts about any individual's nationality, ethnicity,
or origin. Many names legitimately span multiple countries, and the training data
(aggregated, leaked name-frequency tables) is romanized and uneven across countries.

**Do not use for profiling, hiring, credit, immigration, or law-enforcement decisions.**
Names entered into the demo are not logged or stored.

## How it works
Names are normalized to lowercase `a–z` + space (accents stripped, non-Roman dropped) —
identical to training (`common.py`) — then turned into character n-gram TF-IDF features and
classified by a linear model. Runs on CPU; inference is milliseconds.
