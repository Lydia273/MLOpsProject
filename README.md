# Song-Lyrics Sentiment Analysis — an MLOps pipeline

Sentiment analysis of English song lyrics (five classes: Very Negative → Very
Positive) built as a full **MLOps pipeline** rather than a one-off notebook:
versioned data, tracked experiments, reproducible configs, containerised training
and serving, and CI.

> Group project for the **Machine Learning Operations (02476)** course at DTU.
> Team: Afonso Cunha, **Lydia Kasapi**, Sabina Kozłowska, Sandra Castro Gómez.
> This repository is my fork of the team repo
> ([AfonsoCunha22/MLOpsProject](https://github.com/AfonsoCunha22/MLOpsProject));
> see [My contributions](#my-contributions).

## What it does

- **Data**: a subset of the *5M Song Lyrics* dataset (Genius), filtered to English
  lyrics using pre-computed language IDs. Raw CSVs live in `data/raw/`; processed
  tensors are produced by `src/sentiment_analysis/data.py` and tracked with DVC.
- **Model**: `tabularisai/multilingual-sentiment-analysis` (a fine-tuned
  DistilBERT) used as the sentiment classifier / baseline, via Hugging Face
  Transformers.
- **Training**: `src/sentiment_analysis/train.py`, configured with **Hydra**
  (`src/sentiment_analysis/conf/`) and logged to **Weights & Biases**; a sweep
  config is in `configs/sweep.yaml`.
- **Serving**: FastAPI app in `src/sentiment_analysis/api.py`, containerised via
  `dockerfiles/api.dockerfile`.
- **Reproducibility & CI**: `pyproject.toml` + pinned `requirements*.txt`,
  `.pre-commit-config.yaml`, DVC remote on GCS, and a GitHub Actions workflow
  (`.github/workflows/tests.yml`) running `pytest` + coverage on Windows /
  Python 3.11–3.12.

## Tech stack

| Concern | Tool |
|---|---|
| Modelling | Hugging Face Transformers, PyTorch |
| Data versioning | DVC (Google Cloud Storage remote) |
| Experiment tracking | Weights & Biases |
| Config management | Hydra |
| Serving | FastAPI + Docker |
| Quality / CI | pytest, coverage, pre-commit, GitHub Actions |

## Repository layout

```txt
.
├── .github/workflows/       # CI: unit tests + coverage, dependabot
├── configs/                 # Hydra sweep config
├── data/
│   ├── raw/                 # source CSVs (Train.csv, Test.csv, synthetic_social_media_data.csv)
│   └── processed.dvc        # DVC pointer to processed tensors
├── dockerfiles/             # api.dockerfile
├── docs/                    # mkdocs sources
├── models/                  # trained model artifacts
├── src/sentiment_analysis/
│   ├── data.py              # raw CSV -> tokenised tensors
│   ├── model.py             # model definition / loading
│   ├── train.py             # Hydra + W&B training entry point
│   ├── evaluate.py          # evaluation
│   ├── api.py               # FastAPI inference service
│   └── conf/config.yaml     # Hydra config
├── tests/                   # test_data.py, test_model.py, test_api.py
├── train.dockerfile
├── tasks.py                 # invoke tasks (env, requirements, train, ...)
└── pyproject.toml
```

*(The project used to sit one directory deeper inside the repo; it has been
flattened to the root so the entry point is obvious.)*

## Setup

```bash
git clone https://github.com/Lydia273/MLOpsProject.git
cd MLOpsProject
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# pull versioned data (needs access to the DVC remote)
dvc pull

# train
python src/sentiment_analysis/data.py data/raw data/processed
python src/sentiment_analysis/train.py

# serve
uvicorn src.sentiment_analysis.api:app --reload
```

`invoke --list` shows the shortcut tasks defined in `tasks.py`.

## My contributions

<!-- TODO(Lydia): confirm / correct the bullets below before publishing.
     Git history on this fork only attributes data-upload commits to me, so I
     can't reconstruct the split automatically. Replace the bracketed text. -->

- **[confirm]** Data layer — assembling and cleaning the raw lyric CSVs
  (`data/raw/`), the English-language filtering, and the
  `src/sentiment_analysis/data.py` preprocessing into tokenised tensors.
- **[confirm]** [e.g. wiring up W&B logging in `train.py` / writing the Hydra
  config / the `tests/test_data.py` suite / the FastAPI endpoint — keep whichever
  are yours].
- **[confirm]** [DVC + GCS remote setup? CI workflow? Dockerfiles?]

Team members and their focus areas: **[confirm]**.

## License

MIT — see [LICENSE](LICENSE). Created from
[SkafteNicki/mlops_template](https://github.com/SkafteNicki/mlops_template).
