# FakeNews_NLP
# Fake News Classifier — NLP Challenge

## Project Overview
Binary text classifier that distinguishes fake (0) from real (1) news articles
using classical NLP techniques and machine learning on news headlines.

## Dataset
- `dataset/data.csv` — 36,429 labelled articles (label, title, text, subject, date)
- `dataset/validation_data.csv` — unlabelled articles for prediction (label = 2)

## Pipeline
1. **EDA** — class balance, length distributions, stylistic signals, top words per class
2. **Preprocessing** — tokenisation, lowercase, punctuation removal, stop word removal, lemmatisation + stemming
3. **Embeddings** — Bag of Words (top 1,000) and TF-IDF (top 5,000, bigrams, sublinear_tf)
4. **Feature Engineering** — 9 numeric stylistic features (caps ratio, exclamations, word count, etc.)
5. **Modelling** — Naive Bayes, Logistic Regression, Random Forest × 6 input matrices each (18 experiments)
6. **Prediction** — best model applied to validation data, output saved as CSV

## Results
| Model | Accuracy |
|---|---|
| **NB — TF-IDF + Numeric (stem)** ✅ | **99.64%** |
| NB — TF-IDF + Numeric (lem) | 99.62% |
| LR — TF-IDF (stem) | 94.37% |

Best model: **Naive Bayes with TF-IDF + 9 numeric features**.
Only 25 misclassifications out of 7,286 test articles.

## Output
`predictions.csv` — same format as `validation_data.csv` with labels replaced by 0 or 1.

## Requirements
