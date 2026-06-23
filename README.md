# LLM Prompt Engineering & DistilBERT Fine-Tuning: Yelp Star Rating Prediction

Three approaches to predicting customer star ratings (1–5) from Yelp restaurant reviews — spanning the full spectrum from zero-parameter prompt engineering to supervised fine-tuning. Built for the *Large Language Models for Marketing* course at Erasmus University Rotterdam.

**Dataset:** 19,896 Yelp restaurant reviews (heavily skewed: 55% are 5-star). Stratified sampling used throughout to ensure balanced class representation.

---

## Part 1 — System Prompt Design

Two system prompt personas applied to Llama 3.1 8B (via Groq API), holding all other parameters constant (temperature = 0.3):

- **Prompt A — Academic Analyst:** Neutral, structured output — sentiment polarity, key linguistic signals, predicted rating
- **Prompt B — Marketing Consultant:** Action-oriented output — customer verdict, what went well/poorly, one concrete managerial recommendation

Both correctly identified sentiment direction across all five test reviews (one per star level). The key difference emerged on ambiguous 3-star reviews: Prompt A catalogued conflicting signals; Prompt B went straight to a business takeaway. Core trade-off: **analytical depth vs operational utility**.

---

## Part 2 — Zero / One / Few-Shot Prompting

Same Llama 3.1 8B model, evaluated on a stratified 100-review sample:

| Condition | Accuracy | MAE |
|---|---|---|
| **Zero-shot** | **0.66** | **0.36** |
| One-shot (positive example) | 0.55 | 0.48 |
| One-shot (negative example) | 0.51 | 0.58 |
| Few-shot (5 examples) | 0.54 | 0.50 |

**Zero-shot won** — adding examples hurt performance. Explanation: Llama 3.1 8B has strong pre-trained priors on review data; in-context examples compete with rather than complement this knowledge. The one-shot conditions revealed **anchoring bias** — a negative example triggered a contrast effect, inflating mean predicted ratings.

Managerial implication: for large capable LLMs classifying customer feedback at scale, zero-shot is most efficient — no manual example curation required. This pattern likely reverses for smaller models.

---

## Part 3 — DistilBERT Fine-Tuning (Google Colab, T4 GPU)

DistilBERT-base-uncased fine-tuned for 5-class star rating prediction. Two configurations compared:

| | Config A — Full fine-tuning | Config B — Head only |
|---|---|---|
| Trainable parameters | 66,957,317 (100%) | 594,437 (0.9%) |
| Learning rate | 2e-5 | 1e-4 |
| **Test accuracy** | **0.475** | 0.415 |
| **MAE (0–4 scale)** | **0.735** | 0.860 |

**Config A (full fine-tuning) selected.** Per-class breakdown:

| Rating | Precision | Recall | F1 |
|---|---|---|---|
| 1★ | 0.57 | 0.42 | 0.49 |
| 2★ | 0.38 | 0.45 | 0.41 |
| 3★ | 0.39 | 0.35 | 0.37 |
| 4★ | 0.43 | 0.30 | 0.35 |
| 5★ | 0.58 | 0.85 | 0.69 |

5-star reviews are easiest (high recall); 1-star recall is low (0.42) — problematic from a business perspective as these flag the most urgent issues. Middle ratings (2–4★) are hardest to separate.

Compared to zero-shot Llama 3.1 8B (66% accuracy), fine-tuned DistilBERT achieves lower accuracy (47.5%) but offers a practical advantage: it runs locally, requires no API calls, and keeps customer data within the organization.

---

## Overall Takeaway

| Approach | Data needed | Accuracy | Key advantage |
|---|---|---|---|
| Prompt engineering (zero-shot) | None | 66% | No training, scales instantly |
| In-context learning (few-shot) | A few examples | 54% | Flexible, no fine-tuning |
| Fine-tuning (DistilBERT) | 800 labeled reviews | 47.5% | Portable, private, no API cost |

Prompt engineering requires no training data but is sensitive to model scale; fine-tuning requires labeled data but produces a cost-efficient, privacy-preserving model.

---

## Tech Stack
- Python 3
- `groq` (Llama 3.1 8B API) · `transformers` · `torch` · `datasets`
- `scikit-learn` · `pandas` · `matplotlib`
- Google Colab (T4 GPU) for Part 3

## Data
Yelp Restaurant Reviews (~19,896 reviews) — not included. Available on Kaggle.

## Setup
For Part 1 & 2: set your Groq API key as an environment variable — `export GROQ_API_KEY=your_key_here` — then update the notebook cell accordingly. Free tier at [console.groq.com](https://console.groq.com).
