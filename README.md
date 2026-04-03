# Formal Speech → Gen Z Meme Language
### LSTM Seq2Seq Baseline + FLAN-T5 QLoRA Fine-Tuning

**Author:** Divya Gunjan

---

## Overview

This project trains two models to translate formal English into Gen Z slang,
progressing from a simple LSTM baseline to a parameter-efficient transformer fine-tune.

| Model | BLEU | ROUGE-L |
|---|:---:|:---:|
| LSTM Seq2Seq (baseline) | 42 | 0.54 |
| **FLAN-T5-Large + QLoRA** | **58** | **0.76** |

---

## Dataset

[MLBtrio/genz-slang-dataset](https://huggingface.co/datasets/MLBtrio/genz-slang-dataset) — ~1,800 Gen Z slang entries with descriptions and example sentences.

Pairs are constructed by substituting the slang token in each example sentence with its plain-English description:

```
Slang:       "Got the job today, big W!"
→ Formal:    "got the job today, big win!"
```

Split: **70% train / 15% val / 15% test**

---

## Approach

### Part 1 — LSTM Seq2Seq Baseline

- Custom BPE tokenizer trained on the corpus (vocab = 8,000)
- Single-layer encoder–decoder LSTM with 50% teacher forcing
- Trained for 10 epochs with AdamW + linear LR schedule
- Evaluated with SacreBLEU

**Limitation:** no attention mechanism — the encoder must compress the entire source into one fixed hidden vector, leading to poor generation on longer inputs.

### Part 2 — FLAN-T5-Large + QLoRA

- `google/flan-t5-large` (770M params) loaded in **4-bit NF4** via BitsAndBytes — fits on a single T4 GPU
- **QLoRA** (rank 8, alpha 16) adapters on the Q & V attention projections — only ~0.1% of params trained
- Instruction-prompted inputs: `"Translate to GenZ slang: <formal sentence>"`
- Trained for 3 epochs with AdamW (lr = 2e-4)
- Evaluated with SacreBLEU + ROUGE-L

---

## Results

```
★ BLEU score:  58.xx
★ ROUGE-L:     {'rouge1': '0.xxx', 'rouge2': '0.xxx', 'rougeL': '0.76x', ...}
```

Example predictions:

| Formal | Reference | Predicted |
|---|---|---|
| "got the job today, big win!" | "Got the job today, big W!" | "Got the job today, big W!" |

---

## How to Run

Open `genz_translator.ipynb` in **Google Colab** with a **T4 GPU** runtime.

Run all cells top-to-bottom. The notebook is self-contained — it installs all dependencies in the first cell.

```bash
# Local setup (GPU required for Part 2)
pip install -r requirements.txt
jupyter notebook genz_translator.ipynb
```

---

## Files

```
genz-translator/
├── genz_translator.ipynb   # full notebook (LSTM + FLAN-T5)
├── requirements.txt
└── README.md
```

---

## Key Techniques

- **4-bit quantization** (NF4) via BitsAndBytes
- **QLoRA** — Low-Rank Adaptation for parameter-efficient fine-tuning
- **Instruction prompting** for seq2seq task framing
- **SacreBLEU + ROUGE-L** evaluation
- **Teacher forcing** with greedy decoding (LSTM baseline)
