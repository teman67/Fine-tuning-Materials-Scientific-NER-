# 🔬 MatSciBERT NER — Chemical Named Entity Recognition

Fine-tuning **[m3rg-iitd/matscibert](https://huggingface.co/m3rg-iitd/matscibert)** on the
**[CHEMDNER](https://huggingface.co/datasets/kjappelbaum/chemnlp-chemdner)** dataset for
chemical entity recognition in biomedical literature.

> **Live Demo** · [HuggingFace Spaces](https://huggingface.co/spaces/your-username/matscibert-ner)  
> **Model** · [HuggingFace Hub](https://huggingface.co/your-username/matscibert-ner)

---

## Why MatSciBERT + CHEMDNER?

| Choice | Rationale |
|---|---|
| `m3rg-iitd/matscibert` | Pre-trained on 2M+ materials science papers — already understands scientific/chemical terminology before fine-tuning |
| CHEMDNER dataset | 19 440 annotated examples of chemical names and compounds from PubMed abstracts and patents |
| Together | Combines domain-aware language understanding with real chemical NER annotations |

---

## Entity Types

| Label | Description | Example |
|---|---|---|
| `B-CHEM` | Beginning of a chemical entity | *nitric oxide*, *ethanol*, *NaCl* |
| `I-CHEM` | Continuation of a chemical entity | (multi-token spans) |
| `O` | Outside any entity | (non-chemical tokens) |

The model outputs `CHEM` spans after BIO aggregation (e.g. `B-CHEM` + `I-CHEM` → one `CHEM` span).

---

## Architecture

```
CHEMDNER dataset (kjappelbaum/chemnlp-chemdner)
         │  entity strings → BIO token tags (src/dataset.py)
         ▼
m3rg-iitd/matscibert          ← domain BERT, pre-trained on 2M+ papers
         │  fine-tuned with HuggingFace Trainer API
         ▼
Token Classification Head     ← linear layer, 3 outputs (O / B-CHEM / I-CHEM)
         │  aggregation_strategy="simple"
         ▼
Entity spans with confidence scores
```

---

## Quickstart

```bash
git clone https://github.com/teman67/Fine-tuning-Materials-Scientific-NER-
cd Fine-tuning-Materials-Scientific-NER-
pip install -r requirements.txt
```

### Train

```bash
python train.py
```

Downloads the CHEMDNER dataset and MatSciBERT automatically from the HuggingFace Hub.
The dataset is split into ~6 800 training and ~6 800 validation examples.

### Evaluate

```bash
python evaluate_model.py
# or specify a custom model path:
python evaluate_model.py --model_path ./model_output/final
```

### Run Gradio demo locally

```bash
python app.py
```

### Push trained model to HuggingFace Hub

```bash
huggingface-cli login
python train.py --push_to_hub --hub_model_id your-username/matscibert-ner
```

---

## Dataset

**[kjappelbaum/chemnlp-chemdner](https://huggingface.co/datasets/kjappelbaum/chemnlp-chemdner)**
is derived from the [CHEMDNER corpus](https://doi.org/10.1186/1758-2946-7-S1-S2) (Krallinger et al., 2015).
It contains chemical entity annotations (drug and chemical names) from PubMed abstracts and patents.

| Split | Examples |
|---|---|
| Train | ~6 796 |
| Validation | ~6 808 |
| Total (raw) | 19 440 |

The raw dataset provides entity surface strings per sentence; `src/dataset.py` converts these
to BIO-tagged token sequences using character-offset matching.

---

## Results

Fine-tuning MatSciBERT on CHEMDNER:

| Metric | Expected range |
|---|---|
| F1 (CHEM) | 0.9146 |
| Precision | 0.9075 |
| Recall | 0.9219 |

*Exact scores depend on hardware, random seed, and number of epochs.*

---

## Project Structure

```
matscibert-ner/
├── src/
│   ├── dataset.py        # CHEMDNER loader + BIO conversion
│   └── inference.py      # Pipeline loader + entity span extraction
├── train.py              # HuggingFace Trainer fine-tuning script
├── evaluate_model.py     # Standalone evaluation script
├── app.py                # Gradio demo (deploy to HF Spaces)
└── requirements.txt
```

---

## References

- **MatSciBERT:** Gupta et al., "MatSciBERT: A materials domain language model for text mining and information extraction", *npj Computational Materials*, 2022
- **CHEMDNER:** Krallinger et al., "The CHEMDNER corpus of chemicals and drugs and its annotation principles", *Journal of Cheminformatics*, 2015

---

## License

MIT
