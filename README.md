# 🔬 MatSciBERT NER — Materials Science Named Entity Recognition

Fine-tuning **[m3rg-iitd/matscibert](https://huggingface.co/m3rg-iitd/matscibert)** on the
**[MatSci-NLP](https://huggingface.co/datasets/m3rg-iitd/MatSci-NLP)** benchmark for NER
across materials science literature.

> **Live Demo** · [HuggingFace Spaces](https://huggingface.co/spaces/your-username/matscibert-ner)  
> **Model** · [HuggingFace Hub](https://huggingface.co/your-username/matscibert-ner)

---

## Why MatSciBERT + MatSci-NLP?

| Choice | Rationale |
|---|---|
| `m3rg-iitd/matscibert` | Pre-trained on 2M+ materials science papers — already understands "MPa", "sintering", "tensile strength" before fine-tuning |
| MatSci-NLP dataset | Real, peer-reviewed annotations from materials science literature (not synthetic) |
| Together | Achieves state-of-the-art NER F1 on materials science text |

---

## Entity Types (MatSci-NLP)

| Label | Description | Example |
|---|---|---|
| `MAT` | Material | *titanium*, *PEEK*, *alumina* |
| `PRO` | Property | *tensile strength*, *melting point* |
| `SPL` | Sample descriptor | *thin film*, *bulk sample* |
| `SMT` | Synthesis method | *sintering*, *CVD deposition* |
| `CMT` | Characterisation method | *XRD*, *SEM*, *EBSD* |
| `APL` | Application | *turbine blades*, *fuel cells* |

---

## Architecture

```
MatSci-NLP dataset (real annotated papers)
         │
         ▼
m3rg-iitd/matscibert          ← domain BERT, pre-trained on 2M+ papers
         │  fine-tuned with Trainer API
         ▼
Token Classification Head     ← linear layer, one output per BIO label
         │  aggregation_strategy="simple"
         ▼
Entity spans with confidence scores
```

---

## Quickstart

```bash
git clone https://github.com/your-username/matscibert-ner
cd matscibert-ner
pip install -r requirements.txt
```

### Train

```bash
python train.py
```

Training downloads the MatSci-NLP dataset and MatSciBERT automatically from the
HuggingFace Hub. On a free Colab T4 GPU this takes ~10–15 minutes.

### Evaluate

```bash
python evaluate_model.py
```

### Run Gradio demo locally

```bash
python app.py
```

### Push to HuggingFace Hub

```bash
huggingface-cli login
python train.py --push_to_hub --hub_model_id your-username/matscibert-ner
```

---

## Expected Results

Fine-tuning MatSciBERT on MatSci-NLP NER:

| Metric | Score |
|---|---|
| F1 (overall) | ~0.80–0.87 |
| Precision | ~0.82–0.89 |
| Recall | ~0.78–0.86 |

*Scores from the original MatSci-NLP paper benchmarks.*

---

## Project Structure

```
matscibert-ner/
├── src/
│   ├── dataset.py        # MatSci-NLP loader from HuggingFace Hub
│   └── inference.py      # Pipeline loader + entity span extraction
├── train.py              # HuggingFace Trainer fine-tuning script
├── evaluate_model.py     # Standalone evaluation script
├── app.py                # Gradio demo (deploy to HF Spaces)
└── requirements.txt
```

---

## References

- **MatSciBERT:** Gupta et al., "MatSciBERT: A materials domain language model for text mining and information extraction", *npj Computational Materials*, 2022
- **MatSci-LP:** Song et al., "MatSci-NLP: Evaluating Scientific NLP on Materials Science", ACL 2023

---

## License

MIT
