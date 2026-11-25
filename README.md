# GeoRAG-EuroSAT: annotate satellite images and query them in natural language

A compact, end-to-end pipeline that turns a collection of Sentinel-2 satellite images into a knowledge base you can query in French, with answers that cite the images they rely on.

**Stack:** Python, PyTorch, OpenCLIP (multilingual CLIP), scikit-learn, rank-bm25, Groq API (Llama 3.3), pandas, matplotlib

## What it does

| Step | Method |
| --- | --- |
| 1. Classification | Multilingual CLIP (`xlm-roberta-base-ViT-B-32`): zero-shot with French prompt ensembles, then a logistic-regression linear probe on frozen embeddings |
| 2. Description | Pixel attributes (Excess Green vegetation share, water share, brightness, texture), binned into tertiles, plus class keywords, assembled into a French text record per image |
| 3. Knowledge base | Records (JSONL) and CLIP image embeddings for the held-out 30 % of images |
| 4. Retrieval | BM25 over records, CLIP text-to-image similarity, and a hybrid of both via Reciprocal Rank Fusion |
| 5. RAG | An LLM answers only from retrieved records, cites image IDs, and abstains when the base cannot answer |
| 6. Evaluation | Accuracy and macro F1; P@10, nDCG@10 and MRR on 20 French queries with bootstrap CIs; citation validity and abstention on in- and out-of-domain questions |

## Results

Sample: 3,000 images (300 per class), CPU run; 2,100 images train the linear probe and the 900 held-out images form the knowledge base.

| Metric | Value |
| --- | --- |
| Zero-shot accuracy / macro F1 | 0.540 / 0.464 |
| Linear-probe accuracy / macro F1 | 0.910 / 0.910 |
| nDCG@10: BM25 without keywords / with keywords | 0.306 / 0.840 |
| nDCG@10: CLIP text-to-image | 0.731 |
| nDCG@10: hybrid (RRF), 95 % bootstrap CI | 0.865 [0.73, 0.98] |
| Valid citations / correct abstention behaviour (6 questions) | 100 % / 100 % |

Main confusions: permanent crop vs annual crop (10 %), herbaceous vegetation vs forest (9 %), sea/lake vs river (7 %). BM25 with keywords is optimistic: keywords and test queries were written by the same author, and BM25 fails on queries with no shared words ("zones urbaines": 0 vs 0.94 for CLIP). The hybrid's edge over BM25 is not significant with 20 queries.

## Run it

1. Open `GeoRAG_EuroSAT.ipynb` in Google Colab and select a GPU runtime.
2. Optional: add a free Groq API key as a Colab secret named `GROQ_API_KEY`. Without it, the notebook still runs and uses a rule-based fallback answer.
3. Run all cells. EuroSAT downloads automatically through `torchvision`. With 800 images per class a run takes a few minutes on a GPU; on CPU, set `N_PAR_CLASSE = 300` (about 15 minutes).

Outputs are written to `outputs/`: the knowledge base (`base_de_connaissances.jsonl`), CLIP embeddings, and the classification, retrieval and LLM evaluation tables.

## Limitations

- Records inherit classifier errors: a misclassified image is misdescribed.
- RGB attributes are proxies; a real NDVI needs the near-infrared band from multispectral EuroSAT.
- The RGB version has no coordinates, so spatial queries ("near Lyon") are out of scope.
- Relevance is defined at class level and the 20 queries are hand-written.

## Data

EuroSAT (Helber et al., 2019): 27,000 Sentinel-2 RGB patches, 64 × 64 px at 10 m, 10 land-cover classes.
