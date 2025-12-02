# Text-Data-Processing-and-Topic-Modeling-with-SciBErt
This project builds an end-to-end NLP workflow for scientific papers. Using SciBERT and BERT-base for comparisont preprocesses arXiv abstracts, generates embeddings, reduces dimensionality (PCA/UMAP), clusters (K-Means), and evaluates quality (Silhouette, Davies–Bouldin). We also address class imbalance by focusing on top categories and show that UMAP + K-Means significantly outperforms PCA + K-Means for this dataset.

Tecbook Link: https://drive.google.com/file/d/1q3kFhpPmWOrx2uozJcwRVG2t8N9AeUTM/view?usp=sharing

# What’s implemented
- Data source: via kagglehub → sumitm004/arxiv-scientific-research-papers-dataset.

- Columns used: summary, processed_summary, category (plus new embedding columns).

- Preprocessing: NLTK resources, normalization, stopword handling (per your preprocessing cell).

Embeddings:

- BERT-base-uncased and SciBERT (tokenizer + model loaded via transformers), mean-pooling to fixed vectors.

Dimensionality reduction:

- PCA with optimal components search (observed optima in your runs: BERT: 137 comps, SciBERT: 149 comps).

- UMAP for non-linear structure prior to clustering.

Clustering: K-Means with k chosen by elbow/KneeLocator (your notebook points to k ≈ 6).

Evaluation: Silhouette Score and Davies–Bouldin; seaborn/matplotlib plots for inspection.

Comparative result (from your notes):

- PCA + K-Means Silhouette: BERT ~ 0.0287, SciBERT ~ 0.0365.

- UMAP + K-Means (SciBERT) improves to ~0.4297 Silhouette, with lower Davies–Bouldin (better separation).

# Results

SciBERT > BERT on scientific text (higher silhouette at the same k).

UMAP meaningfully improves cluster separability vs. PCA on this corpus.

k ≈ 6 captures broad thematic structure in the filtered top categories.

# Tech stack

transformers, torch, scikit-learn, umap-learn, hdbscan (optional), kneed, pandas, numpy, matplotlib, seaborn, nltk, kagglehub.
