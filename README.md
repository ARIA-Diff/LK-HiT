# LK-HiT

Statute-aware hierarchical transformers for calibrated multi-label legal judgment prediction across jurisdictions.

## Description

LK-HiT reads a case as paragraphs, chunks, or sentences, encodes each segment with a legal pre-trained encoder, and contextualises the segment states with a two-layer transformer. The same encoder reads the text of every candidate statutory label. Those label embeddings are refined by a two-layer graph convolution over co-occurrence and statutory-structure edges, then each label attends over the segment states. Multi-label tasks are trained with the asymmetric loss. CAIL2018 is trained with class-balanced cross-entropy. After training, one temperature per task is fitted on the development set. This repository contains that model, the preprocessing that builds its inputs, the task configurations, and the evaluation protocol for ECtHR-A, ECtHR-B, EUR-LEX, and CAIL2018.

## Dataset Information

Raw corpora are not included. Preprocessing reads the public releases and writes JSON lines under `data/processed/`.

**ECtHR-A.** LexGLUE release of the European Court of Human Rights corpus. Each document is the list of fact paragraphs. Labels are the ten Convention articles found violated (Articles 2, 3, 5, 6, 8, 9, 10, 11 and 14, and Article 1 of Protocol No. 1) plus an explicit no-violation label. Chronological split: 9,000 training cases (2001–2016), 1,000 development cases (2016–2017), 1,000 test cases (2017–2019). Mean length 1,690 tokens and 1.28 labels per document.

- URL: https://github.com/coastalcph/lex-glue
- Hugging Face: https://huggingface.co/datasets/coastalcph/lex_glue (`ecthr_a`)
- DOI: https://doi.org/10.18653/v1/2022.acl-long.297 (LexGLUE); https://doi.org/10.18653/v1/P19-1424 (ECtHR corpus)

**ECtHR-B.** The same documents and chronological split. Labels are the articles the applicant alleged, plus an explicit no-allegation label (1.61 labels per document). Silver rationales are the fact paragraphs referred to in the Court's assessment; 812 ECtHR-A test cases carry them.

- URL: https://github.com/coastalcph/lex-glue
- Hugging Face: https://huggingface.co/datasets/coastalcph/lex_glue (`ecthr_b`); rationales at https://huggingface.co/datasets/ecthr_cases
- DOI: https://doi.org/10.18653/v1/2022.acl-long.297 ; https://doi.org/10.18653/v1/2021.naacl-main.22

**EUR-LEX.** LexGLUE version of EUR-LEX: 65,000 EU legislative acts labelled with the 100 most frequent EuroVoc concepts, split chronologically into 55,000 / 5,000 / 5,000. Mean length 1,240 tokens and 5.1 labels per document.

- URL: https://github.com/coastalcph/lex-glue
- Hugging Face: https://huggingface.co/datasets/coastalcph/lex_glue (`eurlex`)
- DOI: https://doi.org/10.18653/v1/2021.emnlp-main.559 (MultiEURLEX); https://doi.org/10.18653/v1/2022.acl-long.297 (LexGLUE release)
- EuroVoc: https://op.europa.eu/en/web/eu-vocabularies/dataset/-/resource?uri=http://publications.europa.eu/resource/dataset/eurovoc

**CAIL2018.** Chinese criminal judgments with fact descriptions and applicable articles of the Criminal Law. The LADAN protocol keeps CAIL-small cases with one applicable article and drops articles with fewer than 100 training cases, leaving 103 articles, 101,619 training cases and 26,749 test cases. The release has no development set used here: a stratified 10% of those training cases (10,162, seed 1) is held out, and the model trains on the remaining 91,457. Mean fact length 412 characters, one label per document.

- URL: https://github.com/thunlp/CAIL
- DOI: https://doi.org/10.48550/arXiv.1807.02478
- LADAN protocol DOI: https://doi.org/10.18653/v1/2020.acl-main.280

Label descriptions are the English text of the corresponding Convention article (`data/statutes/ecthr.json`; https://www.echr.coe.int/european-convention-on-human-rights), the EuroVoc preferred term plus its scope note and broader terms, or the Criminal Law article text distributed with CAIL2018. The no-violation label is “The Court finds that none of the Convention articles under examination has been violated”. The no-allegation label is “The applicant does not allege a violation of any of the Convention articles under examination”.

## Code Information

- `src/lkhit/preprocess.py` builds segment lists, label texts, and the train, development, and test files.
- `src/lkhit/data.py` loads those files and tokenises paragraphs, 128-token chunks, or sentences.
- `src/lkhit/graph.py` builds positive normalised PMI edges on the training split, adds same-section edges, and symmetrically normalises the sum.
- `src/lkhit/model.py` is the shared encoder, the two-layer paragraph transformer, the two-layer graph convolution, and label-aware attention. It also covers the mean-pooled hierarchical classifier, the 512-token truncation, frozen label encoding, random label embeddings, and a Hugging Face sequence classifier for the sparse-attention and truncated-encoder baselines.
- `src/lkhit/loss.py` is the asymmetric loss and class-balanced cross-entropy.
- `src/lkhit/train.py` runs AdamW, linear warm-up, and early stopping, then fits the temperature and evaluates the selected checkpoint once.
- `src/lkhit/calibrate.py` fits that temperature with L-BFGS.
- `src/lkhit/evaluate.py` reports micro-F1 and macro-F1, or CAIL accuracy and macro precision, recall, and F1; ECE, Brier score, and negative log-likelihood; selective prediction; frequency tiers; year and length subgroups; and, given a second probability file, the paired bootstrap and the correlation of per-label gains with frequency.
- `src/lkhit/rationales.py` returns the five paragraphs with the highest label-aware attention and scores them against silver rationales, together with a random selection, the first five paragraphs, and TF–IDF similarity to the predicted article text.
- `src/lkhit/baselines.py` is word and bigram TF–IDF with a linear SVM.
- `src/lkhit/zeroshot.py` is the Qwen2.5-7B-Instruct prompt.
- `configs/ecthr_a.yaml`, `configs/ecthr_b.yaml`, `configs/eurlex.yaml`, and `configs/cail2018.yaml` fix the hyper-parameters chosen on the ECtHR-A development set.
- `data/statutes/ecthr.json` holds the Convention texts and the section groups used as structural edges.

## Usage Instructions

```bash
pip install -e .
python -m lkhit.preprocess --task ecthr_a
python -m lkhit.preprocess --task ecthr_b
python -m lkhit.preprocess --task eurlex --download-eurovoc
python -m lkhit.preprocess --task cail2018 --cail-train data/raw/cail2018/data_train.json --cail-test data/raw/cail2018/data_test.json --cail-articles data/raw/cail2018/articles.json
python -m lkhit.train --config configs/ecthr_a.yaml --seed 1
python -m lkhit.evaluate --config configs/ecthr_a.yaml --ckpt outputs/ecthr_a/seed_1/best.pt
```

Each processed record has `paragraphs` or `text`, integer `labels`, `segment`, `n_chars`, and, when the source provides them, `year` and `rationales`. Label names, description texts, and structural groups are in `labels.json`. `--all-seeds` runs seeds 1–5 and writes the mean and standard deviation to `outputs/<task>/summary.json`. The checkpoint stores the early-stopped epoch, development macro-F1, and fitted temperature. Decisions use a threshold of 0.5, or the arg-max on CAIL2018. No per-label threshold is tuned.

Ablations use the same file with one override, for example `--set use_label_graph=false`, `--set use_statute_knowledge=false`, `--set use_label_attention=false`, `--set use_hierarchy=false`, `--set loss=bce`, `--set use_general_encoder=true`, or `--set freeze_label_encoder=true`. An encoder baseline uses `--set arch=encoder model_name_or_path=allenai/longformer-base-4096 truncated_max_tokens=4096`. TF–IDF + SVM uses `--set arch=tfidf_svm`. The zero-shot baseline calls a vLLM server at temperature 0:

```bash
python -m lkhit.zeroshot --config configs/ecthr_a.yaml --base-url http://127.0.0.1:8000/v1
```

EuroVoc may be supplied instead of downloaded. The JSON maps each LexGLUE concept id to `pref`, `scope`, `broader`, and `microthesaurus`. CAIL article texts are a JSON object from article number to article text. Paired bootstrap and long-tail gains take `--baseline-probs` pointing at a saved `.npy` array of the other model's test probabilities.

## Requirements

Python 3.10 or newer. The Python libraries are PyTorch, Transformers, Datasets, NumPy, SciPy, scikit-learn, PyYAML, and tqdm. Versions are pinned in `requirements.txt`. Training uses one GPU in mixed precision; the reported runs used an NVIDIA RTX A6000 (48 GB). If no GPU is present the same commands run in full precision on CPU. The zero-shot baseline additionally needs a running vLLM server for Qwen2.5-7B-Instruct and does not install that model through `requirements.txt`.

## Methodology

1. ECtHR-A and ECtHR-B keep the LexGLUE paragraph boundaries and chronological splits. A case with an empty article set receives the no-violation or no-allegation label, and other cases do not. At most 64 paragraphs are kept, each truncated to 128 tokens. This drops the tail of 4.1% of ECtHR documents.
2. EUR-LEX keeps the LexGLUE chronological split. Each act is split into consecutive chunks of 128 encoder tokens, at most 32 chunks (3.7% of acts are truncated). The label text is the EuroVoc preferred term, the scope note where one exists, and the chain of broader terms. Structural edges join concepts in the same micro-thesaurus.
3. CAIL2018 reads the CAIL-small train and test JSON. Cases with more than one applicable article are removed. Articles with fewer than 100 remaining training cases are removed, and test cases of those articles are removed with them. A stratified split (seed 1, 10,162 cases) is taken from the filtered training cases for early stopping and calibration. Facts are split at `。！？；`. At most 32 sentences are kept, each truncated to 64 tokens (0.6% of facts are truncated). Structural edges join articles in the same chapter of the Special Provisions. Because the task is single-label, the co-occurrence matrix is empty.
4. ECtHR structural edges join Articles 2–3, Articles 5–6, and Articles 8–11. Article 14, Protocol No. 1, and the no-violation / no-allegation labels are singletons. Co-occurrence edges are the positive normalised pointwise mutual information on the training split only, with the NPMI value as the weight. The two adjacency matrices are summed, self-loops are added, and the result is symmetrically normalised.
5. The paragraph encoder is LEGAL-BERT-base for English and Chinese RoBERTa-wwm-ext-base for Chinese. Label text is truncated to 256 tokens and encoded by the same model. A learned position embedding is added to each segment state. The document transformer has 2 layers, 8 heads, width 768, and dropout 0.1. The graph convolution is `E = Ã ReLU(Ã E0 W1) W2 + E0`. Label-aware attention is a single head scaled by `1/sqrt(d)`. The score is bilinear in the label-specific summary and the label embedding, with a per-label bias. Training uses temperature 1. CAIL2018 applies a softmax; the other tasks apply a sigmoid. The asymmetric loss uses `(γ+, γ−, m) = (0, 2, 0.05)`. Class-balanced weights use `β = 0.999` and are scaled to mean 1.
6. AdamW uses learning rate `3e-5` on the pre-trained encoder and `1e-4` on the document transformer, graph convolution, and classifier, with weight decay 0.01. The schedule warms up linearly for the first 10% of the planned steps and then decays linearly. The effective batch is 16 documents. Training runs for at most 20 epochs and stops after 3 epochs without an improvement in development macro-F1. The selected checkpoint is evaluated once on the test set. Seeds are 1, 2, 3, 4, and 5.
7. One positive temperature is then fitted on the development set by minimising negative log-likelihood with L-BFGS. Thresholds stay at 0.5. ECE uses 15 equal-width bins, over every label-wise decision for the multi-label tasks and over the arg-max probability for CAIL2018. Selective prediction abstains when one minus the confidence of the least confident label exceeds a threshold; risk is `1 − Jaccard` (1 when both sets are empty) or 0/1 error on CAIL2018. Paired bootstrap draws 10,000 resamples of test documents. The two-sided p-value is twice the fraction of resamples whose macro-F1 difference has the opposite sign to the observed difference.
8. For a document, the rationale score of paragraph `i` is the sum of label-aware attention over the predicted labels, or over the top-scoring label if that set is empty. The five highest-scoring paragraphs are compared with the silver rationales by precision, recall, and F1 at 5. Multi-label neural baselines trained with `arch=encoder` use binary cross-entropy; the CAIL encoder baseline uses unweighted cross-entropy. They share the optimiser, the schedule, the early-stopping rule, and the seeds.

## Citations

Chalkidis, I., Androutsopoulos, I., and Aletras, N. (2019). Neural legal judgment prediction in English. *Proceedings of ACL 2019*, 4317–4323. https://doi.org/10.18653/v1/P19-1424

Chalkidis, I., Fergadiotis, M., Tsarapatsanis, D., Aletras, N., Androutsopoulos, I., and Malakasiotis, P. (2021). Paragraph-level rationale extraction through regularization: A case study on European Court of Human Rights cases. *Proceedings of NAACL 2021*, 226–241. https://doi.org/10.18653/v1/2021.naacl-main.22

Chalkidis, I., Fergadiotis, M., and Androutsopoulos, I. (2021). MultiEURLEX — A multi-lingual and multi-label legal document classification dataset for zero-shot cross-lingual transfer. *Proceedings of EMNLP 2021*, 6974–6996. https://doi.org/10.18653/v1/2021.emnlp-main.559

Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I., Katz, D. M., and Aletras, N. (2022). LexGLUE: A benchmark dataset for legal language understanding in English. *Proceedings of ACL 2022*, 4310–4330. https://doi.org/10.18653/v1/2022.acl-long.297

Xiao, C., Zhong, H., Guo, Z., Tu, C., Liu, Z., Sun, M., Feng, Y., Han, X., Hu, Z., Wang, H., and Xu, J. (2018). CAIL2018: A large-scale legal dataset for judgment prediction. arXiv:1807.02478. https://doi.org/10.48550/arXiv.1807.02478

Xu, N., Wang, P., Chen, L., Pan, L., Wang, X., and Zhao, J. (2020). Distinguish confusing law articles for legal judgment prediction. *Proceedings of ACL 2020*, 3086–3095. https://doi.org/10.18653/v1/2020.acl-main.280

## License & Contribution Guidelines

This implementation is released so the experiments in the paper can be reproduced and may be used and modified for research if the paper is cited. ECtHR, EUR-LEX, EuroVoc, and CAIL2018 are not redistributed here and remain under the terms set by their publishers. The Convention texts in `data/statutes/ecthr.json` are included only as the label descriptions required by the method. Please send corrections to preprocessing, the label graph, training, or evaluation as pull requests to https://github.com/ARIA-Diff/LK-HiT. Changes should keep the datasets, splits, loss, optimiser, early-stopping rule, decision threshold, and metrics specified above.
