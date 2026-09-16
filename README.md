# LK-HiT

Code for *Statute-aware hierarchical transformers for calibrated multi-label
legal judgment prediction across jurisdictions*. The package covers the four
tasks in the paper (ECtHR-A/B, EUR-LEX, CAIL2018), the LK-HiT model, the nine
baselines, the seven ablations, and the evaluation used for discrimination,
long-tail tiers, calibration, selective prediction and rationale agreement.

## Layout

```
configs/            composed YAML: data / models / ablations / experiments
lkhit/              installable package
  data/             LexGLUE, ecthr_cases, CAIL2018, segmentation, label graph
  models/           LK-HiT and the baselines
  engine/           AdamW trainer, evaluator, checkpoint I/O, label knowledge
  cli/              train, evaluate, predict, prepare_cail, aggregate, …
resources/          Convention article texts, section map, Criminal Law chapters,
                    zero-shot prompts
scripts/            seed sweeps
tests/              config, loss, metric and module tests
data/               raw and processed corpora (not versioned)
runs/               one directory per seed: config, log, checkpoints, predictions
```

An experiment file lists the fragments it is built from:

```yaml
defaults:
  - data/ecthr_a
  - models/lkhit
  - train/default
run:
  name: lkhit
```

Command-line overrides use dotted keys (`train.batch_size=2`).

## Setup

```bash
conda create -n lkhit python=3.11 -y && conda activate lkhit
pip install -r requirements.txt
pip install -e .
```

Optional extras: `pip install -e ".[cail,eurovoc,test]"`. The vLLM extra
(`llm`) is only needed for the zero-shot Qwen baseline and is Linux-only in
`requirements.txt`.

A single 48 GB GPU (RTX A6000) was used for the paper. 24 GB is enough for
the hierarchical models with `train.batch_size: 4` and
`train.grad_accumulation: 4` (effective batch 16).

## Data

**ECtHR-A / ECtHR-B / EUR-LEX** are downloaded on first use from
`coastalcph/lex_glue`. Silver rationales for ECtHR-A come from
`AUEB-NLP/ecthr_cases`. Nothing has to be placed by hand.

**CAIL2018.** Download `CAIL2018_ALL_DATA.zip` and put
`exercise_contest/data_train.json` and `data_test.json` under
`data/raw/cail2018/`. Then:

```bash
python -m lkhit.cli.prepare_cail
```

That keeps single-article cases, drops articles with fewer than 100 training
cases (103 remain), and carves a stratified 10% development split. If you
have a `{article: text}` file of the Criminal Law, put it at
`data/raw/cail2018/criminal_law_articles.json` so it is copied next to the
splits.

**Label descriptions.** Convention article texts ship in
`resources/echr_articles.json`. EuroVoc descriptions are built from a SKOS
export placed under `data/raw/eurovoc/`:

```bash
python -m lkhit.cli.build_label_descriptions --task ecthr_a
python -m lkhit.cli.build_label_descriptions --task eurlex --eurovoc-skos data/raw/eurovoc
python -m lkhit.cli.build_label_graph --config configs/experiments/ecthr_a/lkhit.yaml
```

The label graph (positive-PMI co-occurrence plus statutory-structure edges)
is also built automatically the first time a knowledge-using model is trained.

## Training

```bash
python -m lkhit.cli.train --config configs/experiments/ecthr_a/lkhit.yaml --seed 1
python -m lkhit.cli.train --config configs/experiments/eurlex/hier_legal_bert_lsan.yaml --seed 3
python -m lkhit.cli.train --config configs/experiments/cail2018/lkhit_no_label_knowledge.yaml --seed 2
```

The paper uses seeds 1–5. On Windows or a cluster, the same sweep is:

```bash
python scripts/run_sweep.py --suite main
python scripts/run_sweep.py --suite ablations
python scripts/run_sweep.py --task ecthr_a --model lkhit --seeds 1 2 3 4 5
python scripts/run_sweep.py --suite main --dry-run
```

`scripts/run_main.sh` and `scripts/run_ablations.sh` wrap those two suites.
Each run writes `runs/<task>/<name>/seed<k>/` with `config.yaml`,
`train.log`, `metrics.jsonl`, `checkpoints/{best,last}.pt`,
`temperature.json`, and the development/test predictions.

Training already fits a temperature on the development split and evaluates
the selected checkpoint on the test split. Pass `--no-test` to stop after
calibration, or `--overwrite` to retrain when `best.pt` is already there.

## Evaluation

```bash
python -m lkhit.cli.evaluate --run runs/ecthr_a/lkhit/seed1 --split test
python -m lkhit.cli.evaluate --run runs/ecthr_a/lkhit/seed1 --split test --rationales
python -m lkhit.cli.predict  --run runs/ecthr_a/lkhit/seed1 --input my_cases.jsonl --output scored.jsonl
python -m lkhit.cli.aggregate --runs runs/ecthr_a/lkhit --split test
python -m lkhit.cli.aggregate --compare runs/ecthr_a/lkhit runs/ecthr_a/hier_legal_bert_lsan --split test
```

`evaluate` writes `predictions/<split>.npz` (logits, raw and temperature-scaled
probabilities, labels, uncertainty) and `metrics/<split>.json`
(discrimination, calibration, risk-coverage, per-label and tier scores).
`--rationales` adds top-5 paragraph agreement with the ECtHR-A silver
rationales. `aggregate` reports mean ± s.d. over seeds; `--compare` runs the
paired bootstrap on seed-averaged probabilities.

The zero-shot LLM baseline is served with vLLM:

```bash
python -m lkhit.cli.zero_shot_llm --task ecthr_a --model Qwen/Qwen2.5-7B-Instruct --split test
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests load every experiment YAML, check the losses and metrics against
the paper definitions, and run the document transformer, GCN and
label-aware attention on small random tensors. They do not download corpora
or start training.

## Citation

See the manuscript for the full reference list. LexGLUE, ecthr_cases and
CAIL2018 must be cited when those corpora are used. Convention article
texts in `resources/echr_articles.json` follow the official English text of
the Convention for the Protection of Human Rights and Fundamental Freedoms
and Protocol No. 1.
