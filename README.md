# LLM survival reproduction and extensions

Independent implementation of [Davidov et al., arXiv:2506.13593v2](https://arxiv.org/pdf/2506.13593v2).

The completed synthetic experiments compare five calibration methods, neural networks and random forests, known versus estimated censoring probabilities, and a harder hidden-intent mixture. These simulations make no LLM API calls. New research proposals in the accompanying short report have not been implemented or evaluated.

## Included files

- `code/`: implementation, fixed configuration, seven numerical tests, plotting and bilingual report builders.
- `experiment_outputs.zip`: extract to obtain `results/` (7,200 run records, summaries, diagnostics and validation receipts) and `figures/` (seven numerical figures and a representative results table).
- `reference_notes.txt`: differences between the specified paper version and the inspected public implementation.

Fitted model checkpoints are omitted from GitHub because they can be regenerated. They remain in the original local experiment package.

## Reproduce

Use Python 3.12 and run from the repository root:

```bash
python -m zipfile -e experiment_outputs.zip .
python -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python -m pytest -q code/test_methods.py
.venv/bin/python code/run_experiments.py --config code/config.json --out results_rerun
.venv/bin/python code/validate_results.py --results results_rerun --compare results
.venv/bin/python code/plot_results.py --results results_rerun --out figures_rerun
.venv/bin/python code/plot_weight_effect.py --results results_rerun --out figures_rerun
```

On Windows replace `.venv/bin/python` with `.venv\Scripts\python.exe`. The requirements record the versions used for the archived run; matching the supplied environment is recommended for numerical reproduction.

To rebuild reports from the included results and PNG figures:

```bash
.venv/bin/python code/build_report.py
.venv/bin/python code/build_report_en.py
```

These builders recreate the earlier experiment reports; the separate five-page research report also includes proposed future studies.

## Design and interpretation

- Two distributions, two model families, ten budgets and twenty calibration repetitions.
- Per scenario: 40,000 prompts split into 18,000 training, 18,000 calibration and 4,000 test.
- Each training prompt has 500 simulated outputs represented by binomial counts.
- Five methods with known C; four calibrated methods also use learned C.
- Separate C models are fit on independent pilot data using the known distribution family and allocation support.
- Coverage is P(T >= L), with T in {1,2,...}; L=100 concerns the first 99 safe generations on the coverage event.
- Budget constrains expected assigned C. Early-stop cost min(T,C) is reported separately. Training and pilot costs are outside that calibration budget.
- Repetitions keep the data split and fitted models fixed. Shaded bands are sample standard deviations, not confidence intervals or training-seed uncertainty.
- True conditional survival probabilities are evaluation-only.

The results reproduce a qualitative benefit of capping but not every published curve. Naive remains below nominal coverage at larger budgets. The harder scenario is a deliberate model-misspecification stress test with a hypothetical minority group; it makes no empirical claim about real language communities.

## Source and reproducibility

Primary paper: https://arxiv.org/pdf/2506.13593v2

Author code inspected at commit `fd78d544a62cae88da97c056b537c497d0763848`: https://github.com/giladfrid009/llm-survival

This repository contains an independent implementation. The stored validation confirms matching coverage and mean bounds in a same-configuration rerun. It does not establish robustness across independent seeds or real LLM datasets.
