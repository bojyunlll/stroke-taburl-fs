# TabuRL-FS for Ischemic Stroke Subtype Methylation Signatures

Code for the manuscript *"Machine Learning Identifies Stable DNA Methylation Signatures and an
NFYA-Associated Regulatory Network across Ischemic Stroke Subtypes"*.

The pipeline selects CpG features that discriminate small-artery disease (SAD), large-artery
atherosclerosis (LAA) and cardioembolic stroke (CE) from whole-blood DNA methylation
(Illumina HumanMethylation450, GEO [GSE69138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE69138),
n = 400: 141 SAD / 132 LAA / 127 CE) using Tabu reinforcement-learning feature selection (TabuRL-FS).

## Files

| File | Content |
| :--- | :--- |
| `taburl_fs.py` | TabuRL-FS algorithm (tabu search + Transformer policy trained with PPO) |
| `data.py` | Data loading and preprocessing (M-values, per-BeadChip centering) |
| `run_stroke.py` | Full analysis: feature selection, consensus, hold-out evaluation, Table 1, Figures 2-5 |
| `run_all.sh` | Runs the complete pipeline end to end |

## Method summary

1. **Preprocessing** (label-free): drop probes with missing values, clip beta to [0.001, 0.999],
   convert to M-values, and center each probe within each BeadChip.
2. **ANOVA screening**: rank CpGs by one-way ANOVA F-statistic and keep the top 3,000.
3. **TabuRL-FS**, run independently with three wrapped classifiers
   (LinearSVC C=0.01, Logistic Regression C=0.01, SVM-RBF C=1; all `class_weight="balanced"`):
   * subset size k = 70, 3,000 iterations, neighborhood of 100 swap moves, tabu tenure 10
   * fitness = mean balanced accuracy over stratified 5-fold x 5-repeat CV
   * Transformer policy (2 layers, 8 heads, dim 256) trained with PPO (lr 1e-4, update every 20 steps,
     20 epochs, clip 0.2, GAE lambda 0.95, target KL 0.015)
4. **Consensus features**: CpGs selected by at least 2 of the 3 classifiers.
5. **Evaluation**: stratified 80:20 hold-out split (fold 0 of a 5-fold x 2-repeat stratified CV with
   `random_state=2026`; train n = 320, test n = 80: 29 SAD / 26 LAA / 25 CE; `--split_fold` selects another fold).
   LinearSVC, Logistic Regression and SVM-RBF (C=1, balanced) are trained on the training set and
   scored on the test set: accuracy, balanced accuracy, macro F1 / precision / recall with
   percentile-bootstrap 95% CIs (2,000 resamples), per-class precision / sensitivity / specificity / F1,
   row-normalized confusion matrices, and pairwise exact McNemar tests between classifiers.
   Significance by a full-pipeline permutation test (default 1,000 permutations): the labels are
   shuffled and the entire pipeline (ANOVA screening, TabuRL-FS with all three classifiers, consensus,
   hold-out evaluation) is re-run on the shuffled labels, so the null distribution accounts for the
   feature-selection step; p = (b + 1) / (n_perm + 1).
6. **Stability**: TabuRL-FS is repeated 100 times (seeds 0-99); in every run the intersection of the
   three classifier-specific subsets is recorded and the recurrence frequency of each CpG is counted.

As in the manuscript, feature selection (steps 2-4) uses all 400 samples by default.
Pass `--fs_data train` to restrict feature selection to the training split.

## Requirements

Python 3.10+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

A GPU is strongly recommended for the policy network (`--device cuda:0`, about 3 GB of GPU memory
per run). On CPU the PPO update of the Transformer attends over all 3,000 candidate features and can
use 30-55 GB of RAM per run. The CV fitness evaluations run on CPU and are parallelized with `--n_jobs`.

## Data

Place the methylation matrix at `data/stroke_400.csv` (or pass `--data <path>`). Format:

* probes as rows, samples as columns, first column `TargetID` with the CpG probe ID
* one row with `TargetID = infect` holding the subtype label (1 = SAD, 2 = LAA, 3 = CE)
* all other rows contain beta values in [0, 1]
* sample columns are named `<SentrixBarcode>_<Position>` (optionally with a suffix such as `.AVG_Beta`);
  the barcode is used for per-chip centering

The data are available from GEO under accession GSE69138. The preprocessed matrix is cached in
`data/_cache/` after the first run.

## Usage

```bash
# 1. TabuRL-FS with each classifier (can be run as three separate processes)
python run_stroke.py select --clf LinearSVC --n_jobs 8 --device cuda:0
python run_stroke.py select --clf LogReg    --n_jobs 8 --device cuda:0
python run_stroke.py select --clf SVM_RBF   --n_jobs 8 --device cuda:0

# 2. Full-pipeline permutation test (resumable; can be split across processes / GPUs)
python run_stroke.py permutation --perms 0 500    --n_jobs 8 --device cuda:0
python run_stroke.py permutation --perms 500 1000 --n_jobs 8 --device cuda:1

# 3. Consensus features, Table 1 (with permutation p-values) and Figures 2-4
python run_stroke.py evaluate

# 4. 100 repeated runs and Figure 5 (resumable; can be split across processes)
python run_stroke.py recurrence --runs 0 50  --n_jobs 8 --device cuda:0
python run_stroke.py recurrence --runs 50 100 --n_jobs 8 --device cuda:0
python run_stroke.py recurrence --plot_only
```

Or run everything with `bash run_all.sh`. Run `python run_stroke.py -h` for all options.

## Outputs (`results/`)

| File | Content |
| :--- | :--- |
| `features/all_<clf>_k70_seed0.json` | CpGs selected by TabuRL-FS for each classifier |
| `feature_votes.csv`, `consensus_features.csv` | Classifier votes per CpG and the consensus set |
| `table1.csv` | Hold-out accuracy, balanced accuracy, F1, precision, recall, bootstrap 95% CIs and permutation p-values (Table 1) |
| `table1_own_features.csv` | Same as Table 1 when each classifier uses the CpGs selected by TabuRL-FS wrapping that classifier (permutation p-values from the same permutation runs) |
| `per_class_metrics.csv` | Per-class precision, sensitivity, specificity and F1 |
| `model_comparison.csv` | Pairwise classifier differences with bootstrap CIs and exact McNemar tests |
| `permutation/` | Feature sets and hold-out scores of every permutation |
| `fig2_confusion.png` | Row-normalized confusion matrices (Figure 2) |
| `fig3_metrics.png` | Performance comparison (Figure 3) |
| `fig4_heatmap.png` | Hierarchical clustering of consensus CpGs (Figure 4) |
| `recurrence/`, `recurrence_counts.csv` | Per-run selections and recurrence counts |
| `fig5_recurrence.png` | Recurrence analysis over 100 runs (Figure 5) |

## Reproducibility notes

* All random seeds are fixed (`--seed`, `--split_fold`, `--perm_seed`); the CV inside the fitness
  function uses `random_state=42`.
* Results can vary slightly across library versions and hardware (PyTorch numerics).
