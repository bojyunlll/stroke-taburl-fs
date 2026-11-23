"""
TabuRL-FS analysis of ischemic stroke subtypes (SAD / LAA / CE) from whole-blood DNA methylation (GSE69138).

Pipeline (matches the Methods section of the manuscript):
  1. ANOVA F-test ranks all CpGs; the top 3,000 form the candidate pool.
  2. TabuRL-FS is run independently with LinearSVC, Logistic Regression and SVM (RBF)
     as the wrapped classifier (k = 70 CpGs, 3,000 iterations, 5x5 CV balanced-accuracy fitness).
  3. Consensus features = CpGs selected by at least 2 of the 3 classifiers.
  4. Stratified 80:20 hold-out split (fold 0 of a 5-fold x 2 stratified CV, random_state 2026;
     train n = 320, test n = 80: 29 SAD / 26 LAA / 25 CE). The three evaluation
     classifiers are trained on the training set only and scored on the test set: accuracy, balanced
     accuracy, macro F1 / precision / recall with percentile-bootstrap 95% CIs, per-class metrics,
     row-normalized confusion matrices, and pairwise exact McNemar tests between classifiers.
     Repeated stratified CV (5-fold x 10, CpGs fixed) gives a second estimate with Nadeau-Bengio corrected
     95% CIs; every permutation is scored the same way, so the CV estimate has its own permutation p-value.
     Significance is assessed by a full-pipeline permutation test: labels are shuffled and the whole
     pipeline (ANOVA, TabuRL-FS x 3, consensus, evaluation) is re-run; p = (b + 1) / (n_perm + 1).
  5. Hierarchical clustering heatmap of the consensus CpGs.
  6. Stability: TabuRL-FS is repeated 100 times (seeds 0..99); in each run the CpGs in the
     intersection of the three classifier-specific subsets are recorded and their recurrence counted.

Phases:
  select       run TabuRL-FS for one or more classifiers (steps 1-2); results are cached as JSON
  permutation  full-pipeline permutation test (step 4); resumable, split across processes with --perms
  evaluate     consensus + hold-out evaluation + Table 1 and Figures 2-4 (steps 3-5);
               includes permutation p-values for all permutations finished so far
  recurrence   repeated runs and Figure 5 (step 6); resumable, can be split across processes with --runs

By default feature selection uses all 400 samples (--fs_data all), as in the manuscript.
Use --fs_data train to restrict feature selection to the training split.

Examples:
  python run_stroke.py select --clf LinearSVC LogReg SVM_RBF --n_jobs 8
  python run_stroke.py permutation --perms 0 500 --device cuda:0
  python run_stroke.py permutation --perms 500 1000 --device cuda:1
  python run_stroke.py evaluate
  python run_stroke.py recurrence --runs 0 50 --n_jobs 8
  python run_stroke.py recurrence --runs 50 100 --n_jobs 8
  python run_stroke.py recurrence --plot_only
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from sklearn.base import clone
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from scipy.stats import binomtest
from scipy.stats import t as t_dist
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score, recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

import taburl_fs
from data import CLASS_NAMES, load_data

# Classifiers wrapped inside TabuRL-FS (fitness evaluation)
FS_MODELS = {
    "LinearSVC": lambda seed: LinearSVC(C=0.01, dual=False, class_weight="balanced", max_iter=5000, tol=1e-3,
                                        random_state=seed),
    "LogReg": lambda seed: LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000, random_state=seed),
    "SVM_RBF": lambda seed: SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed),
}

# Classifiers used for the final hold-out evaluation (Table 1, Figures 2-3)
EVAL_MODELS = {
    "LinearSVC": LinearSVC(C=1.0, dual=False, class_weight="balanced", max_iter=5000, random_state=42),
    "Logistic Regression": LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42),
}

# TabuRL-FS hyper-parameters used in the manuscript
TABURL_PARAMS = dict(
    neighborhood_size=100, tabu_tenure=10, lr=1e-4, gamma=0.95, gae_lambda=0.95, clip_param=0.2,
    target_kl=0.015, update_every=20, rl_epochs=20, reward_scale=1.0, mlp_moves_ratio=1.0,
    embed_dim=256, nhead=8, num_encoder_layers=2, head_hidden_dim=256, dropout_rate=0.1,
)


# ==============================================================================
# Helpers
# ==============================================================================

def anova_pool(X, y, n):
    """Indices of the top-n features by ANOVA F-statistic."""
    F, _ = f_classif(X, y)
    return np.argsort(-np.nan_to_num(F))[:n]


def holdout_split(y, fold=0):
    """80:20 train/test indices = fold ``fold`` (0-9) of a stratified 5-fold x 2-repeat CV (random_state 2026)."""
    splits = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=2026).split(np.zeros(len(y)), y)
    for i, (tr, te) in enumerate(splits):
        if i == fold:
            return tr, te
    raise ValueError(f"fold must be 0-9, got {fold}")


def fs_scope(args):
    return "all" if args.fs_data == "all" else f"train_fold{args.split_fold}"


def feature_path(args, clf_name, seed, subdir="features"):
    return Path(args.out) / subdir / f"{fs_scope(args)}_{clf_name}_k{args.k}_seed{seed}.json"


def fs_samples(args, X, y, tr):
    """Samples used for feature selection: all samples, or only the training split ``tr``."""
    if args.fs_data == "all":
        return X, y
    return X[tr], y[tr]


def run_taburl(X, y, probes, clf_name, seed, args):
    """ANOVA top-``pool_size`` pool followed by TabuRL-FS; returns the selected probe IDs and run metadata."""
    taburl_fs.seed_everything(seed)
    pool = anova_pool(X, y, args.pool_size)
    model = FS_MODELS[clf_name](seed)
    print(f"\n=== TabuRL-FS {clf_name} k={args.k} seed={seed} n={len(y)} device={args.device} ===", flush=True)
    t0 = time.time()
    sel, fitness, _ = taburl_fs.taburl_fs(
        X[:, pool], y, model, top_k=args.k, max_iterations=args.iters,
        n_jobs=args.n_jobs, device=args.device, **TABURL_PARAMS,
    )
    names = [str(p) for p in probes[pool[sel]]]
    meta = dict(clf=clf_name, k=args.k, seed=seed, cv_fitness=fitness, seconds=round(time.time() - t0, 1),
                pool_size=args.pool_size, iters=args.iters, fs_data=args.fs_data, n_fs_samples=int(len(y)))
    print(f"Done {clf_name} seed={seed}: CV fitness={fitness:.4f}, {len(names)} CpGs, {meta['seconds']:.0f}s",
          flush=True)
    return names, meta


def get_or_run(path, X, y, probes, clf_name, seed, args):
    if path.exists() and not args.force:
        return json.loads(path.read_text())["features"]
    names, meta = run_taburl(X, y, probes, clf_name, seed, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"features": names, "meta": meta}, indent=2))
    return names


# ==============================================================================
# Evaluation
# ==============================================================================

def summary_metrics(y_true, pred):
    return dict(Test_accuracy=accuracy_score(y_true, pred),
                Balanced_accuracy=balanced_accuracy_score(y_true, pred),
                F1_score=f1_score(y_true, pred, average="macro"),
                Precision=precision_score(y_true, pred, average="macro", zero_division=0),
                Recall=recall_score(y_true, pred, average="macro"))


def predict_holdout(X, y, tr, te, cols):
    """Fit every evaluation classifier on the training split and predict the test split.
    With no features (empty consensus) the prediction is the majority class of the training split."""
    preds = {}
    for name, clf in EVAL_MODELS.items():
        if len(cols) == 0:
            preds[name] = np.full(len(te), np.bincount(y[tr]).argmax())
            continue
        pipe = make_pipeline(StandardScaler(), clone(clf)).fit(X[tr][:, cols], y[tr])
        preds[name] = pipe.predict(X[te][:, cols])
    return preds


# Evaluation classifier matched to the classifier wrapped inside TabuRL-FS ("own features" design)
MATCHED_EVAL = {"LinearSVC": "LinearSVC", "LogReg": "Logistic Regression", "SVM_RBF": "SVM (RBF)"}


def predict_own(X, y, tr, te, selected, lookup):
    """Each evaluation classifier on the CpGs selected by TabuRL-FS wrapping the same classifier."""
    preds = {}
    for fs_clf, name in MATCHED_EVAL.items():
        cols = np.array([lookup[f] for f in selected[fs_clf]], dtype=int)
        pipe = make_pipeline(StandardScaler(), clone(EVAL_MODELS[name])).fit(X[tr][:, cols], y[tr])
        preds[name] = pipe.predict(X[te][:, cols])
    return preds


def bootstrap_ci(y_true, preds, n_boot, seed, alpha=0.05):
    """Percentile bootstrap CIs (test samples resampled with replacement) for accuracy and balanced accuracy,
    and for the pairwise differences between classifiers (same resamples for all classifiers)."""
    rng = np.random.default_rng(seed)
    names = list(preds)
    stats = {(n, m): [] for n in names for m in ("acc", "bal")}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        for n in names:
            stats[(n, "acc")].append(accuracy_score(y_true[idx], preds[n][idx]))
            stats[(n, "bal")].append(balanced_accuracy_score(y_true[idx], preds[n][idx]))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    ci = {k: np.percentile(v, q) for k, v in stats.items()}
    diff_ci = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for m in ("acc", "bal"):
                diff_ci[(a, b, m)] = np.percentile(np.subtract(stats[(a, m)], stats[(b, m)]), q)
    return ci, diff_ci


def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact (binomial) McNemar test on the discordant correct/incorrect pairs of two classifiers."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    n01, n10 = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return n01, n10, p


def design_columns(selected, lookup):
    """CpG columns per evaluation classifier for both designs: consensus (>=2/3) and own TabuRL-FS features."""
    consensus, _ = consensus_of(selected)
    ccols = np.array([lookup[f] for f in consensus], dtype=int)
    return {"consensus": {name: ccols for name in EVAL_MODELS},
            "own_features": {name: np.array([lookup[f] for f in selected[fs]], dtype=int)
                             for fs, name in MATCHED_EVAL.items()}}


def repeated_cv(X, y, cols_by_model, n_repeats, strata=None, seed=2026):
    """Repeated stratified 5-fold CV of each evaluation classifier on fixed CpG columns (the first fold is the
    80:20 hold-out split). Folds are stratified on ``strata`` (default ``y``); permutations pass the true labels
    so that every permutation is scored on the same folds. Returns {model: (fold accuracies, fold balanced
    accuracies)}; with no columns the prediction is the majority class of the training folds."""
    strata = y if strata is None else strata
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
                 .split(np.zeros(len(y)), strata))
    out = {}
    for name, cols in cols_by_model.items():
        acc, bal = [], []
        for tr, te in folds:
            if len(cols) == 0:
                pred = np.full(len(te), np.bincount(y[tr]).argmax())
            else:
                pipe = make_pipeline(StandardScaler(), clone(EVAL_MODELS[name])).fit(X[tr][:, cols], y[tr])
                pred = pipe.predict(X[te][:, cols])
            acc.append(accuracy_score(y[te], pred))
            bal.append(balanced_accuracy_score(y[te], pred))
        out[name] = (np.array(acc), np.array(bal))
    return out


def corrected_ci(scores, test_train_ratio=0.25, alpha=0.05):
    """Mean and Nadeau-Bengio corrected t interval for repeated-CV scores; the correction accounts for the
    overlap of training sets across folds (test/train size ratio = 1/4 for 5-fold CV)."""
    j = len(scores)
    half = t_dist.ppf(1 - alpha / 2, j - 1) * np.sqrt((1 / j + test_train_ratio) * np.var(scores, ddof=1))
    return scores.mean(), scores.mean() - half, scores.mean() + half


def per_class_table(y_true, preds):
    rows = []
    for name, pred in preds.items():
        prec, rec, f1, sup = precision_recall_fscore_support(y_true, pred, labels=range(len(CLASS_NAMES)),
                                                             zero_division=0)
        cm = confusion_matrix(y_true, pred, labels=range(len(CLASS_NAMES)))
        for c, cname in enumerate(CLASS_NAMES):
            tn = cm.sum() - cm[c].sum() - cm[:, c].sum() + cm[c, c]
            rows.append(dict(Model=name, Class=cname, N=int(sup[c]), Precision=prec[c],
                             Recall_sensitivity=rec[c], Specificity=tn / (tn + cm[:, c].sum() - cm[c, c]),
                             F1_score=f1[c]))
    return pd.DataFrame(rows)


def plot_confusion(y_te, preds, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (name, pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_te, pred, normalize="true") * 100
        sns.heatmap(cm, annot=True, fmt=".1f", cmap="viridis", vmin=0, vmax=100,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=ax is axes[-1])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle(f"Figure 2. Classification performance (held-out test n={len(y_te)})", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(table, path):
    metrics = ["Test_accuracy", "F1_score", "Precision", "Recall"]
    labels = ["Accuracy", "F1 score", "Precision", "Recall"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    x = np.arange(len(table))
    w = 0.18
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (m, lab, c) in enumerate(zip(metrics, labels, colors)):
        vals = table[m].to_numpy()
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(table["Model"])
    lo = min(0.65, table[metrics].to_numpy().min() - 0.05)
    ax.set_ylim(lo, max(0.85, table[metrics].to_numpy().max() + 0.05))
    ax.set_ylabel("Performance score")
    ax.set_title("Figure 3. Comparative performance")
    if "Permutation_p_accuracy" in table:
        pvals = ", ".join(f"{r.Model} p={r.Permutation_p_accuracy:.4f}" for r in table.itertuples())
        ax.text(0.5, -0.18, f"Full-pipeline permutation test ({int(table.N_permutations.iloc[0])} permutations): "
                f"{pvals}", transform=ax.transAxes, ha="center", fontsize=9)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(X, y, cols, names, path):
    """Z-scored consensus CpGs; CpGs clustered globally, samples grouped by subtype and clustered within subtype."""
    M = X[:, cols].astype(float)
    M = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-8)
    row_order = leaves_list(linkage(pdist(M.T), method="average"))
    sample_order = []
    for c in range(len(CLASS_NAMES)):
        idx = np.where(y == c)[0]
        if len(idx) > 2:
            sample_order.extend(idx[leaves_list(linkage(pdist(M[idx][:, row_order]), method="average"))])
        else:
            sample_order.extend(idx)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(M[np.array(sample_order)][:, row_order].T, cmap="RdBu_r", center=0, xticklabels=False,
                yticklabels=np.asarray(names)[row_order], ax=ax, cbar_kws={"label": "Z-score"})
    subtype_colors = {0: "#4C72B0", 1: "#E891A5", 2: "#55A868"}
    for i, s in enumerate(sample_order):  # subtype color bar above the heatmap
        ax.add_patch(plt.Rectangle((i, -2.5), 1, 2, color=subtype_colors[int(y[s])], clip_on=False))
    ax.set_title(f"Figure 4. Hierarchical clustering of {len(names)} consensus CpGs "
                 f"({' / '.join(f'{n}' for n in CLASS_NAMES)})")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_recurrence(freq, n_runs, path):
    freq = freq.sort_values("runs", ascending=False).reset_index(drop=True)
    freq["rank"] = np.arange(1, len(freq) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.loglog(freq["rank"], freq["runs"], ".", ms=3)
    for _, r in freq.head(10).iterrows():
        ax.annotate(r.feature, (r["rank"], r["runs"]), fontsize=7)
    ax.set_xlabel("Rank")
    ax.set_ylabel(f"Runs selected (of {n_runs})")
    ax.set_title("A. Rank-frequency")

    ax = axes[0, 1]
    edges = [1, 2, 3, 4, 5, 6, 11, 21, 31, 41]
    names = ["1", "2", "3", "4", "5", "6-10", "11-20", "21-30", "31-40", ">40"]
    keep = [i for i, e in enumerate(edges) if e <= n_runs]
    bins = [edges[i] for i in keep] + [n_runs + 1]
    labels = [names[i] for i in keep]
    counts = pd.cut(freq["runs"], bins=bins, right=False, labels=labels).value_counts().reindex(labels).fillna(0)
    ax.bar(range(len(counts)), counts.to_numpy())
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Number of CpGs")
    ax.set_title("B. Frequency distribution")

    ax = axes[1, 0]
    ax.semilogx(freq["rank"], freq["runs"].cumsum() / freq["runs"].sum() * 100)
    ax.set_xlabel("Top N CpGs")
    ax.set_ylabel("Cumulative coverage (%)")
    ax.set_title("C. Cumulative coverage")

    ax = axes[1, 1]
    ax.axis("off")
    top = "\n".join(f"{i + 1:2d}. {r.feature}  {int(r.runs)}/{n_runs}" for i, r in freq.head(10).iterrows())
    ax.text(0.05, 0.95, "D. Top 10\n" + top, va="top", family="monospace", fontsize=10)

    stats = (f"unique={len(freq)}; >=10 runs: {(freq.runs >= 10).sum()}; "
             f"1 run: {(freq.runs == 1).sum()} ({(freq.runs == 1).mean() * 100:.1f}%)")
    fig.suptitle(f"Figure 5. Recurrent CpGs from {n_runs} TabuRL-FS runs\n{stats}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Phases
# ==============================================================================

def consensus_of(selected, min_votes=2):
    votes = Counter(f for s in selected.values() for f in s)
    return sorted(f for f, c in votes.items() if c >= min_votes), votes


def phase_select(args):
    X, y, probes, _ = load_data(args.data)
    tr, _ = holdout_split(y, args.split_fold)
    X_fs, y_fs = fs_samples(args, X, y, tr)
    for clf_name in args.clf:
        names = get_or_run(feature_path(args, clf_name, args.seed), X_fs, y_fs, probes, clf_name, args.seed, args)
        print(f"{clf_name}: {len(names)} CpGs -> {feature_path(args, clf_name, args.seed)}", flush=True)


def permutation_dir(args, i):
    return f"permutation/perm{i:04d}"


def phase_permutation(args):
    """
    Full-pipeline permutation test. For permutation i the labels of all samples are shuffled once
    (seed = perm_seed + i), and the complete pipeline is re-run with the shuffled labels:
    ANOVA pool -> TabuRL-FS with the three classifiers -> consensus -> hold-out evaluation on the same
    train/test split, scored against the same shuffled labels. Because feature selection is repeated on
    the shuffled labels, the null distribution includes any optimism of the feature-selection step.
    Resumable; permutations START..STOP-1 can be distributed over several processes with --perms.
    """
    out = Path(args.out)
    X, y, probes, _ = load_data(args.data)
    tr, te = holdout_split(y, args.split_fold)
    lookup = {p: i for i, p in enumerate(probes)}
    start, stop = args.perms
    for i in range(start, stop):
        score_path = out / permutation_dir(args, i) / f"{fs_scope(args)}_k{args.k}_fold{args.split_fold}_scores.json"
        if score_path.exists() and not args.force:
            continue
        y_perm = np.random.default_rng(args.perm_seed + i).permutation(y)
        X_fs, y_fs = fs_samples(args, X, y_perm, tr)
        selected = {c: get_or_run(feature_path(args, c, args.seed, subdir=permutation_dir(args, i)),
                                  X_fs, y_fs, probes, c, args.seed, args) for c in FS_MODELS}
        consensus, _ = consensus_of(selected)
        cols = np.array([lookup[f] for f in consensus], dtype=int)
        preds = predict_holdout(X, y_perm, tr, te, cols)
        scores = {name: summary_metrics(y_perm[te], pred) for name, pred in preds.items()}
        own = predict_own(X, y_perm, tr, te, selected, lookup)
        own_scores = {name: summary_metrics(y_perm[te], pred) for name, pred in own.items()}
        cv_scores = {}
        for design, cols_by_model in design_columns(selected, lookup).items():
            res = repeated_cv(X, y_perm, cols_by_model, args.cv_repeats, strata=y)
            cv_scores[design] = {n: dict(CV_accuracy=float(a.mean()), CV_balanced_accuracy=float(b.mean()))
                                 for n, (a, b) in res.items()}
        score_path.write_text(json.dumps(dict(perm=i, n_consensus=len(consensus), scores=scores,
                                              own_scores=own_scores, cv_repeats=args.cv_repeats,
                                              cv_scores=cv_scores), indent=2))
        print(f"[permutation {i}] consensus={len(consensus)} | "
              + " | ".join(f"{n}: acc={s['Test_accuracy']:.3f}" for n, s in scores.items())
              + " || own features: " + " | ".join(f"{n}: acc={s['Test_accuracy']:.3f}" for n, s in own_scores.items()),
              flush=True)


def load_permutation_null(args, key="scores"):
    """key = "scores" (consensus design) or "own_scores" (each classifier on its own TabuRL-FS features)."""
    rows = []
    for f in sorted(Path(args.out).glob(f"permutation/perm*/{fs_scope(args)}_k{args.k}_fold{args.split_fold}_scores.json")):
        d = json.loads(f.read_text())
        for name, s in d.get(key, {}).items():
            rows.append(dict(perm=d["perm"], Model=name, n_consensus=d["n_consensus"], **s))
    return pd.DataFrame(rows)


def load_cv_null(args):
    """Repeated-CV scores of every finished permutation (same cv_repeats as this run)."""
    rows = []
    for f in sorted(Path(args.out).glob(f"permutation/perm*/{fs_scope(args)}_k{args.k}_fold{args.split_fold}_scores.json")):
        d = json.loads(f.read_text())
        if d.get("cv_repeats") != args.cv_repeats:
            continue
        for design, models in d.get("cv_scores", {}).items():
            for name, s in models.items():
                rows.append(dict(perm=d["perm"], Design=design, Model=name, **s))
    return pd.DataFrame(rows)


def results_table(y_te, preds, ci, null):
    """Hold-out metrics, bootstrap 95% CIs and (when available) full-pipeline permutation p-values."""
    rows = []
    for name, pred in preds.items():
        row = dict(Model=name, **summary_metrics(y_te, pred))
        row["Accuracy_95CI"] = "{:.3f}-{:.3f}".format(*ci[(name, "acc")])
        row["Balanced_accuracy_95CI"] = "{:.3f}-{:.3f}".format(*ci[(name, "bal")])
        if len(null):
            nm = null[null.Model == name]
            for metric, col in (("Test_accuracy", "Permutation_p_accuracy"),
                                ("Balanced_accuracy", "Permutation_p_balanced_accuracy")):
                b = int(np.sum(nm[metric].to_numpy() >= row[metric]))
                row[col] = (b + 1) / (len(nm) + 1)
            row["N_permutations"] = len(nm)
            row["Null_mean_accuracy"] = nm["Test_accuracy"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def phase_evaluate(args):
    out = Path(args.out)
    X, y, probes, _ = load_data(args.data)
    tr, te = holdout_split(y, args.split_fold)
    print(f"Hold-out split: train {len(tr)} / test {len(te)}, test counts "
          f"{dict(zip(CLASS_NAMES, np.bincount(y[te]).tolist()))}", flush=True)

    selected = {}
    for clf_name in FS_MODELS:
        path = feature_path(args, clf_name, args.seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run: python run_stroke.py select --clf {clf_name}")
        selected[clf_name] = json.loads(path.read_text())["features"]

    consensus, votes = consensus_of(selected)
    print("Selected CpGs: " + ", ".join(f"{c}={len(s)}" for c, s in selected.items())
          + f" | consensus (>=2/3) = {len(consensus)}", flush=True)
    pd.DataFrame([dict(feature=f, n_classifiers=votes[f], in_consensus=votes[f] >= 2,
                       **{c: f in s for c, s in selected.items()})
                  for f in sorted(votes, key=lambda f: (-votes[f], f))]).to_csv(out / "feature_votes.csv", index=False)
    pd.DataFrame({"feature": consensus, "n_classifiers": [votes[f] for f in consensus]}).to_csv(
        out / "consensus_features.csv", index=False)

    lookup = {p: i for i, p in enumerate(probes)}
    cols = np.array([lookup[f] for f in consensus], dtype=int)
    preds = predict_holdout(X, y, tr, te, cols)
    ci, diff_ci = bootstrap_ci(y[te], preds, args.n_boot, args.boot_seed)

    # Table 1: overall metrics, bootstrap 95% CIs and full-pipeline permutation p-values
    table = results_table(y[te], preds, ci, load_permutation_null(args, "scores"))
    table.to_csv(out / "table1.csv", index=False)
    print("Consensus features:\n" + table.round(4).to_string(index=False), flush=True)

    # Same table for the design in which each classifier uses its own TabuRL-FS features
    own = predict_own(X, y, tr, te, selected, lookup)
    own_ci, _ = bootstrap_ci(y[te], own, args.n_boot, args.boot_seed)
    own_table = results_table(y[te], own, own_ci, load_permutation_null(args, "own_scores"))
    own_table.to_csv(out / "table1_own_features.csv", index=False)
    print("Own TabuRL-FS features (classifier matched to the wrapped classifier):\n"
          + own_table.round(4).to_string(index=False), flush=True)
    # Repeated stratified CV (5-fold x cv_repeats) with the selected CpGs fixed, for both designs
    cv_null = load_cv_null(args)
    cv_rows = []
    for design, cols_by_model in design_columns(selected, lookup).items():
        for name, (acc, bal) in repeated_cv(X, y, cols_by_model, args.cv_repeats).items():
            m, lo, hi = corrected_ci(acc)
            mb, lob, hib = corrected_ci(bal)
            row = dict(Design=design, Model=name, CV_accuracy=m, CV_accuracy_SD=acc.std(ddof=1),
                       CV_accuracy_95CI=f"{lo:.3f}-{hi:.3f}", CV_balanced_accuracy=mb,
                       CV_balanced_accuracy_95CI=f"{lob:.3f}-{hib:.3f}", N_folds=len(acc))
            nm = cv_null[(cv_null.Design == design) & (cv_null.Model == name)] if len(cv_null) else cv_null
            if len(nm):
                for metric, val, col in (("CV_accuracy", m, "Permutation_p_CV_accuracy"),
                                         ("CV_balanced_accuracy", mb, "Permutation_p_CV_balanced_accuracy")):
                    row[col] = (int(np.sum(nm[metric].to_numpy() >= val)) + 1) / (len(nm) + 1)
                row["N_permutations"] = len(nm)
                row["Null_mean_CV_accuracy"] = nm["CV_accuracy"].mean()
            cv_rows.append(row)
    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(out / "cv_metrics.csv", index=False)
    print(f"Repeated stratified CV (5-fold x {args.cv_repeats}):\n" + cv_table.round(4).to_string(index=False), flush=True)
    if "N_permutations" not in table:
        print("No permutation results yet: run `python run_stroke.py permutation` and re-run evaluate.", flush=True)

    per_class = per_class_table(y[te], preds)
    per_class.to_csv(out / "per_class_metrics.csv", index=False)

    # Pairwise classifier comparison on the same test samples
    comp, names = [], list(preds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n01, n10, p_mc = mcnemar_exact(y[te], preds[a], preds[b])
            comp.append(dict(Model_A=a, Model_B=b,
                             Accuracy_difference=accuracy_score(y[te], preds[a]) - accuracy_score(y[te], preds[b]),
                             Accuracy_difference_95CI="{:.3f} to {:.3f}".format(*diff_ci[(a, b, "acc")]),
                             Balanced_accuracy_difference_95CI="{:.3f} to {:.3f}".format(*diff_ci[(a, b, "bal")]),
                             A_correct_B_wrong=n01, A_wrong_B_correct=n10, McNemar_exact_p=p_mc))
    pd.DataFrame(comp).to_csv(out / "model_comparison.csv", index=False)
    print(pd.DataFrame(comp).round(4).to_string(index=False), flush=True)

    plot_confusion(y[te], preds, out / "fig2_confusion.png")
    plot_metrics(table, out / "fig3_metrics.png")
    plot_heatmap(X, y, cols, consensus, out / "fig4_heatmap.png")
    print(f"Tables and Figures 2-4 written to {out}", flush=True)


def phase_recurrence(args):
    out = Path(args.out)
    start, stop = args.runs
    if not args.plot_only:
        X, y, probes, _ = load_data(args.data)
        X_fs, y_fs = fs_samples(args, X, y, holdout_split(y, args.split_fold)[0])
        for run in range(start, stop):
            sets = [set(get_or_run(feature_path(args, c, run, subdir="recurrence"), X_fs, y_fs, probes, c, run, args))
                    for c in FS_MODELS]
            print(f"[recurrence] run {run}: |intersection| = {len(set.intersection(*sets))}", flush=True)

    # Aggregate over every completed run found on disk (each CpG counted at most once per run)
    counts, done = Counter(), []
    for run in range(args.n_runs):
        paths = [feature_path(args, c, run, subdir="recurrence") for c in FS_MODELS]
        if all(p.exists() for p in paths):
            done.append(run)
            counts.update(set.intersection(*[set(json.loads(p.read_text())["features"]) for p in paths]))
    if not counts:
        print("No completed recurrence runs yet.")
        return
    freq = pd.DataFrame({"feature": list(counts), "runs": list(counts.values())}).sort_values(
        ["runs", "feature"], ascending=[False, True])
    freq.to_csv(out / "recurrence_counts.csv", index=False)
    print(f"Completed runs: {len(done)}/{args.n_runs} | unique CpGs: {len(freq)} | "
          f">=10 runs: {(freq.runs >= 10).sum()}", flush=True)
    if len(done) == args.n_runs:
        plot_recurrence(freq, args.n_runs, out / "fig5_recurrence.png")
        print(f"Figure 5 written to {out / 'fig5_recurrence.png'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["select", "evaluate", "permutation", "recurrence"])
    p.add_argument("--data", default="data/stroke_400.csv", help="input CSV (see data.py for the format)")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--clf", nargs="+", default=list(FS_MODELS), choices=list(FS_MODELS),
                   help="[select] classifiers to run TabuRL-FS with")
    p.add_argument("--k", type=int, default=70, help="number of CpGs selected by TabuRL-FS")
    p.add_argument("--pool_size", type=int, default=3000, help="ANOVA candidate pool size")
    p.add_argument("--iters", type=int, default=3000, help="TabuRL-FS iterations")
    p.add_argument("--seed", type=int, default=0, help="TabuRL-FS seed for the main analysis")
    p.add_argument("--fs_data", choices=["all", "train"], default="all",
                   help="samples used for feature selection: all 400 (manuscript) or the training split only")
    p.add_argument("--split_fold", type=int, default=0,
                   help="80:20 hold-out = this fold (0-9) of a stratified 5-fold x 2-repeat CV (random_state 2026)")
    p.add_argument("--n_perm", type=int, default=1000, help="[permutation] total number of label permutations")
    p.add_argument("--perms", type=int, nargs=2, metavar=("START", "STOP"), default=None,
                   help="[permutation] only run permutations START..STOP-1 in this process")
    p.add_argument("--perm_seed", type=int, default=0, help="permutation i shuffles labels with seed perm_seed + i")
    p.add_argument("--n_boot", type=int, default=2000, help="[evaluate] bootstrap resamples of the test set")
    p.add_argument("--boot_seed", type=int, default=0)
    p.add_argument("--cv_repeats", type=int, default=10,
                   help="[evaluate/permutation] repeats of the stratified 5-fold CV (5 x 10 = 50 folds)")
    p.add_argument("--n_runs", type=int, default=100, help="[recurrence] total number of repeated runs")
    p.add_argument("--runs", type=int, nargs=2, metavar=("START", "STOP"), default=None,
                   help="[recurrence] only run seeds START..STOP-1 in this process")
    p.add_argument("--plot_only", action="store_true", help="[recurrence] only aggregate finished runs and plot")
    p.add_argument("--device", default="cpu", help="device for the policy network, e.g. cpu or cuda:0")
    p.add_argument("--n_jobs", type=int, default=8, help="parallel workers for fitness evaluation")
    p.add_argument("--force", action="store_true", help="re-run TabuRL-FS even if cached results exist")
    args = p.parse_args()
    if args.runs is None:
        args.runs = (0, args.n_runs)
    if args.perms is None:
        args.perms = (0, args.n_perm)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    {"select": phase_select, "evaluate": phase_evaluate, "permutation": phase_permutation,
     "recurrence": phase_recurrence}[args.phase](args)


if __name__ == "__main__":
    main()
