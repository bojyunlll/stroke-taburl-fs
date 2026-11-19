"""
TabuRL-FS analysis of ischemic stroke subtypes (SAD / LAA / CE) from whole-blood DNA methylation (GSE69138).

Pipeline (matches the Methods section of the manuscript):
  1. ANOVA F-test ranks all CpGs; the top 3,000 form the candidate pool.
  2. TabuRL-FS is run independently with LinearSVC, Logistic Regression and SVM (RBF)
     as the wrapped classifier (k = 70 CpGs, 3,000 iterations, 5x5 CV balanced-accuracy fitness).
  3. Consensus features = CpGs selected by at least 2 of the 3 classifiers.
  4. Stratified 70:30 hold-out split (test n = 120: 42 SAD / 40 LAA / 38 CE). The three evaluation
     classifiers are trained on the training set only and scored on the test set
     (accuracy, macro F1 / precision / recall, row-normalized confusion matrix),
     with a label-permutation test (p = (b + 1) / (n_perm + 1)).
  5. Hierarchical clustering heatmap of the consensus CpGs.
  6. Stability: TabuRL-FS is repeated 100 times (seeds 0..99); in each run the CpGs in the
     intersection of the three classifier-specific subsets are recorded and their recurrence counted.

Phases:
  select      run TabuRL-FS for one or more classifiers (steps 1-2); results are cached as JSON
  evaluate    consensus + hold-out evaluation + Table 1 and Figures 2-4 (steps 3-5)
  recurrence  repeated runs and Figure 5 (step 6); resumable, can be split across processes with --runs

By default feature selection uses all 400 samples (--fs_data all), as in the manuscript.
Use --fs_data train to restrict feature selection to the training split.

Examples:
  python run_stroke.py select --clf LinearSVC LogReg SVM_RBF --n_jobs 8
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
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


def holdout_split(y, test_size, seed):
    return train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)


def fs_scope(args):
    return "all" if args.fs_data == "all" else f"train_split{args.split_seed}_test{args.test_size:g}"


def feature_path(args, clf_name, seed, subdir="features"):
    return Path(args.out) / subdir / f"{fs_scope(args)}_{clf_name}_k{args.k}_seed{seed}.json"


def fs_samples(args, X, y):
    """Samples used for feature selection: all samples, or only the training split."""
    if args.fs_data == "all":
        return X, y
    tr, _ = holdout_split(y, args.test_size, args.split_seed)
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

def evaluate_holdout(X, y, tr, te, cols, n_perm, perm_seed):
    """Hold-out metrics for each evaluation classifier plus an empirical permutation p-value."""
    rows, preds = [], {}
    for name, clf in EVAL_MODELS.items():
        pipe = make_pipeline(StandardScaler(), clone(clf)).fit(X[tr][:, cols], y[tr])
        pred = pipe.predict(X[te][:, cols])
        preds[name] = pred
        rows.append(dict(Model=name,
                         Test_accuracy=accuracy_score(y[te], pred),
                         F1_score=f1_score(y[te], pred, average="macro"),
                         Precision=precision_score(y[te], pred, average="macro"),
                         Recall=recall_score(y[te], pred, average="macro")))

    # Permutation test: shuffle all labels, refit on the training split, score on the test split
    rng = np.random.default_rng(perm_seed)
    for row, (name, clf) in zip(rows, EVAL_MODELS.items()):
        null = []
        for _ in range(n_perm):
            yp = rng.permutation(y)
            pipe = make_pipeline(StandardScaler(), clone(clf)).fit(X[tr][:, cols], yp[tr])
            null.append(accuracy_score(yp[te], pipe.predict(X[te][:, cols])))
        b = int(np.sum(np.asarray(null) >= row["Test_accuracy"]))
        row["Permutation_p"] = (b + 1) / (n_perm + 1)
        row["Null_mean_accuracy"] = float(np.mean(null))
    return pd.DataFrame(rows), preds


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
    pvals = ", ".join(f"{r.Model} p={r.Permutation_p:.4f}" for r in table.itertuples())
    ax.text(0.5, -0.18, f"Permutation test: {pvals}", transform=ax.transAxes, ha="center", fontsize=9)
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

def phase_select(args):
    X, y, probes, _ = load_data(args.data)
    X_fs, y_fs = fs_samples(args, X, y)
    for clf_name in args.clf:
        names = get_or_run(feature_path(args, clf_name, args.seed), X_fs, y_fs, probes, clf_name, args.seed, args)
        print(f"{clf_name}: {len(names)} CpGs -> {feature_path(args, clf_name, args.seed)}", flush=True)


def phase_evaluate(args):
    out = Path(args.out)
    X, y, probes, _ = load_data(args.data)
    tr, te = holdout_split(y, args.test_size, args.split_seed)
    print(f"Hold-out split: train {len(tr)} / test {len(te)}, test counts "
          f"{dict(zip(CLASS_NAMES, np.bincount(y[te]).tolist()))}", flush=True)

    selected = {}
    for clf_name in FS_MODELS:
        path = feature_path(args, clf_name, args.seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run: python run_stroke.py select --clf {clf_name}")
        selected[clf_name] = json.loads(path.read_text())["features"]

    votes = Counter(f for s in selected.values() for f in s)
    consensus = sorted(f for f, c in votes.items() if c >= 2)
    print("Selected CpGs: " + ", ".join(f"{c}={len(s)}" for c, s in selected.items())
          + f" | consensus (>=2/3) = {len(consensus)}", flush=True)
    pd.DataFrame([dict(feature=f, n_classifiers=votes[f], in_consensus=votes[f] >= 2,
                       **{c: f in s for c, s in selected.items()})
                  for f in sorted(votes, key=lambda f: (-votes[f], f))]).to_csv(out / "feature_votes.csv", index=False)
    pd.DataFrame({"feature": consensus, "n_classifiers": [votes[f] for f in consensus]}).to_csv(
        out / "consensus_features.csv", index=False)

    lookup = {p: i for i, p in enumerate(probes)}
    cols = np.array([lookup[f] for f in consensus])
    table, preds = evaluate_holdout(X, y, tr, te, cols, args.n_perm, args.perm_seed)
    table.to_csv(out / "table1.csv", index=False)
    print(table.round(4).to_string(index=False), flush=True)

    plot_confusion(y[te], preds, out / "fig2_confusion.png")
    plot_metrics(table, out / "fig3_metrics.png")
    plot_heatmap(X, y, cols, consensus, out / "fig4_heatmap.png")
    print(f"Table 1 and Figures 2-4 written to {out}", flush=True)


def phase_recurrence(args):
    out = Path(args.out)
    start, stop = args.runs
    if not args.plot_only:
        X, y, probes, _ = load_data(args.data)
        X_fs, y_fs = fs_samples(args, X, y)
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
    p.add_argument("phase", choices=["select", "evaluate", "recurrence"])
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
    p.add_argument("--test_size", type=float, default=0.3, help="hold-out test fraction (manuscript: 70:30)")
    p.add_argument("--split_seed", type=int, default=0, help="random_state of the stratified hold-out split")
    p.add_argument("--n_perm", type=int, default=100, help="label permutations for the permutation test")
    p.add_argument("--perm_seed", type=int, default=0)
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
    Path(args.out).mkdir(parents=True, exist_ok=True)

    {"select": phase_select, "evaluate": phase_evaluate, "recurrence": phase_recurrence}[args.phase](args)


if __name__ == "__main__":
    main()
