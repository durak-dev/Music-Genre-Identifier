import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             precision_recall_fscore_support, ConfusionMatrixDisplay,
                             roc_curve, auc)


def cooccurrence_matrix_generator(
    data: pd.DataFrame | pl.DataFrame | list,
    text_col: str = "cleaned_lyrics",
    chunk_size: int = 5000,
    as_polars: bool | None = None,
) -> pl.DataFrame | pd.DataFrame:
    """Computes full vocabulary co-occurrences without hitting SciPy matmul memory errors

    by chunking the matrix multiplication.
    """
    is_polars_input = isinstance(data, (pl.DataFrame, pl.Series))

    # 1. Normalize input
    if isinstance(data, pl.DataFrame):
        sentences = data[text_col].to_list()
    elif isinstance(data, pd.DataFrame):
        sentences = data[text_col].tolist()
    elif isinstance(data, (pl.Series, pd.Series)):
        sentences = data.to_list() if hasattr(data, "to_list") else data.tolist()
    else:
        sentences = data

    return_polars = is_polars_input if as_polars is None else as_polars

    # 2. Vectorize full vocabulary
    vec = CountVectorizer(
        analyzer=lambda x: list(set(x)) if isinstance(x, (list, set)) else x.split(),
        binary=True,
    )
    A = vec.fit_transform(tqdm(sentences, desc="Vectorizing vocabulary"))
    words = vec.get_feature_names_out()
    n_words = len(words)

    # Convert A to CSC format for fast column slicing
    A = A.tocsc()
    AT = A.T.tocsr()

    rows_list = []
    cols_list = []
    data_list = []

    # 3. Multiply in chunks to stay well below RAM limits
    for start in tqdm(
        range(0, n_words, chunk_size), desc="Computing pairs in blocks"
    ):
        end = min(start + chunk_size, n_words)

        # Slice just this batch of words against the entire corpus
        chunk_AT = AT[start:end]
        C_chunk = (chunk_AT @ A).tocoo()

        # Remove self-co-occurrences (diagonal where global row == col)
        global_rows = C_chunk.row + start
        mask = global_rows != C_chunk.col

        if np.any(mask):
            rows_list.append(global_rows[mask].astype(np.uint32))
            cols_list.append(C_chunk.col[mask].astype(np.uint32))
            data_list.append(C_chunk.data[mask].astype(np.float32))

    # Free memory before concatenation
    del A, AT

    print("Assembling final co-occurrence table...")
    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_counts = np.concatenate(data_list)
    del rows_list, cols_list, data_list

    # 4. Build Polars Categorical DataFrame by indexing into the vocabulary
    vocab = pl.Series("vocab", words, dtype=pl.Categorical)

    df_pl = pl.DataFrame(
        {
            "word_1": vocab.gather(all_rows).alias("word_1"),
            "word_2": vocab.gather(all_cols).alias("word_2"),
            "count": all_counts,
        }
    )

    if return_polars:
        return df_pl
    return df_pl.to_pandas()


def song_vector(tokens, vectors, index, idf=None):
    ids = [index[t] for t in tokens if t in index]
    if not ids:
        return np.zeros(vectors.shape[1], dtype=np.float32)
    if idf is None:
        return vectors[ids].mean(axis=0)
    weights = np.array([idf[t] for t in tokens if t in index], dtype=np.float32)
    return np.average(vectors[ids], axis=0, weights=weights)


def save_glove_csv(X_glove, y, path, label_name="genre"):
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    y_arr = y_arr.ravel()                      # flattens a one column frame to 1D

    assert len(X_glove) == len(y_arr), "features and labels have different lengths"

    cols = [f"dim_{k}" for k in range(X_glove.shape[1])]
    df = pl.DataFrame(X_glove, schema=cols, orient="row")
    df = df.with_columns(pl.Series(label_name, y_arr))
    df.write_csv(path)
    return df


def diagnose(y_true, P, classes, pred=None, name="model", train_accuracy=None, save_path=None, fig=None):
    """Print the headline scores and draw four diagnostic plots for any classifier.

    y_true  : true labels, either genre names or integer ids that index into `classes`
    P       : (n_songs, n_classes) probabilities, columns in the same order as `classes`
              (for a transformer, apply a softmax to the logits first)
    classes : genre names, e.g. rfc.classes_ or le.classes_
    pred    : optional predicted labels; defaults to the most probable class
    fig     : optional (sub)figure to draw into, e.g. one cell of fig.subfigures(...). Then nothing is shown or
              saved here, the caller does that (this is how diagnose_runs puts several runs in a grid)
    """
    classes = np.asarray(classes)
    P = np.asarray(P)
    y_true = np.asarray(y_true).ravel()
    if np.issubdtype(y_true.dtype, np.integer):
        y_true = classes[y_true]
    assert P.shape == (len(y_true), len(classes)), f"P is {P.shape}, expected {(len(y_true), len(classes))}"
    pred = classes[P.argmax(1)] if pred is None else np.asarray(pred).ravel()

    # headline numbers, compared with always guessing the biggest genre
    majority = pd.Series(y_true).value_counts(normalize=True).max()
    if train_accuracy is not None:
        print(f"train accuracy: {train_accuracy:.3f}")
    print(f"val accuracy:   {accuracy_score(y_true, pred):.3f}  (always guessing the biggest genre: {majority:.3f})")
    print(f"macro F1:       {f1_score(y_true, pred, average='macro'):.3f}")
    print(f"weighted F1:    {f1_score(y_true, pred, average='weighted'):.3f}")
    print(classification_report(y_true, pred, zero_division=0))

    prec, rec, f1, support = precision_recall_fscore_support(y_true, pred, labels=classes, zero_division=0)
    x = np.arange(len(classes))

    own_fig = fig is None
    if own_fig:
        fig = plt.figure(figsize=(14, 11), layout="constrained")
    axes = fig.subplots(2, 2)
    fig.suptitle(name, fontsize=14)

    # 1. confusion matrix: each row sums to 1, so the diagonal is the recall per genre
    ConfusionMatrixDisplay.from_predictions(y_true, pred, labels=classes, normalize="true",
                                            values_format=".2f", cmap="Blues", colorbar=False,
                                            xticks_rotation=45, ax=axes[0, 0])
    axes[0, 0].set_title("Confusion matrix (rows = true genre)")

    # 2. precision / recall / F1 per genre, with the number of validation songs
    ax = axes[0, 1]
    ax.bar(x - 0.27, prec, 0.27, label="precision")
    ax.bar(x, rec, 0.27, label="recall")
    ax.bar(x + 0.27, f1, 0.27, label="F1")
    ax.set_xticks(x, [f"{c}\n(n={n})" for c, n in zip(classes, support)])
    ax.set_ylim(0, 1)
    ax.set_title("Scores per genre")
    ax.legend()

    # 3. how often each genre really occurs vs. how often the model predicts it
    ax = axes[1, 0]
    true_share = support / support.sum()
    pred_share = pd.Series(pred).value_counts().reindex(classes, fill_value=0).values / len(pred)
    ax.bar(x - 0.2, true_share, 0.4, label="true")
    ax.bar(x + 0.2, pred_share, 0.4, label="predicted")
    ax.set_xticks(x, classes)
    ax.set_ylabel("share of songs")
    ax.set_title("True vs predicted genre share")
    ax.legend()

    # 4. one vs rest ROC curves
    ax = axes[1, 1]
    Y = label_binarize(y_true, classes=classes)
    grid = np.linspace(0, 1, 200)
    tprs = []
    for i, genre in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y[:, i], P[:, i])
        tprs.append(np.interp(grid, fpr, tpr))
        ax.plot(fpr, tpr, lw=1, alpha=0.7, label=f"{genre} (AUC {auc(fpr, tpr):.2f})")
    macro = np.mean(tprs, axis=0)
    ax.plot(grid, macro, "k", lw=2.5, label=f"macro average (AUC {auc(grid, macro):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One vs rest ROC")
    ax.legend(fontsize=8, loc="lower right")

    if own_fig:
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()

    return {"accuracy": accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, average="macro"),
            "weighted_f1": f1_score(y_true, pred, average="weighted")}


# ---------- classic models: fitting, saving, evaluating, comparing, tuning ----------
# A "run" is the dict {"proba": validation probabilities, "train_accuracy": ..., "fit_seconds": ...} that fit_run
# returns. The dicts of several runs, keyed by a name or a tuple like (features, model), go into the functions below.

def make_models(seed=80085, balanced=True):
    """Fresh, unfitted classic models, so that every feature set gets its own copy."""
    weight = "balanced" if balanced else None
    return {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, min_samples_leaf=50, l2_regularization=0.1,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
            class_weight=weight, random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample" if balanced else None,
            n_jobs=-1, random_state=seed),
        "NaiveBayes": GaussianNB(),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=100, class_weight=weight, random_state=seed)),
    }


def make_forest(config, n_trees, seed=80085, balanced=True):
    """A random forest for one Hyperband configuration. warm_start lets it grow more trees later."""
    return RandomForestClassifier(n_estimators=n_trees, max_features="sqrt", warm_start=True,
                                  oob_score=config["bootstrap"],
                                  class_weight="balanced_subsample" if balanced else None,
                                  n_jobs=-1, random_state=seed, **config)


def fit_run(model, X_train, y_train, X_val, classes, target=None, name=""):
    """Fit one model and return its run: validation probabilities, training accuracy and fit time.

    target : the labels to fit on, if they are not y_train (the 0/1 genre columns of a classifier chain)
    The probabilities are scaled to sum to one per song, which the chain's per genre probabilities need.
    """
    start = time.time()
    model.fit(X_train, y_train if target is None else target)
    fit_seconds = round(time.time() - start, 1)
    if target is None:
        assert list(model.classes_) == list(classes), "class order differs from the genre list"
    P_train, P_val = [P / P.sum(axis=1, keepdims=True).clip(min=1e-12)
                      for P in (model.predict_proba(X_train), model.predict_proba(X_val))]
    if name:
        print(f"{name:40} fitted in {fit_seconds:>6.1f}s")
    return {"proba": P_val, "train_accuracy": float(np.mean(np.asarray(classes)[P_train.argmax(axis=1)] == y_train)),
            "fit_seconds": fit_seconds}


def save_runs(runs, proba_path, meta_path, importances=None):
    """Save the runs: probabilities in a compressed .npz, scalars in a .json (far smaller than the fitted models).

    importances : optional {name: array} that is stored next to the probabilities
    """
    names = {key: "__".join(key) if isinstance(key, tuple) else key for key in runs}
    arrays = {names[key]: run["proba"] for key, run in runs.items()}
    arrays.update({f"importance__{name}": values for name, values in (importances or {}).items()})
    Path(proba_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(proba_path, **arrays)
    meta = {names[key]: {"train_accuracy": run["train_accuracy"], "fit_seconds": run["fit_seconds"]}
            for key, run in runs.items()}
    Path(meta_path).write_text(json.dumps(meta, indent=2))


def load_runs(proba_path, meta_path):
    """Load what save_runs wrote. Returns (runs, importances)."""
    arrays = np.load(proba_path)
    runs = {}
    for name, scalars in json.loads(Path(meta_path).read_text()).items():
        runs[tuple(name.split("__")) if "__" in name else name] = {"proba": arrays[name], **scalars}
    importances = {k.split("__")[1]: arrays[k] for k in arrays.files if k.startswith("importance__")}
    return runs, importances


def _run_parts(key, labels=None):
    """The parts of a run key (a name or a tuple of names), shown with their labels if they have one."""
    return [(labels or {}).get(part, part) for part in (key if isinstance(key, tuple) else (key,))]


def diagnose_runs(y_true, runs, classes, labels=None, cols=1, save_path=None):
    """diagnose for every run of a dict of runs, titled with the key of the run (parts joined by " | ").

    cols : number of runs next to each other. 1 draws one figure per run, 2 puts two runs side by side,
           4 puts four in a row, and so on. With cols > 1 all runs share one figure, filled row by row
           (e.g. 4 runs with cols=2 make a 2 x 2 grid of diagnoses)
    save_path : optional file for the grid figure (only used when cols > 1)
    """
    if cols <= 1:
        for key, run in runs.items():
            diagnose(y_true, run["proba"], classes, name=" | ".join(_run_parts(key, labels)),
                     train_accuracy=run["train_accuracy"])
        return

    n = len(runs)
    rows = -(-n // cols)   # ceil
    fig = plt.figure(figsize=(7 * cols, 5.5 * rows), layout="constrained")
    cells = np.atleast_1d(fig.subfigures(rows, cols)).ravel()
    for cell, (key, run) in zip(cells, runs.items()):
        diagnose(y_true, run["proba"], classes, name=" | ".join(_run_parts(key, labels)),
                 train_accuracy=run["train_accuracy"], fig=cell)
    for cell in cells[n:]:   # empty cells of the last row
        cell.set_visible(False)
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def _plot_compare(names, scores, majority):
    """Bar charts of accuracy and macro F1. names: one tuple of display parts per run, scores: matching dicts."""
    metrics = [("accuracy", "Validation accuracy"), ("macro_f1", "Validation macro F1")]
    hues = list(dict.fromkeys(n[0] for n in names)) if all(len(n) == 2 for n in names) else []

    if len(hues) != 2:   # no head to head: one bar per run
        fig, axes = plt.subplots(1, 2, figsize=(14, max(3, 0.5 * len(names) + 1.5)))
        for ax, (metric, title) in zip(axes, metrics):
            bars = ax.barh([" | ".join(n) for n in names], [row[metric] for row in scores])
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
            ax.invert_yaxis()   # the first run on top
            if metric == "accuracy":
                ax.axvline(majority, color="gray", ls="--", label="always the biggest genre")
                ax.legend()
            ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return

    groups = list(dict.fromkeys(n[1] for n in names))
    value = dict(zip(names, scores))
    colors = ["#2b6cb0", "#ed8936"]
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (metric, title) in zip(axes, metrics):
        a = np.array([value[(hues[0], g)][metric] for g in groups])
        b = np.array([value[(hues[1], g)][metric] for g in groups])
        for k, (vals, rival, hue) in enumerate([(a, b, hues[0]), (b, a, hues[1])]):
            bars = ax.bar(x + (k - 0.5) * 0.4, vals, 0.38, color=colors[k], label=hue)
            for bar, v, r in zip(bars, vals, rival):   # the winner of every pair is solid and outlined, the loser faded
                bar.set_alpha(1 if v >= r else 0.5)
                if v >= r:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(2.2)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        gain = b - a
        for xi, g, top in zip(x, gain, np.maximum(a, b)):
            tone = "#2f855a" if g >= 0 else "#c53030"
            ax.text(xi, top + 0.075 * max(a.max(), b.max()), f"{'▲' if g >= 0 else '▼'} {abs(g):.3f}", ha="center",
                    fontsize=11, fontweight="bold", color=tone, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=tone))
        if metric == "accuracy":
            ax.axhline(majority, color="gray", ls="--", lw=1, label="always the biggest genre")
        ax.set_ylim(0, max(a.max(), b.max()) * 1.25)
        ax.set_xticks(x, groups, rotation=15)
        ax.set_title(f"{title}\n{hues[1]} wins {int((gain > 0).sum())} of {len(groups)}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9, frameon=False)
    fig.text(0.47, 0.985, hues[0], color=colors[0], ha="right", va="top", fontsize=18, fontweight="bold")
    fig.text(0.50, 0.985, "VS", ha="center", va="top", fontsize=14, fontweight="bold", color="gray")
    fig.text(0.53, 0.985, hues[1], color=colors[1], ha="left", va="top", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.925, "▲ / ▼ = change from the first to the second, the outlined bar wins", ha="center",
             fontsize=9, color="gray")
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()


def compare(y_true, runs, classes, labels=None, key_names=("run",), plot=True):
    """Compare several runs on the same validation songs, like diagnose does for a single run.

    Shows a table with the scores of every run (best macro F1 first), a table with the F1 per genre and, with
    plot=True, bar charts of accuracy and macro F1 (in the order of the runs) next to the accuracy of always
    guessing the biggest genre. If the keys are (A, B) pairs with exactly two different A, like (features, model),
    the chart is a head to head: the two A side by side for every B, the winner outlined and the gain of the
    second A over the first on top.

    runs      : {key: run}, the runs of fit_run or load_runs
    labels    : optional {part of a key: text to show instead}
    key_names : the column names for the parts of the key, e.g. ("features", "model") for tuple keys
    Returns (scores, f1_per_genre), both polars DataFrames in the same order, one row per run.
    """
    from IPython.display import display

    classes = np.asarray(classes)
    y_true = np.asarray(y_true).ravel()
    rows, f1_rows = [], []
    for key, run in runs.items():
        names = dict(zip(key_names, _run_parts(key, labels)))
        pred = classes[run["proba"].argmax(axis=1)]
        f1 = precision_recall_fscore_support(y_true, pred, labels=classes, zero_division=0)[2]
        rows.append({**names, "train_accuracy": run["train_accuracy"], "accuracy": accuracy_score(y_true, pred),
                     "macro_f1": f1_score(y_true, pred, average="macro"),
                     "weighted_f1": f1_score(y_true, pred, average="weighted"), "fit_seconds": run["fit_seconds"]})
        f1_rows.append({**names, **{str(c): float(v) for c, v in zip(classes, f1)}})
    order = np.argsort([-row["macro_f1"] for row in rows])
    scores = pl.DataFrame([rows[i] for i in order])
    f1_per_genre = pl.DataFrame([f1_rows[i] for i in order])
    display(scores.with_columns(pl.col(pl.Float64).round(3)))
    display(f1_per_genre.with_columns(pl.col(pl.Float64).round(3)))

    if plot:   # the chart keeps the order of the runs, the tables above are sorted by macro F1
        _plot_compare([tuple(str(row[name]) for name in key_names) for row in rows], rows,
                      majority=max(np.mean(y_true == c) for c in classes))
    return scores, f1_per_genre


def make_hgb(config, n_iter, seed=80085, balanced=True):
    """A histogram gradient boosting model for one Hyperband configuration, with the iterations (trees per genre) as
    the budget. No early stopping, so the budget alone decides how long it boosts. warm_start lets it continue."""
    return HistGradientBoostingClassifier(max_iter=n_iter, early_stopping=False, warm_start=True,
                                          class_weight="balanced" if balanced else None, random_state=seed, **config)


def forest_oob(model, config):
    """Extra Hyperband column for random forests: the out-of-bag accuracy (bootstrap forests only)."""
    return {"oob_accuracy": model.oob_score_ if config["bootstrap"] else None}


def hyperband(build, search_space, X_fit, y_fit, X_hold, y_hold, max_budget=300, eta=3, min_budget=10,
              seed=80085, budget_param="n_estimators", budget_name="trees", extra=None):
    """Hyperband (Li et al., 2017) for any scikit-learn style model, with one parameter as the budget.

    Many random configurations start with a small budget, after every round the best 1/eta get eta times more
    (a model with warm_start=True keeps what it has and only grows, any other model is refitted). Every bracket
    starts with a different mix of many short and few long runs. Configurations are scored by the macro F1 on the
    holdout (X_hold, y_hold).

    build        : function (config, budget) -> unfitted model, e.g. lambda c, n: make_forest(c, n, SEED)
    search_space : {parameter: list of values}, one random value per parameter makes a configuration. The budget
                   parameter does not belong in it
    budget_param : the parameter of the model that the budget sets, e.g. "n_estimators" or "max_iter"
    budget_name  : the name of the budget column in the log
    extra        : optional function (fitted model, config) -> dict of more columns for the log
    Returns a polars DataFrame with one row per evaluation: configuration, bracket, rung, budget, the configuration,
    macro F1 and accuracy on the holdout, the extra columns and the fit time.
    """
    rng = np.random.default_rng(seed)
    s_max = int(np.log(max_budget / min_budget) / np.log(eta) + 1e-9)
    total = (s_max + 1) * max_budget
    rows, config_id = [], 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
        for s in range(s_max, -1, -1):   # one bracket per s: from many short runs (s = s_max) to a few full ones (s = 0)
            n_configs = int(np.ceil(total / max_budget * eta ** s / (s + 1)))
            alive = []                   # [config_id, config, model] of the configurations still in this bracket
            for _ in range(n_configs):
                config = {name: values[rng.integers(len(values))] for name, values in search_space.items()}
                alive.append([config_id, config, None])
                config_id += 1
            for rung in range(s + 1):
                budget = int(round(max_budget * eta ** (rung - s)))
                scored = []
                for cid, config, model in alive:
                    if model is not None and model.get_params().get("warm_start"):
                        model.set_params(**{budget_param: budget})   # continue where it stopped
                    else:
                        model = build(config, budget)
                    start = time.time()
                    model.fit(X_fit, y_fit)
                    seconds = round(time.time() - start, 1)
                    pred = model.predict(X_hold)
                    row = {"config_id": cid, "bracket": s, "rung": rung, budget_name: budget, **config,
                           "macro_f1": f1_score(y_hold, pred, average="macro"),
                           "accuracy": accuracy_score(y_hold, pred),
                           **(extra(model, config) if extra else {}),
                           "fit_seconds": seconds}
                    rows.append(row)
                    scored.append((row["macro_f1"], cid, config, model))
                scored.sort(key=lambda item: -item[0])
                keep = max(int(len(scored) / eta), 1)
                alive = [[cid, config, model] for _, cid, config, model in scored[:keep]]
    return pl.DataFrame(rows)


def _fmt_value(v):
    """Short tick label for a parameter value (1e-06 instead of 0.000001)."""
    if isinstance(v, float) and v != 0 and abs(v) < 1e-2:
        return f"{v:.0e}"
    return str(v)


def _pair_grid(log, a, b, values_a, values_b):
    """Best macro F1 and number of distinct configurations for every (a, b) value pair, in search space order.
    Pairs that were never sampled are NaN (and 0 configurations)."""
    best = np.full((len(values_a), len(values_b)), np.nan)
    count = np.zeros(best.shape, dtype=int)
    pos_a = {v: i for i, v in enumerate(values_a)}
    pos_b = {v: j for j, v in enumerate(values_b)}
    grouped = log.group_by([a, b]).agg(pl.col("macro_f1").max().alias("best"),
                                       pl.col("config_id").n_unique().alias("n"))
    for va, vb, f1, n in grouped.iter_rows():
        if va in pos_a and vb in pos_b:
            best[pos_a[va], pos_b[vb]] = f1
            count[pos_a[va], pos_b[vb]] = n
    return best, count


def _rank_consistency(ax, log, budget_name, color):
    """Score of every configuration at one budget against its score at the next budget, one color per step, with
    the Spearman rank correlation of every step. Tests Hyperband's assumption that early rankings hold later."""
    budgets = sorted(log[budget_name].unique())
    low_all, high_all = log["macro_f1"].min() - 0.01, log["macro_f1"].max() + 0.01
    for low, high in zip(budgets[:-1], budgets[1:]):
        pairs = (log.filter(pl.col(budget_name) == low).select("config_id", pl.col("macro_f1").alias("x"))
                 .join(log.filter(pl.col(budget_name) == high).select("config_id", pl.col("macro_f1").alias("y")),
                       on="config_id"))
        if pairs.height == 0:
            continue
        label = f"{low} → {high} (n={pairs.height})"
        if pairs.height >= 3:
            label = f"{low} → {high}: ρ = {pairs.select(pl.corr('x', 'y', method='spearman')).item():.2f} (n={pairs.height})"
        ax.scatter(pairs["x"], pairs["y"], s=28, color=color[high], ec="white", lw=0.5, label=label, zorder=3)
    ax.plot([low_all, high_all], [low_all, high_all], "--", color="gray", lw=1, zorder=1)
    ax.set_xlim(low_all, high_all)
    ax.set_ylim(low_all, high_all)
    ax.set_xlabel(f"macro F1 at the smaller budget ({budget_name})", fontsize=9)
    ax.set_ylabel("macro F1 at the next budget", fontsize=9)
    ax.set_title("Rank consistency: do early scores predict later ones?\n"
                 "(ρ = Spearman, only survivors are promoted, dashed = no change)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

def plot_hyperband(log, search_space, budget_name="trees", budget=None, cmap="viridis"):
    """
    Creates a Heatmap Survival Funnel. 
    One panel per parameter. X-axis is the budget, Y-axis is the parameter value.
    Cells are colored by the best macro F1 score reached by that value at that budget.
    The text shows the number of surviving configurations. Blank cells were eliminated.
    """
    params = list(search_space)
    P = len(params)
    all_budgets = sorted(log[budget_name].unique())

    # Handle backwards-compatible budget filtering
    if budget is None:
        data, scope = log, "survival funnel across all budgets"
        budgets = all_budgets
    else:
        chosen = all_budgets[0] if budget == "min" else budget
        assert chosen in all_budgets, f"budget {chosen} is not in the log, the budgets are {all_budgets}"
        data, scope = log.filter(pl.col(budget_name) == chosen), f"evaluations at {chosen} {budget_name}"
        budgets = [chosen]

    best = log.filter(pl.col(budget_name) == all_budgets[-1]).sort("macro_f1", descending=True).row(0, named=True)

    fig, axes = plt.subplots(1, P, figsize=(3.5 * P + 1, 4.5))
    if P == 1:
        axes = [axes]

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("#e5e5e5") # Gray out eliminated parameter values
    norm = plt.Normalize(data["macro_f1"].min(), data["macro_f1"].max())

    for i, param in enumerate(params):
        ax = axes[i]
        vy = search_space[param]
        vx = budgets
        
        # Build the heatmap matrix for this parameter
        f1_matrix = np.full((len(vy), len(vx)), np.nan)
        count_matrix = np.zeros((len(vy), len(vx)), dtype=int)
        
        for c_idx, b in enumerate(vx):
            b_data = data.filter(pl.col(budget_name) == b)
            if b_data.is_empty():
                continue
                
            grouped = b_data.group_by(param).agg(
                pl.col("macro_f1").max().alias("max_f1"),
                pl.len().alias("count")
            )
            
            # Map back to the search space matrix
            for row in grouped.iter_rows(named=True):
                if row[param] in vy:
                    r_idx = vy.index(row[param])
                    f1_matrix[r_idx, c_idx] = row["max_f1"]
                    count_matrix[r_idx, c_idx] = row["count"]

        # Draw the heatmap
        image = ax.imshow(np.ma.masked_invalid(f1_matrix), cmap=colormap, norm=norm, aspect="auto", origin="lower")
        
        # Add the survival counts as text
        for r in range(len(vy)):
            for c in range(len(vx)):
                if count_matrix[r, c] == 0:
                    ax.text(c, r, "–", ha="center", va="center", fontsize=8, color="gray")
                else:
                    text_color = "black" if norm(f1_matrix[r, c]) > 0.6 else "white"
                    ax.text(c, r, f"{f1_matrix[r, c]:.3f}\nn={count_matrix[r, c]}", 
                            ha="center", va="center", fontsize=7, color=text_color)
                            
        # Highlight the best overall parameter value in gold at the final budget
        if best[param] in vy and (budget is None or budget == all_budgets[-1]):
            best_r = vy.index(best[param])
            best_c = vx.index(all_budgets[-1]) if budget is None else 0
            ax.add_patch(plt.Rectangle((best_c - 0.5, best_r - 0.5), 1, 1, fill=False, ec="gold", lw=3, zorder=5))

        # Formatting
        def _fmt(val):
            return f"{val:g}" if isinstance(val, float) else str(val)
            
        ax.set_yticks(range(len(vy)), [_fmt(v) for v in vy], fontsize=8)
        ax.set_xticks(range(len(vx)), [_fmt(v) for v in vx], fontsize=8)
        ax.set_title(param, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Budget ({budget_name})", fontsize=10)
        
    cbar = fig.colorbar(image, ax=axes, pad=0.02, aspect=40)
    cbar.set_label("Max Macro F1 Reached")

    fig.suptitle(f"Hyperband Survival Funnel ({scope})\n"
                 "Gold = Best configuration at max budget", fontsize=12, y=1.05)
    plt.show()