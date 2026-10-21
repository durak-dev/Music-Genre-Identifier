import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import label_binarize
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


def diagnose(y_true, P, classes, pred=None, name="model", train_accuracy=None, save_path=None):
    """Print the headline scores and draw four diagnostic plots for any classifier.

    y_true  : true labels, either genre names or integer ids that index into `classes`
    P       : (n_songs, n_classes) probabilities, columns in the same order as `classes`
              (for a transformer, apply a softmax to the logits first)
    classes : genre names, e.g. rfc.classes_ or le.classes_
    pred    : optional predicted labels; defaults to the most probable class
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
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

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return {"accuracy": accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, average="macro"),
            "weighted_f1": f1_score(y_true, pred, average="weighted")}