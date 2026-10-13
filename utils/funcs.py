import random
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer


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

    
def balance_classes(
    df: pd.DataFrame | pl.DataFrame,
    class_col: str = "genre",
    target_count: int | None = None,
    tolerance: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame | pl.DataFrame:
    """Balances class sizes by randomly sampling each class to a different size

    within the [1 - tolerance, 1 + tolerance] interval.
    Accepts and returns either a Polars or Pandas DataFrame.
    """
    is_pandas = isinstance(df, pd.DataFrame)
    rng = random.Random(seed)

    # ------------------ PANDAS IMPLEMENTATION ------------------
    if is_pandas:
        counts = df[class_col].value_counts()

        if target_count is None:
            target_count = counts.min()

        lower_bound = int(target_count * (1.0 - tolerance))
        upper_bound = int(target_count * (1.0 + tolerance))

        balanced_groups = []
        for class_val, group in df.groupby(class_col):
            n_rows = len(group)
            class_target = rng.randint(lower_bound, upper_bound)

            if n_rows > class_target:
                sampled = group.sample(
                    n=class_target,
                    random_state=rng.randint(0, 10_000),
                    replace=False,
                )
            else:
                sampled = group

            balanced_groups.append(sampled)

        # Concat and shuffle
        return (
            pd.concat(balanced_groups, axis=0)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )

    # ------------------ POLARS IMPLEMENTATION ------------------
    elif isinstance(df, pl.DataFrame):
        counts = df[class_col].value_counts()
        # In Polars, value_counts creates a column named 'count' (or class_col_count in older versions)
        count_col = "count" if "count" in counts.columns else counts.columns[1]

        if target_count is None:
            target_count = counts[count_col].min()

        lower_bound = int(target_count * (1.0 - tolerance))
        upper_bound = int(target_count * (1.0 + tolerance))

        balanced_partitions = []
        for group_df in df.partition_by(class_col, as_dict=False):
            n_rows = len(group_df)
            class_target = rng.randint(lower_bound, upper_bound)

            if n_rows > class_target:
                sampled = group_df.sample(
                    n=class_target, seed=rng.randint(0, 10_000), shuffle=True
                )
            else:
                sampled = group_df

            balanced_partitions.append(sampled)

        return pl.concat(balanced_partitions).sample(
            fraction=1.0, shuffle=True, seed=seed
        )

    else:
        raise TypeError(
            f"Expected Polars or Pandas DataFrame, got {type(df).__name__}"
        )