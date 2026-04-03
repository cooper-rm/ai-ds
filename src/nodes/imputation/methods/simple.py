"""Simple imputation methods — no external ML deps beyond sklearn."""
import numpy as np

NUMERIC_DTYPES = ("int8","int16","int32","int64","float32","float64")


def mean(df, col, ev):
    return df[col].fillna(df[col].mean())


def median(df, col, ev):
    return df[col].fillna(df[col].median())


def mode(df, col, ev):
    m = df[col].mode()
    if len(m) == 0:
        return None
    return df[col].fillna(m[0])


def grouped_median(df, col, ev):
    cat_corr = ev.get("top_categorical_correlate")
    if not cat_corr or cat_corr not in df.columns:
        return df[col].fillna(df[col].median())
    grouped = df.groupby(cat_corr)[col].transform("median")
    return df[col].fillna(grouped).fillna(df[col].median())


def knn(df, col, ev):
    """
    KNN imputation.
    - Numeric target: standard KNNImputer on numeric feature set.
    - Categorical target: encode categories as integer codes, run KNN,
      then map codes back to original category labels.
    """
    from sklearn.impute import KNNImputer
    import pandas as pd

    is_numeric = str(df[col].dtype) in NUMERIC_DTYPES
    correlates  = ev.get("mar_correlates", [])
    feature_cols = [col] + [c for c in correlates if c in df.columns]

    if is_numeric:
        numeric_sub = df[feature_cols].select_dtypes(include="number")
        if col not in numeric_sub.columns:
            return None
        imputer = KNNImputer(n_neighbors=5)
        result  = imputer.fit_transform(numeric_sub)
        col_idx = list(numeric_sub.columns).index(col)
        out     = df[col].copy()
        out[df[col].isnull()] = result[df[col].isnull().values, col_idx]
        return out

    # ── Categorical target ────────────────────────────────────────────────────
    # Encode the target as float codes (NaN stays NaN), run KNN, decode back.
    cat_codes = df[col].astype("category")
    categories = cat_codes.cat.categories          # original labels
    encoded = cat_codes.cat.codes.astype(float)
    encoded[df[col].isnull()] = np.nan             # restore NaN mask

    # Build numeric feature matrix: encoded target + any numeric correlates
    numeric_features = df[feature_cols].select_dtypes(include="number")
    work = numeric_features.copy()
    work[col] = encoded                            # replace with encoded col

    imputer = KNNImputer(n_neighbors=5)
    result  = imputer.fit_transform(work)
    col_idx = list(work.columns).index(col)

    # Round imputed codes and map back to category labels
    imputed_codes = np.round(result[:, col_idx]).astype(int).clip(0, len(categories) - 1)
    out = df[col].copy()
    missing_mask = df[col].isnull()
    out[missing_mask] = pd.Categorical.from_codes(
        imputed_codes[missing_mask.values], categories=categories
    )
    return out
