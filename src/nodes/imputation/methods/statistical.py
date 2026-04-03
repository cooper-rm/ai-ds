"""Statistical imputation methods — regression, MICE, MissForest, EM, SoftImpute, PMM, hotdeck."""
import numpy as np

NUMERIC_DTYPES = ("int8","int16","int32","int64","float32","float64")


def _numeric_correlates(df, ev):
    return [
        c for c in ev.get("mar_correlates", [])
        if c in df.columns and str(df[c].dtype) in NUMERIC_DTYPES
    ]


def regression(df, col, ev, noise=False):
    from sklearn.linear_model import LinearRegression
    feature_cols = _numeric_correlates(df, ev)
    if not feature_cols:
        return None
    mask_c = df[col].notnull() & df[feature_cols].notnull().all(axis=1)
    mask_m = df[col].isnull()  & df[feature_cols].notnull().all(axis=1)
    if mask_c.sum() < 10 or mask_m.sum() == 0:
        return None
    model = LinearRegression()
    model.fit(df.loc[mask_c, feature_cols], df.loc[mask_c, col])
    preds = model.predict(df.loc[mask_m, feature_cols])
    if noise:
        residuals = df.loc[mask_c, col].values - model.predict(df.loc[mask_c, feature_cols])
        preds = preds + np.random.normal(0, 0.5 * residuals.std(), len(preds))
    out = df[col].copy()
    out.loc[mask_m] = preds
    if out.isnull().any():
        out = out.fillna(df[col].median())
    return out


def stochastic_regression(df, col, ev):
    return regression(df, col, ev, noise=True)


def pmm(df, col, ev):
    """Predictive mean matching — map prediction to closest observed donor."""
    from sklearn.linear_model import LinearRegression
    feature_cols = _numeric_correlates(df, ev)
    if not feature_cols:
        return None
    mask_c = df[col].notnull() & df[feature_cols].notnull().all(axis=1)
    mask_m = df[col].isnull()  & df[feature_cols].notnull().all(axis=1)
    if mask_c.sum() < 10 or mask_m.sum() == 0:
        return None
    model = LinearRegression()
    model.fit(df.loc[mask_c, feature_cols], df.loc[mask_c, col])
    preds_obs  = model.predict(df.loc[mask_c, feature_cols])
    preds_miss = model.predict(df.loc[mask_m, feature_cols])
    obs_vals   = df.loc[mask_c, col].values
    out        = df[col].copy()
    for idx, pred in zip(df[mask_m].index, preds_miss):
        donor_idx   = np.argmin(np.abs(preds_obs - pred))
        out.loc[idx] = obs_vals[donor_idx]
    return out


def hotdeck(df, col, ev):
    """Fill each missing row from the most similar complete row (Euclidean distance)."""
    feature_cols = _numeric_correlates(df, ev)
    out          = df[col].copy()
    if not feature_cols:
        fallback = df[col].mode()[0] if df[col].notnull().any() else 0
        return out.fillna(fallback)
    complete_mask = df[col].notnull() & df[feature_cols].notnull().all(axis=1)
    missing_mask  = df[col].isnull()  & df[feature_cols].notnull().all(axis=1)
    if complete_mask.sum() == 0:
        return None
    complete_rows = df.loc[complete_mask, feature_cols].values
    for idx in df[missing_mask].index:
        row   = df.loc[idx, feature_cols].values
        dists = np.sqrt(((complete_rows - row) ** 2).sum(axis=1))
        donor = df.loc[complete_mask].index[np.argmin(dists)]
        out.loc[idx] = df.loc[donor, col]
    return out


def mice(df, col, ev):
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    numeric_df = df.select_dtypes(include="number")
    if col not in numeric_df.columns:
        return None
    imputer = IterativeImputer(max_iter=10, random_state=42)
    result  = imputer.fit_transform(numeric_df)
    col_idx = list(numeric_df.columns).index(col)
    out     = df[col].copy()
    out[df[col].isnull()] = result[df[col].isnull().values, col_idx]
    return out


def missforest(df, col, ev):
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    is_numeric   = str(df[col].dtype) in NUMERIC_DTYPES
    feature_cols = [c for c in df.columns if c != col and str(df[c].dtype) in NUMERIC_DTYPES]
    if not feature_cols:
        return None
    X      = df[feature_cols].values
    si     = SimpleImputer(strategy="median")
    X      = si.fit_transform(X)
    mask_c = df[col].notnull().values
    mask_m = df[col].isnull().values
    if mask_c.sum() < 10 or mask_m.sum() == 0:
        return None
    Model  = RandomForestRegressor if is_numeric else RandomForestClassifier
    model  = Model(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X[mask_c], df.loc[mask_c, col].values)
    preds  = model.predict(X[mask_m])
    out    = df[col].copy()
    out.iloc[np.where(mask_m)[0]] = preds
    return out


def em(df, col, ev):
    """EM under multivariate normal — iterate predict / refit."""
    from sklearn.linear_model import LinearRegression
    feature_cols = _numeric_correlates(df, ev)
    if not feature_cols:
        return None
    out    = df[col].fillna(df[col].median()).copy()
    mask_m = df[col].isnull()
    if mask_m.sum() == 0:
        return out
    for _ in range(10):
        mask_c = ~mask_m
        model  = LinearRegression()
        model.fit(df.loc[mask_c, feature_cols].fillna(0), out[mask_c])
        out.loc[mask_m] = model.predict(df.loc[mask_m, feature_cols].fillna(0))
    return out


def softimpute(df, col, ev):
    """Nuclear norm matrix completion via iterative SVD shrinkage."""
    numeric_df = df.select_dtypes(include="number")
    if col not in numeric_df.columns or len(numeric_df.columns) < 2:
        return None
    means    = numeric_df.mean()
    stds     = numeric_df.std().replace(0, 1)
    M        = (numeric_df - means) / stds
    M_filled = M.fillna(0).values.copy()
    mask_m   = numeric_df[col].isnull().values
    col_idx  = list(numeric_df.columns).index(col)
    obs_mask = ~numeric_df.isnull().values
    for _ in range(20):
        U, s, Vt   = np.linalg.svd(M_filled, full_matrices=False)
        s_shrunk   = np.maximum(s - 0.5 * s[0], 0)
        M_filled   = (U * s_shrunk) @ Vt
        M_filled[obs_mask] = M.values[obs_mask]
    result = M_filled * stds.values + means.values
    out    = df[col].copy()
    out.iloc[np.where(mask_m)[0]] = result[mask_m, col_idx]
    return out
