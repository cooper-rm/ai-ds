"""
Deep learning imputation methods — GAIN, MIDA, HI-VAE.
All require torch. Raise RuntimeError with a clear message if not available.
"""
import numpy as np

NUMERIC_DTYPES = ("int8","int16","int32","int64","float32","float64")


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        raise RuntimeError(
            "torch is not installed. Run: pip install torch  "
            "or re-run the pipeline — torch is in requirements.txt and will be installed on next run."
        )


def gain(df, col, ev):
    """
    GAIN — Generative Adversarial Imputation Network.
    Generator fills in missing values; discriminator tries to identify imputed entries.
    The hint matrix (partially revealed mask) guides the discriminator.
    """
    torch, nn = _require_torch()

    numeric_df = df.select_dtypes(include="number")
    if col not in numeric_df.columns:
        return None

    cols    = list(numeric_df.columns)
    col_idx = cols.index(col)
    data    = numeric_df.values.astype(np.float32)
    mask    = (~np.isnan(data)).astype(np.float32)

    col_means = np.nanmean(data, axis=0)
    col_stds  = np.nanstd(data, axis=0)
    col_stds[col_stds == 0] = 1
    data_norm = (np.where(np.isnan(data), 0, data) - col_means) / col_stds

    n, d = data_norm.shape
    h    = max(d * 2, 32)

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d * 2, h), nn.ReLU(),
                nn.Linear(h, h),     nn.ReLU(),
                nn.Linear(h, d),
            )
        def forward(self, x, m):
            return self.net(torch.cat([x, m], dim=1))

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d * 2, h), nn.ReLU(),
                nn.Linear(h, h),     nn.ReLU(),
                nn.Linear(h, d),     nn.Sigmoid(),
            )
        def forward(self, x, h_in):
            return self.net(torch.cat([x, h_in], dim=1))

    G     = Generator()
    D     = Discriminator()
    opt_G = torch.optim.Adam(G.parameters(), lr=1e-3)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-3)

    X = torch.tensor(data_norm)
    M = torch.tensor(mask)

    for _ in range(300):
        hint_rate = 0.9
        H         = (torch.rand(n, d) < hint_rate).float() * M
        noise     = torch.randn(n, d) * 0.01
        X_in      = M * X + (1 - M) * noise
        G_sample  = G(X_in, M)
        X_hat     = M * X + (1 - M) * G_sample

        D_prob = D(X_hat.detach(), H)
        D_loss = -torch.mean(
            M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1 - D_prob + 1e-8)
        )
        opt_D.zero_grad(); D_loss.backward(); opt_D.step()

        D_prob = D(X_hat, H)
        G_loss = -torch.mean((1 - M) * torch.log(D_prob + 1e-8))
        MSE    = torch.mean((M * X - M * G_sample) ** 2) / (torch.mean(M) + 1e-8)
        opt_G.zero_grad(); (G_loss + 2.0 * MSE).backward(); opt_G.step()

    with torch.no_grad():
        X_in    = M * X
        imputed = G(X_in, M).numpy()

    imputed_denorm = imputed * col_stds + col_means
    out            = df[col].copy()
    missing_mask   = df[col].isnull().values
    out.iloc[np.where(missing_mask)[0]] = imputed_denorm[missing_mask, col_idx]
    return out


def mida(df, col, ev):
    """
    MIDA — Multiple Imputation using Denoising Autoencoders.
    Trains a DAE on observed rows (randomly corrupting inputs during training)
    so the network learns to reconstruct full rows from partial observations.
    """
    torch, nn = _require_torch()

    numeric_df = df.select_dtypes(include="number")
    if col not in numeric_df.columns:
        return None

    cols    = list(numeric_df.columns)
    col_idx = cols.index(col)
    data    = numeric_df.values.astype(np.float32)

    means    = np.nanmean(data, axis=0)
    stds     = np.nanstd(data, axis=0); stds[stds == 0] = 1
    filled   = np.where(np.isnan(data), means, data)
    norm     = (filled - means) / stds
    obs_mask = (~np.isnan(data)).astype(np.float32)

    n, d = norm.shape
    h    = max(d * 3, 64)

    class DAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d, h),     nn.ReLU(),
                nn.Linear(h, h // 2), nn.ReLU(),
            )
            self.dec = nn.Sequential(
                nn.Linear(h // 2, h), nn.ReLU(),
                nn.Linear(h, d),
            )
        def forward(self, x):
            return self.dec(self.enc(x))

    model    = DAE()
    opt      = torch.optim.Adam(model.parameters(), lr=1e-3)
    X        = torch.tensor(norm)
    obs      = torch.tensor(obs_mask)

    for _ in range(400):
        corrupt_mask = (torch.rand(n, d) > 0.2).float() * obs
        X_corrupt    = X * corrupt_mask
        recon        = model(X_corrupt)
        loss         = torch.mean(obs * (recon - X) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        imputed = model(X).numpy() * stds + means

    out          = df[col].copy()
    missing_mask = df[col].isnull().values
    out.iloc[np.where(missing_mask)[0]] = imputed[missing_mask, col_idx]
    return out


def hivae(df, col, ev):
    """
    HI-VAE — Heterogeneous-Incomplete Variational Autoencoder.
    VAE trained only on observed entries; missing values are imputed
    by sampling from the posterior given observed dimensions.
    Supports mixed column types.
    """
    torch, nn = _require_torch()

    numeric_df = df.select_dtypes(include="number")
    if col not in numeric_df.columns or len(numeric_df.columns) < 2:
        return None

    cols    = list(numeric_df.columns)
    col_idx = cols.index(col)
    data    = numeric_df.values.astype(np.float32)

    means = np.nanmean(data, axis=0)
    stds  = np.nanstd(data, axis=0); stds[stds == 0] = 1
    norm  = np.where(np.isnan(data), 0, (data - means) / stds)
    obs   = (~np.isnan(data)).astype(np.float32)

    n, d  = norm.shape
    lat   = max(d // 2, 2)
    h     = max(d * 2, 32)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(nn.Linear(d, h), nn.ReLU())
            self.mu     = nn.Linear(h, lat)
            self.logvar = nn.Linear(h, lat)
        def forward(self, x):
            hid = self.shared(x)
            return self.mu(hid), self.logvar(hid)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(lat, h), nn.ReLU(), nn.Linear(h, d))
        def forward(self, z):
            return self.net(z)

    enc = Encoder(); dec = Decoder()
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    X   = torch.tensor(norm)
    M   = torch.tensor(obs)

    for _ in range(400):
        mu, logvar = enc(X * M)
        std        = torch.exp(0.5 * logvar)
        z          = mu + std * torch.randn_like(std)
        recon      = dec(z)
        recon_loss = torch.mean(M * (recon - X) ** 2)
        kl_loss    = -0.5 * torch.mean(1 + logvar - mu ** 2 - torch.exp(logvar))
        loss       = recon_loss + 0.01 * kl_loss
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        mu, _  = enc(X * M)
        imputed = dec(mu).numpy() * stds + means

    out          = df[col].copy()
    missing_mask = df[col].isnull().values
    out.iloc[np.where(missing_mask)[0]] = imputed[missing_mask, col_idx]
    return out
