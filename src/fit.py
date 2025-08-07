import pandas as pd
import numpy as np

def add_density_from_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates density from distance, clipped to [0, 1]."""
    df["density"] = (1 - df["distance"]).clip(0, 1)
    return df

def recompute_density_from_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Recomputes density from proximity for QA."""
    x_binary = df.pivot(index="country_iso3_code", columns="product_hs92_code", values="x_binary").fillna(0)
    cooccurrence = x_binary.T @ x_binary
    p_q_given_p = cooccurrence.T / np.diag(cooccurrence)
    p_p_given_q = cooccurrence / np.diag(cooccurrence)
    phi = np.minimum(p_q_given_p, p_p_given_q)
    density_recomputed = (x_binary @ phi) / phi.sum(axis=1)
    density_recomputed = density_recomputed.stack().reset_index()
    density_recomputed.columns = ["country_iso3_code", "product_hs92_code", "density_recomputed"]
    df = df.merge(density_recomputed, on=["country_iso3_code", "product_hs92_code"], how="left")
    return df
