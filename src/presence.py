import pandas as pd

def add_rca_binary(df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """Adds a binary specialization column based on RCA."""
    df["x_binary"] = (df["export_rca"] >= threshold).astype(int)
    return df

def add_peer_relative_presence(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Calculates peer-relative presence metrics."""
    df["share"] = df["export_value"] / df.groupby("country_iso3_code")["export_value"].transform("sum")
    peer_share = df.groupby("product_hs92_code")["share"].transform("mean")
    df["rel_presence"] = df["share"] / (peer_share + eps)
    df["abs_presence"] = df["share"] - peer_share
    return df
