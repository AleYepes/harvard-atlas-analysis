import pandas as pd

def add_rca_binary(df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    temp = df.copy()
    temp["binary_specialization"] = (temp["export_rca"] >= threshold).astype(int)
    return temp

def add_peer_relative_presence(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    temp = df.copy()
    temp["share"] = temp["export_value"] / temp.groupby("country_iso3_code")["export_value"].transform("sum")
    peer_share = temp.groupby("product_hs92_code")["share"].transform("mean")
    temp["rel_presence"] = temp["share"] / (peer_share + eps)
    temp["abs_presence"] = temp["share"] - peer_share
    return temp
