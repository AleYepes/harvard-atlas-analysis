import pandas as pd
import numpy as np
import os

def load_country_product(year: int, data_path: str = "../data/hs92_country_product_year_4.csv") -> pd.DataFrame:
    dtype_spec = {
        'country_id': 'uint16',
        'country_iso3_code': 'str',
        'product_id': 'uint16',
        'product_hs92_code': 'str',
        'year': 'uint16',
        'export_value': 'uint64',
        'import_value': 'int64',
        'global_market_share': 'float32',
        'export_rca': 'float32',
        'distance': 'float32',
        'cog': 'float32',
        'pci': 'float32'
    }
    df = pd.read_csv(data_path, dtype=dtype_spec)
    df = df[df["year"] == year]
    if df.empty:
        raise ValueError(f"{year} data not available")
    df.columns = [col.lower() for col in df.columns]

    df['product_hs92_code'] = df['product_hs92_code'].apply(lambda x:x if x.isdigit() else np.nan)
    df['product_hs92_code'] = df['product_hs92_code'].astype('Int32') # Nullable int for trade error products

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    return df.drop(columns=['country_id', 'product_id', 'year', 'import_value', 'global_market_share'], errors='ignore')

def load_product_meta(data_path: str = "../data/product_hs92.csv") -> pd.DataFrame:
    dtype_spec = {
        'product_id': 'uint16',
        'product_hs92_code': 'str',
        'product_level': 'uint8',
        'product_name': 'str',
        'product_name_short': 'str',
        'product_parent_id': 'Int16', # Nullable int for root products w/out parents
        'product_id_hierarchy': 'str',
        'show_feasibility': 'bool',
        'natural_resource': 'bool'
    }
    df = pd.read_csv(data_path, dtype=dtype_spec)
    df.columns = [col.lower() for col in df.columns]

    df['product_hs92_code'] = df['product_hs92_code'].apply(lambda x:x if x.isdigit() else np.nan)
    df['product_hs92_code'] = df['product_hs92_code'].astype('Int32') # Nullable int for trade error products
    # df = df.dropna(subset=['product_hs92_code'])

    return df.drop(columns=['product_level', 'product_id', 'green_product', 'product_id_hierarchy'], errors='ignore')

def load_product_space_vectors(data_path: str = "../data/umap_layout_hs92.csv") -> pd.DataFrame:
    dtype_spec = {
        'product_hs92_code': 'str',
        'product_space_x': 'float64',
        'product_space_y': 'float64',
        'product_space_cluster_name': 'str'
    }
    df = pd.read_csv(data_path, dtype=dtype_spec)
    df.columns = [col.lower() for col in df.columns]

    df['product_hs92_code'] = df['product_hs92_code'].apply(lambda x:x if x.isdigit() else np.nan)
    df['product_hs92_code'] = df['product_hs92_code'].astype('Int32') # Nullable int for trade error products

    return df

def load_product_space_edges(data_path: str = "../data/top_edges_hs92.csv") -> pd.DataFrame:
    dtype_spec = {
        'product_hs92_code_source': 'int',
        'product_hs92_code_target': 'int'
    }
    df = pd.read_csv(data_path, dtype=dtype_spec)
    df.columns = ["product_hs92_code_source", "product_hs92_code_target"]
    return df

def load_country_year(data_path: str = "../data/hs92_country_year.csv") -> pd.DataFrame:
    dtype_spec = {
        'country_id': 'uint16',
        'country_iso3_code': 'str',
        'year': 'uint16',
        'export_value': 'uint64', # Changed to nullable integer
        'import_value': 'int64', # Changed to nullable integer
        'eci': 'float32',
        'coi': 'float32',
        'diversity': 'float32',
        'growth_proj': 'float32'
    }
    df = pd.read_csv(data_path, dtype=dtype_spec)
    df.columns = [col.lower() for col in df.columns]
    df = df.rename(columns={"export_value": "export_value_country_total"})
    return df.drop(columns=['import_value', ])