import pandas as pd

def load_country_product(year: int, data_path: str = "data/hs92_country_product_year_4.csv") -> pd.DataFrame:
    """Loads the country-product-year data and filters by year."""
    dtype_spec = {
        'country_id': 'int32',
        'country_iso3_code': 'str',
        'product_id': 'int32',
        'product_hs92_code': 'str',
        'year': 'int16',
        'export_value': 'int64',
        'import_value': 'int64',
        'global_market_share': 'float32',
        'export_rca': 'float32',
        'distance': 'float32',
        'cog': 'float32',
        'pci': 'float32'
    }
    df = pd.read_csv(data_path, dtype=dtype_spec)
    df = df[df["year"] == year].copy()
    df.columns = [col.lower() for col in df.columns]
    # Ensure distance is numeric and handle NaNs before clipping
    if 'distance' in df.columns:
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
        df['distance'] = df['distance'].fillna(0) # Fill NaN with 0, or another appropriate value
        df['distance'] = df['distance'].clip(0, 1)
    if 'product_hs92_code' in df.columns:
        df['product_hs92_code'] = df['product_hs92_code'].astype(str)
    return df

def load_product_meta(data_path: str = "data/product_hs92.csv") -> pd.DataFrame:
    """Loads the product metadata."""
    dtype_spec = {
        'product_id': 'int32',
        'product_hs92_code': 'str',
        'product_level': 'int8',
        'product_name': 'str',
        'product_name_short': 'str',
        'product_parent_id': 'Int32', # Use Int32 for nullable integer
        'product_id_hierarchy': 'str',
        'show_feasibility': 'bool',
        'natural_resource': 'bool'
    }
    df = pd.read_csv(data_path, dtype=dtype_spec)
    df.columns = [col.lower() for col in df.columns]
    if 'product_hs92_code' in df.columns:
        df['product_hs92_code'] = df['product_hs92_code'].astype(str)
    return df

def load_layout(data_path: str = "data/umap_layout_hs92.csv") -> pd.DataFrame:
    """Loads the product space layout data."""
    dtype_spec = {
        'product_hs92_code': 'str',
        'product_space_x': 'float32',
        'product_space_y': 'float32',
        'product_space_cluster_name': 'str'
    }
    df = pd.read_csv(data_path, dtype=dtype_spec)
    df.columns = [col.lower() for col in df.columns]
    if 'product_hs92_code' in df.columns:
        df['product_hs92_code'] = df['product_hs92_code'].astype(str)
    return df

def load_edges(data_path: str = "data/top_edges_hs92.csv") -> pd.DataFrame:
    """Loads the product space edges and normalizes column names."""
    dtype_spec = {
        'product_hs92_code_source': 'str',
        'product_hs92_code_target': 'str'
    }
    df = pd.read_csv(data_path, dtype=dtype_spec)
    df.columns = ["product_hs92_code_source", "product_hs92_code_target"]
    return df

def load_country_year(data_path: str = "data/hs92_country_year.csv") -> pd.DataFrame:
    """Loads the country-year data and renames the export value column."""
    dtype_spec = {
        'country_id': 'int32',
        'country_iso3_code': 'str',
        'year': 'int16',
        'export_value': 'Int64', # Changed to nullable integer
        'import_value': 'Int64', # Changed to nullable integer
        'eci': 'float32',
        'coi': 'float32',
        'diversity': 'float32',
        'growth_proj': 'float32'
    }
    df = pd.read_csv(data_path, dtype=dtype_spec)
    df.columns = [col.lower() for col in df.columns]
    df = df.rename(columns={"export_value": "export_value_total"})
    return df