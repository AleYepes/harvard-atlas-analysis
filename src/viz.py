import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def plot_product_space(country_df: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame, product_meta: pd.DataFrame) -> go.Figure:
    """Plots the product space map for a given country."""
    # Merge product_name into nodes
    nodes_with_names = nodes.merge(
        product_meta[["product_hs92_code", "product_name"]],
        on="product_hs92_code",
        how="left"
    )

    # Merge product_name into nodes
    nodes_with_names = nodes.merge(
        product_meta[["product_hs92_code", "product_name"]],
        on="product_hs92_code",
        how="left"
    )

    # Merge export_rca into nodes for sizing
    nodes_with_rca = nodes_with_names.merge(
        country_df[["product_hs92_code", "export_rca"]],
        on="product_hs92_code",
        how="left"
    )
    nodes_with_rca["export_rca"] = nodes_with_rca["export_rca"].fillna(0) # Fill NaN with 0 for products not in country_df

    fig = px.scatter(
        nodes_with_rca,
        x="product_space_x",
        y="product_space_y",
        hover_data=["product_hs92_code", "product_name", "export_rca"],
        color="product_space_cluster_name",
        size=nodes_with_rca["export_rca"].apply(lambda x: np.log1p(x)),
        title=f"Product Space - {country_df['country_iso3_code'].iloc[0]}"
    )

    edge_x = []
    edge_y = []

    # Filter edges to only include products present in the nodes DataFrame
    valid_product_codes = nodes["product_hs92_code"].unique()
    filtered_edges = edges[
        (edges["product_hs92_code_source"].isin(valid_product_codes)) &
        (edges["product_hs92_code_target"].isin(valid_product_codes))
    ]

    for _, edge in filtered_edges.iterrows():
        source_node = nodes[nodes["product_hs92_code"] == edge["product_hs92_code_source"]]
        target_node = nodes[nodes["product_hs92_code"] == edge["product_hs92_code_target"]]

        if not source_node.empty and not target_node.empty:
            x0 = source_node["product_space_x"].values[0]
            y0 = source_node["product_space_y"].values[0]
            x1 = target_node["product_space_x"].values[0]
            y1 = target_node["product_space_y"].values[0]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5, color="#888"), hoverinfo="none"))
    return fig

def plot_opportunities_scatter(country_df: pd.DataFrame, use: str = "density", presence: str = "rca") -> go.Figure:
    """Plots the growth opportunities scatter plot."""
    x_axis = use
    y_axis = "export_rca" if presence == "rca" else "rel_presence"

    fig = px.scatter(
        country_df,
        x=x_axis,
        y=y_axis,
        hover_data=["product_hs92_code", "product_name"],
        title=f"Growth Opportunities - {country_df['country_iso3_code'].iloc[0]}"
    )
    return fig
