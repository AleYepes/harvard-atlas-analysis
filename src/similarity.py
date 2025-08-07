import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

def country_similarity_cosine(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates cosine similarity between countries based on RCA vectors."""
    rca_matrix = df.pivot(index="country_iso3_code", columns="product_hs92_code", values="export_rca").fillna(0)
    similarity_matrix = cosine_similarity(rca_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=rca_matrix.index, columns=rca_matrix.index)
    return similarity_df

def country_similarity_jaccard(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Jaccard similarity between countries based on binary specialization."""
    binary_matrix = df.pivot(index="country_iso3_code", columns="product_hs92_code", values="x_binary").fillna(0)
    similarity_matrix = np.zeros((binary_matrix.shape[0], binary_matrix.shape[0]))
    for i in range(binary_matrix.shape[0]):
        for j in range(binary_matrix.shape[0]):
            similarity_matrix[i, j] = jaccard_score(binary_matrix.iloc[i], binary_matrix.iloc[j])
    similarity_df = pd.DataFrame(similarity_matrix, index=binary_matrix.index, columns=binary_matrix.index)
    return similarity_df
