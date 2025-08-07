# Metroverse-like analysis using Atlas HS92 data

## Objective
Build a country–product analytics pipeline that mirrors Metroverse concepts (Relative Presence, Knowledge Clusters, Industry Space, Similarity, Growth Opportunities) using Atlas HS92 data, prioritizing simplicity and clear outputs.

## Project structure
- `data/` raw CSVs
- `outputs/` processed CSV artifacts and figures
- `src/` modules:
  - `io_load.py` (load, schema checks, type casting; column normalization)
  - `presence.py` (RCA and optional peer-relative presence)
  - `fit.py` (fit from `distance` as density; optional recomputation for QA)
  - `similarity.py` (cosine on RCA; Jaccard optional)
  - `viz.py` (Product Space map; Opportunities scatter)
  - `pipeline.py` (orchestrator; CLI; logging)
- `config.yaml` (core toggles and defaults)
- `notebooks/` (EDA and QA)

## Key inputs
- `data/hs92_country_product_year_4.csv`
  - Keys: `country_iso3_code`, `product_hs92_code`, `year`
  - Fields: `export_value`, `global_market_share`, `export_rca`, `distance`, `cog`, `pci`
- `data/product_hs92.csv`
  - Keys: `product_hs92_code`
  - Fields: `product_name`, `product_id_hierarchy`, `natural_resource`
- `data/umap_layout_hs92.csv`
  - Keys: `product_hs92_code`
  - Fields: `product_space_x`, `product_space_y`, `product_space_cluster_name`
- `data/top_edges_hs92.csv`
  - Fields: `product_hs92_code_source`, `product_hs92_code_target` (normalize on load if needed)
- `data/hs92_country_year.csv`
  - Keys: `country_iso3_code`, `year`
  - Fields: `export_value` (rename to `export_value_total`), `eci`, `coi`, `diversity`, `growth_proj`

Optional:
- `data/hs92_product_year_4.csv` for cross-year checks (QA only)

### Reading CSV files

Each input CSV is accompanied with a similarly named CSV, postfixed with "_features", that describes the features of the dataset in detail. 
Read these feature files instead of their raw data counterparts to learn about the data.

## Configuration defaults (`config.yaml`)
- `year`: latest available
- `rca_threshold`: 1.0
- `presence_metric`: `rca` | `peer_relative` (default `rca`)
- `fit_recompute`: false (if true, compute `density_recomputed` for QA)
- `similarity_metric`: `cosine_rca` (default), `jaccard_binary` (optional separate run)
- `exclude_natural_resources`: false
- `smoothing_years`: 1 (set 3 to enable trailing average)
- `random_seed`: 42

## High-level approach
- Presence: use `export_rca` (binary at `rca_threshold`); optional peer-relative presence for Metroverse-style comparisons.
- Fit: use `density = 1 − distance` (clipped to [0,1]); optionally recompute density from proximities for validation.
- Product Space: use `umap_layout_hs92` coordinates and `product_space_cluster_name`; edges from `top_edges_hs92`.
- Similarity: cosine on country–product RCA vectors; Jaccard on binary specialization is optional.
- Prioritization: rank opportunities using `density`, `pci`, `cog`, and penalty for presence; provide country context with `eci`, `growth_proj`.
- Validation: correlations between provided and recomputed fit; coverage; light sensitivity on thresholds.

## Pipeline steps

### 0) Setup and logging
- Import `pandas`, `numpy`, `scikit-learn`, `plotly` (or `matplotlib`/`seaborn`).
- Set `random_seed`; configure logging.
- Persist a config snapshot to `outputs/`.

### 1) Load and harmonize
- Load `hs92_country_product_year_4`, filter to `year`.
- Normalize keys/types and column names (snake_case).
- Join `product_hs92` to add `product_name`, `natural_resource`.
- Merge `umap_layout_hs92` for `product_space_x`, `product_space_y`, `product_space_cluster_name`.
- Load `top_edges_hs92` and normalize edge column names.
- Load `hs92_country_year`; rename `export_value` to `export_value_total` and keep `eci`, `growth_proj`, `diversity`, `coi`.
- QA:
  - Assert unique `country_iso3_code`–`product_hs92_code` per year.
  - Verify `distance` in [0,1].
  - Report coverage: share of products from layout present in the panel.

### 2) Presence metrics
- RCA (canonical):
  - Continuous presence: `export_rca`.
  - Binary specialization: `x_binary = 1[export_rca ≥ rca_threshold]`.
- Peer-relative presence (optional):
  - `share_c,p = export_value_c,p / sum_p export_value_c,p`
  - `share_peer_p = sum_{c∈P} export_value_c,p / sum_{c∈P} sum_p export_value_c,p`
  - `rel_presence = share_c,p / max(share_peer_p, eps)`
  - `abs_presence = share_c,p − share_peer_p`
- Persist: `export_value`, `export_rca`, `x_binary`, `share`, `rel_presence`, `abs_presence`.

### 3) Fit metric (technological relatedness)
- Provided fit:
  - `density = clip(1 − distance, 0, 1)`
- Optional recomputation for QA:
  - Build binary matrix `X` from `x_binary`.
  - Compute proximity `phi_{p,q} = min{P(p|q), P(q|p)}` from co-occurrence.
  - `density_recomputed_c,p = sum_q phi_{p,q} x_c,q / sum_q phi_{p,q}`
- QA: per-country correlation between `density` and `density_recomputed`; summarize stats.

### 4) Clusters
- Use `product_space_cluster_name` from layout.
- If absent, leave cluster blank and flag in QA (no recomputation in core).

### 5) Visualizations
- Product Space map:
  - Nodes: `product_space_x`, `product_space_y`; color: `product_space_cluster_name`; size: `log1p(export_rca)` or `log1p(export_value)`.
  - Edges: `top_edges_hs92` (thin for readability if needed).
  - Hover: `product_hs92_code`, `product_name`, `export_rca`, `share`, `density`, `pci`, `cog`.
- Growth Opportunities scatter:
  - X: `density` (or `density_recomputed` if selected).
  - Y: `log1p(export_rca)` (toggle to `rel_presence`).
  - Quadrants: `fit_split = median(density)`; `presence_split = rca_threshold` (or `rel_presence = 1`).

### 6) Opportunity ranking
- Candidate filter:
  - High fit: `density ≥ median(density)` (country-specific).
  - Low presence: `export_rca < rca_threshold` (or `rel_presence < 1`).
- Scoring (single transparent formula):
  - `score = z(density) + 0.5*z(pci) + 0.5*z(cog) − 0.5*z(presence)`
- Optional filter: exclude `natural_resource = True`.
- Output top-N per country.

### 7) Country similarity
- Build `M_rca` with rows=`country_iso3_code`, columns=`product_hs92_code`, values=`export_rca` (fill 0).
- Compute cosine similarity; export full matrix and top-N peers per country.
- Optional: Jaccard similarity on `x_binary` in a separate run.

### 8) Country context and summaries
- Attach `eci`, `growth_proj`, `diversity`, `coi`, `export_value_total`.
- Produce per-country summaries:
  - Top strengths: high presence and high fit.
  - Top opportunities: from step 6.
  - Cluster composition: share of specialized products by `product_space_cluster_name`.

### 9) Validation and sensitivity
- Validation:
  - Correlation between `density` and `density_recomputed` (if computed).
  - Coverage and missingness summaries; `distance` bounds check.
- Sensitivity (lightweight):
  - `rca_threshold` sweep (0.75, 1.0, 1.25) tracking opportunity counts and overlap.
  - Optional trailing average (3-year) for `export_rca` and `density` if `smoothing_years = 3`.
- Log config, library versions, and `random_seed`.

## Outputs
- `outputs/country_product_panel.csv`:
  - `country_iso3_code`, `product_hs92_code`, `year`, `export_value`, `export_rca`, `x_binary`, `share`, `rel_presence`, `abs_presence`, `distance`, `density`, `density_recomputed` (if computed), `pci`, `cog`, `product_space_cluster_name`
- `outputs/similarity_cosine.csv`
- `outputs/product_space_nodes.csv`, `outputs/product_space_edges.csv`
- `outputs/country_summary.csv`:
  - `export_value_total`, `eci`, `growth_proj`, `diversity`, top strengths, top opportunities (IDs and ranks)
- Visuals:
  - Product Space map per country
  - Fit vs presence (Growth Opportunities) scatter

## Minimal code scaffold (illustrative)

- `src/pipeline.py`
  - Parse `config.yaml`; set `YEAR`, toggles, `random_seed`
  - Orchestrate steps; write outputs; log config and QA summaries

- `src/io_load.py`
  - `load_country_product(year)`
  - `load_product_meta()`
  - `load_layout()`
  - `load_edges()`  // normalize edge column names
  - `load_country_year()`  // rename `export_value` → `export_value_total`

- `src/presence.py`
  - `add_rca_binary(df, threshold=1.0)`
  - `add_peer_relative_presence(df, peer_group='global', eps=1e-12)`

- `src/fit.py`
  - `add_density_from_distance(df)`  // `density = clip(1 − distance, 0, 1)`
  - `recompute_density_from_proximity(df_binary)`  // optional QA path

- `src/similarity.py`
  - `country_similarity_cosine(matrix)`
  - `country_similarity_jaccard(binary_matrix)`  // optional run

- `src/viz.py`
  - `plot_product_space(country_df, nodes, edges)`
  - `plot_opportunities_scatter(country_df, use='density', presence='rca')`

## Example snippets

- Load and filter by year:
  - `df = pd.read_csv('data/hs92_country_product_year_4.csv')`
  - `df = df[df['year'] == YEAR].copy()`

- Presence:
  - `df['x_binary'] = (df['export_rca'] >= rca_threshold).astype(int)`
  - `df['share'] = df['export_value'] / df.groupby('country_iso3_code')['export_value'].transform('sum')`

- Fit:
  - `df['density'] = (1.0 - df['distance']).clip(0, 1)`

- Similarity (cosine on RCA):
  - `M = df.pivot(index='country_iso3_code', columns='product_hs92_code', values='export_rca').fillna(0.0)`
  - `from sklearn.metrics.pairwise import cosine_similarity`
  - `S = cosine_similarity(M)`

## Practical notes
- Use `log1p` for skewed variables; cap extremes for readability in visuals.
- Guard against divide-by-zero with a small `eps` for peer-relative presence.
- Keep product keys consistent across joins (`product_hs92_code`).
- Apply optional 3-year trailing averages cautiously; document when used.
- Export only top-N peers if similarity matrices are large.

## Success criteria
- Reproducible country–product panel with presence, fit (provided and/or recomputed for QA), complexity, and clusters.
- Intuitive visuals aligned with Metroverse: Product Space map and Opportunities scatter.
- Robust similarity rankings (cosine), with optional Jaccard check.
- Clear validation (fit correlation), coverage, and sensitivity results logged and exportable.