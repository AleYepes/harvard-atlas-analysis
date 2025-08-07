# Metroverse-like analysis using Atlas HS92 data

## Objective
Build a country–product analytics pipeline that mirrors Metroverse concepts (Relative Presence, Knowledge Clusters, Industry Space, Similarity, Growth Opportunities) using Atlas HS92 data, with a focus on exports, relatedness, and complexity.

## Project structure
- `data/` raw CSVs
- `outputs/` processed CSV artifacts and figures
- `src/` modules:
  - `io_load.py` (load, schema checks, type casting; column renaming for consistency)
  - `presence.py` (RCA and peer-relative presence)
  - `fit.py` (fit from `distance` as provided density; optional recomputation)
  - `similarity.py` (cosine and Jaccard; supports dense or sparse)
  - `clusters.py` (cluster assignment; optional community detection from `top_edges_hs92`)
  - `viz.py` (Product Space, Opportunities scatter, comparison plots)
  - `pipeline.py` (orchestrator; CLI; logging)
- `config.yaml` (analysis year, thresholds, toggles, peer group, smoothing)
- `notebooks/` (EDA, QA, validation)

## Key inputs
- `data/hs92_country_product_year_4.csv`
  - Keys: `country_iso3_code`, `product_hs92_code`, `year`
  - Fields: `export_value`, `global_market_share`, `export_rca`, `distance`, `cog`, `pci`
- `data/product_hs92.csv`
  - Keys: `product_hs92_code`
  - Fields: `product_name`, `product_id_hierarchy`, `show_feasibility`, `natural_resource`
- `data/umap_layout_hs92.csv`
  - Keys: `product_hs92_code`
  - Fields: `product_space_x`, `product_space_y`, `product_space_cluster_name`
- `data/top_edges_hs92.csv`
  - Fields (will be normalized on load): `product_hs92_code_source`, `product_hs92_code_target` (or `source`/`target`)
- `data/hs92_country_year.csv`
  - Keys: `country_iso3_code`, `year`
  - Fields: `export_value` (will be renamed to `export_value_total` on merge), `eci`, `coi`, `diversity`, `growth_proj`
- Optional:
  - `data/hs92_product_year_4.csv` for cross-year checks
  - Services datasets (placeholders): `data/services_unilateral_*`, `data/product_services_unilateral.csv` (verify actual names before use)

## Configuration defaults (`config.yaml`)
- `year`: latest available
- `presence_metric`: `rca` | `peer_relative` (default `rca`)
- `rca_threshold`: 1.0
- `fit_metric`: `distance_provided` | `density_recompute` (default `distance_provided`)
- `similarity_metrics`: list, e.g., `['cosine_rca', 'jaccard_binary']` (default compute and export both)
- `peer_group`: `global` (optional filters by region/income)
- `smoothing_years`: integer window size (default 1; set 3 for temporal smoothing; trailing average)
- `exclude_natural_resources`: false
- `random_seed`: 42 (used for community detection, sampling, and any randomized steps)

## High-level summary
- Use Atlas-native presence and fit first, with transparent toggles:
  - Presence: default to `export_rca`; optional peer-relative presence to mirror Metroverse.
  - Fit: default to `density_provided = 1 − distance`; optionally recompute `density_recomputed` from proximities for validation.
- Product Space: use `data/umap_layout_hs92.csv` coordinates; overlay edges from `data/top_edges_hs92.csv`; use `product_space_cluster_name` clusters if available.
- Similarity: provide cosine on continuous vectors and Jaccard on binary specialization; configurable.
- Prioritization: integrate `pci` and `cog` for opportunity ranking; use `eci`, `growth_proj` for country context.
- Deliverables: reusable artifacts, country dashboards, and QA/validation reports.

## Pipeline steps

### 0) Setup and logging
- Import `pandas`, `numpy`, `scikit-learn`, `networkx`, `plotly` (or `matplotlib`/`seaborn`).
- Set deterministic seeds with `random_seed` and configure logging.

### 1) Load and harmonize
- Load `data/hs92_country_product_year_4.csv`, filter to `year`.
- Cast keys to consistent types; standardize column names to snake_case.
- Join product metadata from `data/product_hs92.csv` (attach `product_name`, `natural_resource`, hierarchy).
- Load `data/umap_layout_hs92.csv` and merge coordinates + `product_space_cluster_name` by `product_hs92_code`.
- Load `data/top_edges_hs92.csv`; if columns are `source`/`target` or `p`/`q`, rename to `product_hs92_code_source`/`product_hs92_code_target` for consistency.
- Load `data/hs92_country_year.csv`, rename `export_value` to `export_value_total` before joining; keep `eci`, `growth_proj`, `diversity`, `coi`.
- QA:
  - Assert no duplicate `country_iso3_code`–`product_hs92_code` rows.
  - Report coverage: share of `product_hs92_code` from layout present in panel.

### 2) Presence metrics
- RCA (default):
  - Use `export_rca` as continuous presence.
  - Create `x_binary = 1[export_rca ≥ rca_threshold]` (pass `rca_threshold` from config).
- Peer-relative presence (toggle):
  - For chosen peer group P:
    - `share_c,p = export_value_c,p / sum_p export_value_c,p`.
    - `share_peer_p = sum_{c∈P} export_value_c,p / sum_{c∈P} sum_p export_value_c,p`.
    - `rel_presence = share_c,p / max(share_peer_p, eps)`; optionally cap to an upper bound for stability in visuals.
    - `abs_presence = share_c,p − share_peer_p`.
  - Optionally define `rel_binary = 1[rel_presence ≥ 1]`.
- Persist: `export_value`, `export_rca`, `x_binary`, `share`, `rel_presence`, `abs_presence`.

### 3) Fit metric (technological relatedness)
- If `fit_metric = distance_provided`:
  - `density_provided_c,p = clip(1 − distance_c,p, 0, 1)`.
  - `density_recomputed = sum_q Φ_{p,q} x_c,q / sum_q Φ_{p,q}`.
- Optional recomputation for validation (`fit_metric = density_recompute` or always compute for QA):
  - Build binary country–product matrix `X` from `x_binary`.
  - Compute product–product proximity `phi_{p,q} = min{P(p|q), P(q|p)}` from co-occurrence in `X`.
  - `density_recomputed_c,p = sum_q phi_{p,q} x_c,q / sum_q phi_{p,q}` for each `c,p`.
- Store `distance`, `density_provided`, and (if computed) `density_recomputed`; report their correlation per country for QA.

### 4) Clusters
- Prefer `product_space_cluster_name` from `data/umap_layout_hs92.csv`.
- If missing or sparse, build a `networkx` graph from `data/top_edges_hs92.csv`, run Louvain/Leiden with `random_seed`, assign:
  - `cluster_id` (integer), and optionally map to a human-readable `cluster_name`.
- Keep a canonical `product_space_cluster_name` (string) alongside `cluster_id` if both exist.

### 5) Product Space visualization scaffold
- Node positions from `data/umap_layout_hs92.csv`: `product_space_x`, `product_space_y`.
- Edge overlay from `data/top_edges_hs92.csv` (optionally sample or threshold for readability).
- Per selected country:
  - Node size: function of `export_rca` or `export_value` (e.g., log1p scaling).
  - Node color: `product_space_cluster_name` (or `cluster_id`).
  - Hover: `product_hs92_code`, `product_name`, `export_rca`, `share`, `density_provided`, `density_recomputed` (if available), `pci`, `cog`.

### 6) Growth Opportunities (fit vs presence)
- X-axis: `density_provided` (or `density_recomputed` if selected).
- Y-axis: presence:
  - Default: `presence = log1p(export_rca)`; or toggle to `presence = rel_presence` (consider log-scale or caps for readability).
- Quadrants:
  - Thresholds: `fit_split = median density` (or fixed 0.5), `presence_split = rca_threshold` or `rel_presence = 1`.
  - Bottom-right (high fit, low presence): opportunities.
- Ranking (explicit):
  - Base score = `w_fit * density + w_pci * z(pci) + w_cog * z(cog) − w_pres * z(presence)`, with defaults such as `w_fit=1.0`, `w_pci=0.5`, `w_cog=0.5`, `w_pres=0.5`. Or use a Pareto filter: keep high-fit/low-presence, then sort by `pci` and `cog`.
  - Optional filters: exclude `natural_resource = True`.

### 7) Country similarity
- Build country–product matrices:
  - Continuous: `M_rca` with `export_rca` or `rel_presence`.
  - Binary: `M_bin` with `x_binary`.
- Compute metrics from `similarity_metrics`:
  - `cosine_rca`: cosine similarity on `M_rca`.
  - `jaccard_binary`: Jaccard similarity on `M_bin`.
  - Use sparse matrices for scalability when needed.
- Output top-N similar countries per country and full similarity matrices for each metric computed.

### 8) Country context
- From `data/hs92_country_year.csv`, attach `eci`, `growth_proj`, `diversity`, `coi`, and `export_value_total` to country summaries.
- Provide quick tables:
  - Top strengths (high presence and fit).
  - Top opportunities (high fit, low presence, high `pci`/`cog`).
  - Cluster composition: share of presence by cluster.

### 9) Services extension (optional)
- Mirror steps 1–8 on services datasets (`data/services_unilateral_*`, `data/product_services_unilateral.csv`).
- Note: relatedness may need to be computed from co-occurrence if no `distance`.

### 10) Validation and sensitivity
- Report correlations between `density_provided` and `density_recomputed` (or between `density_provided` and `1 − distance` if only aliasing).
- Sensitivity tests:
  - Presence threshold (`rca_threshold` variations, e.g., 0.75, 1.0, 1.25).
  - Peer definitions and peer set size.
  - Temporal smoothing (`smoothing_years` ≥ 3) on `export_rca`, `share`, and fit.
- Data QA:
  - Verify `distance` is in [0,1] and interpretability (higher = farther).
  - Coverage checks by year, products, and countries.
  - Missingness and zero-exports handling.
  - Deterministic outputs with logged `random_seed`, library versions, and config snapshot.

### 11) Outputs
- `outputs/country_product_panel.csv`:
  - `country_iso3_code`, `product_hs92_code`, `year`, `export_value`, `export_rca`, `x_binary`, `share`, `rel_presence`, `abs_presence`, `distance`, `density_provided`, `density_recomputed` (if available), `pci`, `cog`, `product_space_cluster_name`, `cluster_id`
- `outputs/similarity_cosine.csv` (if computed), `outputs/similarity_jaccard.csv` (if computed)
- `outputs/product_space_nodes.csv` (with positions and clusters), `outputs/product_space_edges.csv`
- `outputs/country_summary.csv`:
  - Totals (`export_value_total`), `eci`, `growth_proj`, `diversity`, top strengths, top opportunities (IDs and ranks)
- Visuals (static HTML/PNG):
  - Product Space maps per country
  - Fit vs presence scatter with quadrant shading
  - Peer similarity tables

## Minimal code scaffold (illustrative)

- `src/pipeline.py`
  - Parse `config.yaml`, set `YEAR`, toggles, `random_seed`
  - Call functions in sequence; write outputs; log config snapshot

- `src/io_load.py`
  - Functions:
    - `load_country_product(year)`
    - `load_product_meta()`
    - `load_layout()`
    - `load_edges()` (normalize edge column names)
    - `load_country_year()` (rename `export_value` → `export_value_total`)

- `src/presence.py`
  - `add_rca_binary(df, threshold=1.0)`
  - `add_peer_relative_presence(df, peer_group='global', cap=None, eps=1e-12)`

- `src/fit.py`
  - `add_density_provided(df)`  # sets `density_provided = clip(1 - distance, 0, 1)`
  - `recompute_density_from_proximity(df_binary)`  # returns `density_recomputed`

- `src/similarity.py`
  - `country_similarity_cosine(matrix, sparse=False)`
  - `country_similarity_jaccard(binary_matrix, sparse=True)`

- `src/viz.py`
  - `plot_product_space(country_df, nodes, edges)`
  - `plot_opportunities_scatter(country_df, use='density_provided', presence='rca')`

## Example snippets

- Load and filter by year:
  - `df = pd.read_csv('data/hs92_country_product_year_4.csv')`
  - `df = df[df['year'] == YEAR].copy()`

- Presence:
  - `df['x_binary'] = (df['export_rca'] >= rca_threshold).astype(int)`
  - `df['share'] = df['export_value'] / df.groupby('country_iso3_code')['export_value'].transform('sum')`

- Fit:
  - `df['density_provided'] = (1.0 - df['distance']).clip(0, 1)`

- Similarity (cosine on RCA):
  - `M = df.pivot(index='country_iso3_code', columns='product_hs92_code', values='export_rca').fillna(0.0)`
  - `from sklearn.metrics.pairwise import cosine_similarity`
  - `S = cosine_similarity(M)`

## Practical notes
- Use `log1p` for skewed variables in plots; consider capping extrema for readability.
- Guard against divide-by-zero with a small `eps`; consider capping `rel_presence` for outliers.
- Keep product keys consistent across joins (`product_hs92_code`).
- Consider 3-year trailing averages to reduce volatility for opportunity ranking.
- Use sparse matrices for large similarity computations; export only top-N per country if matrices are huge.
- Document toggles and defaults in `config.yaml` and log them in outputs for reproducibility.

## Success criteria
- Reproducible country–product panel with presence, fit (provided and/or recomputed), complexity, and clusters.
- Intuitive visuals: Product Space map and Opportunities scatter that align with Metroverse-style interpretation.
- Robust similarity rankings with configurable metrics.
- Documented methodological toggles (RCA vs peer-relative presence; provided fit vs recomputed density) with validation and sensitivity results.