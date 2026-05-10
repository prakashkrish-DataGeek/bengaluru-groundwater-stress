"""
03_covariate_merge.py
=====================
Merge WRIS groundwater station observations with GEE-extracted
pixel-level covariates (NDVI, LULC, Slope, Elevation, Rainfall).

Workflow:
  1. Load WRIS station CSV (lat/lon + groundwater depth measurements)
  2. Load GEE covariates CSV (bengaluru_covariates.csv from Script 02)
  3. Spatial join: nearest covariate pixel → each station
  4. Enrich with storm-drain proximity (from Script 04)
  5. Export final training dataset for ML model

Input files:
    data/wris_stations.csv          (WRIS groundwater data)
    data/bengaluru_covariates.csv   (GEE output from Script 02)
    data/stormwater_drains.geojson  (from Script 01)

Output:
    data/final_training_data.csv
    data/final_training_data.geojson
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR         = 'data'
WRIS_CSV         = os.path.join(DATA_DIR, 'wris_stations.csv')
COVARIATES_CSV   = os.path.join(DATA_DIR, 'bengaluru_covariates.csv')
DRAINS_GEOJSON   = os.path.join(DATA_DIR, 'stormwater_drains.geojson')
LINEAMENTS_GEOJSON = os.path.join(DATA_DIR, 'lineaments.geojson')

OUT_CSV          = os.path.join(DATA_DIR, 'final_training_data.csv')
OUT_GEOJSON      = os.path.join(DATA_DIR, 'final_training_data.geojson')

BENGALURU_CRS    = 'EPSG:32643'   # UTM Zone 43N — metres
WGS84            = 'EPSG:4326'

# ── Sample WRIS data generator (replace with real WRIS download) ──────────────
def create_sample_wris_data() -> pd.DataFrame:
    """
    Creates sample WRIS-style groundwater station data.
    Replace this function with actual WRIS CSV when available.
    Download from: https://indiawris.gov.in/wris/#/groundWater
    """
    rng = np.random.default_rng(42)
    n = 150
    # Scatter stations across Bengaluru
    lons = rng.uniform(77.45, 77.78, n)
    lats = rng.uniform(12.82, 13.12, n)

    # Simulate groundwater depth (mbgl) — typical Bengaluru range 5–60m
    gw_depth = 10 + 30 * rng.beta(2, 3, n) + 5 * rng.normal(0, 1, n)
    gw_depth = np.clip(gw_depth, 2, 80)

    # Simulate seasonal variation (pre/post monsoon)
    seasonal_fluctuation = rng.uniform(-3, 8, n)

    df = pd.DataFrame({
        'station_id':           [f'KAR_{i:04d}' for i in range(1, n+1)],
        'station_name':         [f'Bengaluru_Station_{i}' for i in range(1, n+1)],
        'latitude':             lats,
        'longitude':            lons,
        'gw_depth_mbgl':        gw_depth.round(2),
        'seasonal_fluctuation': seasonal_fluctuation.round(2),
        'year':                 rng.choice([2022, 2023, 2024], n),
        'aquifer_type':         rng.choice(['Fractured', 'Alluvial', 'Weathered'], n),
    })
    print(f"  [NOTE] Using synthetic WRIS data ({n} stations).")
    print("         Replace with real data from: https://indiawris.gov.in")
    return df


# ── Core Functions ─────────────────────────────────────────────────────────────

def load_stations(csv_path: str) -> gpd.GeoDataFrame:
    """Load WRIS station CSV and convert to GeoDataFrame."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  Loaded WRIS stations: {len(df)} rows")
    else:
        print(f"  WRIS CSV not found at {csv_path} — using synthetic data")
        df = create_sample_wris_data()

    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84)
    return gdf


def load_covariates(csv_path: str) -> gpd.GeoDataFrame:
    """Load GEE covariate CSV (output from Script 02) and convert to GeoDataFrame."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  Loaded covariates: {len(df)} rows")
    else:
        # Simulate covariates if not available
        print(f"  Covariates CSV not found at {csv_path} — generating mock data")
        rng = np.random.default_rng(0)
        n = 5000
        df = pd.DataFrame({
            'longitude':       rng.uniform(77.40, 77.82, n),
            'latitude':        rng.uniform(12.77, 13.18, n),
            'NDVI':            rng.uniform(-0.1, 0.8, n).round(4),
            'EVI':             rng.uniform(-0.05, 0.6, n).round(4),
            'MNDWI':           rng.uniform(-0.6, 0.5, n).round(4),
            'LULC':            rng.choice([1,2,4,5,7,8,11], n),
            'Slope':           rng.exponential(5, n).round(2),
            'Elevation_m':     (870 + rng.normal(0, 30, n)).round(1),
            'Annual_Rainfall_mm': rng.normal(900, 150, n).round(0),
        })

    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84)
    return gdf


def merge_with_covariates(stations_gdf: gpd.GeoDataFrame,
                           covariates_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Spatial join: find nearest covariate pixel for each WRIS station.
    This gives each ground station its corresponding NDVI, LULC, Slope context.
    """
    print("  Running spatial join (stations → nearest covariate pixel)...")
    merged = gpd.sjoin_nearest(
        stations_gdf,
        covariates_gdf.drop(columns=['longitude', 'latitude'], errors='ignore'),
        distance_col='pixel_dist_deg',
        how='left'
    )
    # Rename conflicting index columns
    if 'index_right' in merged.columns:
        merged = merged.drop(columns=['index_right'])
    print(f"  → Merged: {len(merged)} records")
    return merged


def add_drain_proximity(gdf: gpd.GeoDataFrame, drains_path: str) -> gpd.GeoDataFrame:
    """Calculate distance (m) from each station to the nearest storm drain."""
    if not os.path.exists(drains_path):
        print(f"  [SKIP] Drains GeoJSON not found at {drains_path}")
        gdf['dist_to_drain_m'] = np.nan
        return gdf

    print("  Computing distance to nearest storm drain...")
    drains = gpd.read_file(drains_path)

    # Project both to metres
    gdf_proj    = gdf.to_crs(BENGALURU_CRS)
    drains_proj = drains.to_crs(BENGALURU_CRS)

    # Union all drain geometries for efficient distance computation
    drains_union = drains_proj.geometry.union_all()

    gdf_proj['dist_to_drain_m'] = gdf_proj.geometry.apply(
        lambda x: x.distance(drains_union)
    ).round(1)

    # Back to WGS84
    gdf['dist_to_drain_m'] = gdf_proj['dist_to_drain_m'].values
    print(f"  → Mean drain distance: {gdf['dist_to_drain_m'].mean():.0f} m")
    return gdf


def add_lineament_proximity(gdf: gpd.GeoDataFrame, lineaments_path: str) -> gpd.GeoDataFrame:
    """Calculate distance (m) from each station to the nearest lineament."""
    if not os.path.exists(lineaments_path):
        print(f"  [SKIP] Lineaments GeoJSON not found at {lineaments_path}")
        gdf['dist_to_lineament_m'] = np.nan
        return gdf

    print("  Computing distance to nearest lineament...")
    lineaments    = gpd.read_file(lineaments_path)
    gdf_proj      = gdf.to_crs(BENGALURU_CRS)
    lin_proj      = lineaments.to_crs(BENGALURU_CRS)
    lin_union     = lin_proj.geometry.union_all()

    gdf_proj['dist_to_lineament_m'] = gdf_proj.geometry.apply(
        lambda x: x.distance(lin_union)
    ).round(1)
    gdf['dist_to_lineament_m'] = gdf_proj['dist_to_lineament_m'].values
    return gdf


def engineer_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create derived features useful for ML groundwater depth prediction."""
    gdf = gdf.copy()

    # LULC binary flags
    gdf['is_urban']      = (gdf.get('LULC', np.nan) == 7).astype(int)
    gdf['is_vegetated']  = (gdf.get('LULC', np.nan).isin([2, 4, 5, 11])).astype(int) \
                            if hasattr(gdf.get('LULC', pd.Series()), 'isin') else 0

    # Topographic wetness proxy: low elevation deviation from mean + low slope
    if 'Elevation_m' in gdf.columns:
        elev_mean = gdf['Elevation_m'].mean()
        gdf['elev_deviation'] = (gdf['Elevation_m'] - elev_mean).abs()

    # Drought stress proxy: low NDVI + low rainfall
    if 'NDVI' in gdf.columns and 'Annual_Rainfall_mm' in gdf.columns:
        ndvi_norm  = (gdf['NDVI'] - gdf['NDVI'].min()) / (gdf['NDVI'].max() - gdf['NDVI'].min() + 1e-9)
        rain_norm  = (gdf['Annual_Rainfall_mm'] - gdf['Annual_Rainfall_mm'].min()) / \
                     (gdf['Annual_Rainfall_mm'].max() - gdf['Annual_Rainfall_mm'].min() + 1e-9)
        gdf['drought_stress_index'] = (1 - ndvi_norm) * 0.5 + (1 - rain_norm) * 0.5

    # Recharge potential score (simple)
    if 'Slope' in gdf.columns and 'NDVI' in gdf.columns:
        slope_score = 1 / (1 + gdf['Slope'].fillna(10))
        ndvi_score  = gdf['NDVI'].clip(0, 1).fillna(0)
        gdf['recharge_score'] = ((slope_score + ndvi_score) / 2).round(4)

    return gdf


def summarise_training_data(gdf: gpd.GeoDataFrame) -> None:
    """Print summary statistics of the final training dataset."""
    print("\n" + "=" * 50)
    print("Final Training Dataset Summary")
    print("=" * 50)
    print(f"  Total samples : {len(gdf)}")
    print(f"  Features      : {len(gdf.columns) - 1}")

    numeric_cols = gdf.select_dtypes(include=np.number).columns
    print(f"\n  Key variable statistics:")
    print(gdf[numeric_cols[:8]].describe().round(2).to_string())


if __name__ == '__main__':
    print("=" * 60)
    print("Covariate Merge Pipeline")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1] Loading WRIS stations...")
    stations = load_stations(WRIS_CSV)

    print("\n[2] Loading GEE covariates...")
    covariates = load_covariates(COVARIATES_CSV)

    # Step 3: Spatial join
    print("\n[3] Merging stations with covariates...")
    merged = merge_with_covariates(stations, covariates)

    # Step 4: Add drain proximity
    print("\n[4] Adding drain proximity...")
    merged = add_drain_proximity(merged, DRAINS_GEOJSON)

    # Step 5: Add lineament proximity
    print("\n[5] Adding lineament proximity...")
    merged = add_lineament_proximity(merged, LINEAMENTS_GEOJSON)

    # Step 6: Feature engineering
    print("\n[6] Engineering derived features...")
    merged = engineer_features(merged)

    # Step 7: Summary
    summarise_training_data(merged)

    # Step 8: Save
    print(f"\n[7] Saving outputs...")
    merged.to_csv(OUT_CSV, index=False)
    print(f"  ✓ CSV → {OUT_CSV}")

    # GeoJSON — only keep essential columns
    geo_cols = ['geometry', 'station_id', 'latitude', 'longitude',
                'gw_depth_mbgl', 'NDVI', 'LULC', 'Slope', 'Elevation_m',
                'Annual_Rainfall_mm', 'dist_to_drain_m', 'recharge_score']
    available = [c for c in geo_cols if c in merged.columns]
    merged[available].to_file(OUT_GEOJSON, driver='GeoJSON')
    print(f"  ✓ GeoJSON → {OUT_GEOJSON}")

    print("\nDone ✓")
