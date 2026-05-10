"""
01_kml_to_geojson.py
====================
Convert Bengaluru Primary Storm Water Drain (SWD) KML to GeoJSON and Shapefile.

Input  : data/stormwater_drains.kml  (swd_primary layer)
Outputs: data/stormwater_drains.geojson
         data/stormwater_drains.shp   (and sidecar files)
         data/stormwater_drains_stats.csv

Usage:
    pip install geopandas fiona lxml
    python python/01_kml_to_geojson.py
"""

import os
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, mapping
import fiona

# ── Configuration ──────────────────────────────────────────────────────────────
KML_PATH     = os.path.join('data', 'stormwater_drains.kml')
GEOJSON_OUT  = os.path.join('data', 'stormwater_drains.geojson')
SHP_OUT      = os.path.join('data', 'stormwater_drains.shp')
STATS_OUT    = os.path.join('data', 'stormwater_drains_stats.csv')

# Target CRS — WGS84 for GeoJSON, projected for analysis
WGS84        = 'EPSG:4326'
BENGALURU_CRS = 'EPSG:32643'   # UTM Zone 43N


def convert_kml_to_geojson(kml_path: str, geojson_path: str) -> gpd.GeoDataFrame:
    """Read KML with fiona (enable KML driver) and save as GeoJSON."""
    # Enable the KML driver
    fiona.drvsupport.supported_drivers['KML']  = 'rw'
    fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

    print(f"Reading KML: {kml_path}")
    gdf = gpd.read_file(kml_path, driver='KML')
    print(f"  → {len(gdf)} features loaded")
    print(f"  → CRS: {gdf.crs}")
    print(f"  → Columns: {gdf.columns.tolist()}")
    print(f"  → Geometry types: {gdf.geom_type.value_counts().to_dict()}")

    # Ensure WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    else:
        gdf = gdf.to_crs(WGS84)

    # Save GeoJSON
    gdf.to_file(geojson_path, driver='GeoJSON')
    print(f"  ✓ Saved GeoJSON → {geojson_path}")
    return gdf


def extract_attributes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Parse Description HTML (if any) and enrich with geometry metrics."""
    # Project to metres for length calculation
    gdf_proj = gdf.to_crs(BENGALURU_CRS)

    # Compute length in km
    gdf_proj['drain_length_km'] = gdf_proj.geometry.length / 1000.0

    # Compute bounding box area per drain segment
    gdf_proj['bbox_area_m2'] = gdf_proj.geometry.envelope.area

    # Centroid coordinates (back in WGS84 for display)
    gdf_wgs = gdf_proj.to_crs(WGS84)
    gdf_wgs['centroid_lon'] = gdf_proj.geometry.centroid.to_crs(WGS84).x
    gdf_wgs['centroid_lat'] = gdf_proj.geometry.centroid.to_crs(WGS84).y
    gdf_wgs['drain_length_km'] = gdf_proj['drain_length_km'].values

    return gdf_wgs


def add_buffer_zones(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add 50m, 100m, 250m buffer polygons as separate columns."""
    gdf_proj = gdf.to_crs(BENGALURU_CRS)
    gdf_out  = gdf.copy()

    for buf_m in [50, 100, 250]:
        buf_geom = gdf_proj.geometry.buffer(buf_m).to_crs(WGS84)
        gdf_out[f'buffer_{buf_m}m'] = buf_geom.astype(str)  # Stored as WKT

    return gdf_out


def save_shapefile(gdf: gpd.GeoDataFrame, shp_path: str) -> None:
    """Save to Shapefile (column names truncated to 10 chars for DBF)."""
    gdf_shp = gdf[['geometry', 'drain_length_km', 'centroid_lon', 'centroid_lat']].copy()
    gdf_shp.columns = ['geometry', 'len_km', 'cen_lon', 'cen_lat']
    gdf_shp.to_file(shp_path, driver='ESRI Shapefile')
    print(f"  ✓ Saved Shapefile → {shp_path}")


def compute_network_stats(gdf: gpd.GeoDataFrame) -> dict:
    """Compute network-level statistics."""
    gdf_proj = gdf.to_crs(BENGALURU_CRS)

    total_length_km    = gdf_proj.geometry.length.sum() / 1000
    total_features     = len(gdf)
    mean_segment_km    = gdf_proj.geometry.length.mean() / 1000
    max_segment_km     = gdf_proj.geometry.length.max() / 1000
    bbox               = gdf.total_bounds  # [minx, miny, maxx, maxy]

    stats = {
        'total_features':    total_features,
        'total_length_km':   round(total_length_km, 2),
        'mean_segment_km':   round(mean_segment_km, 3),
        'max_segment_km':    round(max_segment_km, 3),
        'bbox_minx':         round(bbox[0], 6),
        'bbox_miny':         round(bbox[1], 6),
        'bbox_maxx':         round(bbox[2], 6),
        'bbox_maxy':         round(bbox[3], 6),
    }
    return stats


def generate_gee_ready_geojson(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    """
    Generate a simplified GeoJSON suitable for uploading to GEE as an asset.
    Only retains geometry + essential properties; simplifies coordinates.
    """
    gdf_simple = gdf[['geometry', 'drain_length_km']].copy()
    # Simplify geometry to reduce file size (tolerance: 0.0001° ≈ 10m)
    gdf_simple.geometry = gdf_simple.geometry.simplify(0.0001, preserve_topology=True)
    out = out_path.replace('.geojson', '_gee.geojson')
    gdf_simple.to_file(out, driver='GeoJSON')
    print(f"  ✓ Saved GEE-ready GeoJSON → {out}")


if __name__ == '__main__':
    print("=" * 60)
    print("Bengaluru SWD KML → GeoJSON Converter")
    print("=" * 60)

    # Step 1: Convert
    gdf = convert_kml_to_geojson(KML_PATH, GEOJSON_OUT)

    # Step 2: Enrich attributes
    gdf = extract_attributes(gdf)

    # Step 3: Save shapefile
    save_shapefile(gdf, SHP_OUT)

    # Step 4: GEE-ready version
    generate_gee_ready_geojson(gdf, GEOJSON_OUT)

    # Step 5: Statistics
    stats = compute_network_stats(gdf)
    print("\nNetwork Statistics:")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    pd.DataFrame([stats]).to_csv(STATS_OUT, index=False)
    print(f"\n  ✓ Stats saved → {STATS_OUT}")

    # Step 6: Save enriched GeoJSON
    enriched_path = GEOJSON_OUT.replace('.geojson', '_enriched.geojson')
    gdf[['geometry', 'drain_length_km', 'centroid_lon', 'centroid_lat']].to_file(
        enriched_path, driver='GeoJSON')
    print(f"  ✓ Enriched GeoJSON → {enriched_path}")

    print("\nDone ✓")
