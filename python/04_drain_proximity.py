"""
04_drain_proximity.py
=====================
Calculate proximity of groundwater prediction grid points to the
primary storm water drain (SWD) network of Bengaluru.

Distance to drains is a key covariate because:
  • Drains channel surface runoff → reduce recharge
  • Points close to drains may have shallow water table
  • Drain density correlates with urbanisation level

Input:
    data/stormwater_drains.geojson   (from Script 01)
    data/final_training_data.csv     (from Script 03)

Outputs:
    data/drain_proximity_grid.geojson   (500m grid with drain distances)
    data/drain_proximity_grid.csv
    data/drain_buffer_zones.geojson     (50m / 100m / 250m buffer polygons)
    figures/drain_proximity_map.png
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Point, box
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR        = 'data'
FIGURES_DIR     = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

DRAINS_GEOJSON  = os.path.join(DATA_DIR, 'stormwater_drains.geojson')
STATIONS_CSV    = os.path.join(DATA_DIR, 'final_training_data.csv')

GRID_OUT_GJ     = os.path.join(DATA_DIR, 'drain_proximity_grid.geojson')
GRID_OUT_CSV    = os.path.join(DATA_DIR, 'drain_proximity_grid.csv')
BUFFER_OUT      = os.path.join(DATA_DIR, 'drain_buffer_zones.geojson')
FIGURE_OUT      = os.path.join(FIGURES_DIR, 'drain_proximity_map.png')

BENGALURU_BBOX  = (77.40, 12.77, 77.82, 13.18)   # (minlon, minlat, maxlon, maxlat)
GRID_SPACING_M  = 500                             # 500m analysis grid
BENGALURU_CRS   = 'EPSG:32643'
WGS84           = 'EPSG:4326'
BUFFER_DISTANCES = [50, 100, 250, 500]


# ── Functions ─────────────────────────────────────────────────────────────────

def load_drains(path: str) -> gpd.GeoDataFrame:
    """Load drain network; fall back to sample geometry if file missing."""
    if os.path.exists(path):
        gdf = gpd.read_file(path)
        print(f"  Loaded {len(gdf)} drain segments")
        return gdf
    else:
        print(f"  [WARN] {path} not found — creating synthetic drain network")
        from shapely.geometry import LineString
        # Simulate major drain corridors across Bengaluru
        lines = [
            LineString([(77.55, 12.90), (77.60, 12.95), (77.65, 13.00)]),
            LineString([(77.45, 12.85), (77.52, 12.90), (77.58, 12.92)]),
            LineString([(77.60, 12.82), (77.62, 12.88), (77.63, 12.95)]),
            LineString([(77.70, 12.90), (77.72, 12.98), (77.75, 13.05)]),
            LineString([(77.48, 13.00), (77.55, 13.05), (77.62, 13.10)]),
        ]
        return gpd.GeoDataFrame({'geometry': lines, 'drain_length_km': [2.5, 3.1, 2.0, 3.8, 2.7]},
                                 crs=WGS84)


def create_analysis_grid(bbox: tuple, spacing_m: int,
                          crs_proj: str = BENGALURU_CRS) -> gpd.GeoDataFrame:
    """Create a regular grid of analysis points within the bounding box."""
    minlon, minlat, maxlon, maxlat = bbox

    # Convert bbox to projected CRS for even spacing
    bbox_gdf = gpd.GeoDataFrame(
        {'geometry': [box(minlon, minlat, maxlon, maxlat)]},
        crs=WGS84
    ).to_crs(crs_proj)

    bounds = bbox_gdf.total_bounds   # [minx, miny, maxx, maxy] in metres
    xs = np.arange(bounds[0], bounds[2], spacing_m)
    ys = np.arange(bounds[1], bounds[3], spacing_m)
    xx, yy = np.meshgrid(xs, ys)

    points = [Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())]
    grid = gpd.GeoDataFrame({'geometry': points}, crs=crs_proj)
    grid = grid.to_crs(WGS84)
    grid['longitude'] = grid.geometry.x
    grid['latitude']  = grid.geometry.y
    print(f"  Created analysis grid: {len(grid):,} points ({spacing_m}m spacing)")
    return grid


def calculate_drain_distance_kdtree(grid_gdf: gpd.GeoDataFrame,
                                     drains_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Fast distance calculation using cKDTree.
    Densifies drain geometries into point clouds for efficient lookup.
    """
    drains_proj = drains_gdf.to_crs(BENGALURU_CRS)
    grid_proj   = grid_gdf.to_crs(BENGALURU_CRS)

    # Sample points along drain lines (every 50m)
    drain_points = []
    for geom in drains_proj.geometry:
        if geom.geom_type == 'LineString':
            length = geom.length
            n_pts  = max(2, int(length / 50))
            for i in range(n_pts + 1):
                pt = geom.interpolate(i / n_pts, normalized=True)
                drain_points.append((pt.x, pt.y))
        elif geom.geom_type in ('MultiLineString', 'MultiGeometry'):
            for part in geom.geoms:
                if hasattr(part, 'length'):
                    length = part.length
                    n_pts = max(2, int(length / 50))
                    for i in range(n_pts + 1):
                        pt = part.interpolate(i / n_pts, normalized=True)
                        drain_points.append((pt.x, pt.y))

    if not drain_points:
        grid_gdf['dist_to_drain_m'] = np.nan
        return grid_gdf

    drain_arr = np.array(drain_points)
    tree      = cKDTree(drain_arr)

    grid_coords = np.column_stack([
        grid_proj.geometry.x,
        grid_proj.geometry.y
    ])

    distances, _ = tree.query(grid_coords, k=1)
    grid_gdf = grid_gdf.copy()
    grid_gdf['dist_to_drain_m'] = distances.round(1)

    print(f"  Distance stats (m): min={distances.min():.0f}, "
          f"mean={distances.mean():.0f}, max={distances.max():.0f}")
    return grid_gdf


def classify_drain_zones(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Classify points into drain influence zones."""
    gdf = gdf.copy()
    gdf['drain_zone'] = 'Distant (>500m)'
    gdf.loc[gdf['dist_to_drain_m'] <= 500,  'drain_zone'] = 'Near (100–500m)'
    gdf.loc[gdf['dist_to_drain_m'] <= 100,  'drain_zone'] = 'Adjacent (50–100m)'
    gdf.loc[gdf['dist_to_drain_m'] <= 50,   'drain_zone'] = 'Immediate (<50m)'

    # Influence score: 1 (distant) to 4 (immediate)
    zone_map = {'Distant (>500m)': 1, 'Near (100–500m)': 2,
                'Adjacent (50–100m)': 3, 'Immediate (<50m)': 4}
    gdf['drain_influence_score'] = gdf['drain_zone'].map(zone_map)
    return gdf


def create_buffer_polygons(drains_gdf: gpd.GeoDataFrame,
                            distances: list) -> gpd.GeoDataFrame:
    """Create buffer zones around drain network."""
    drains_proj = drains_gdf.to_crs(BENGALURU_CRS)
    records = []
    for dist in distances:
        buf = drains_proj.geometry.buffer(dist).union_all()
        records.append({'geometry': buf, 'buffer_m': dist, 'label': f'{dist}m buffer'})

    gdf_buf = gpd.GeoDataFrame(records, crs=BENGALURU_CRS).to_crs(WGS84)
    return gdf_buf


def calculate_lake_effect(grid_gdf: gpd.GeoDataFrame,
                           drainage_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Original function from Storm water drains RTF — preserved and extended.
    For every point in the grid, calculate distance to the nearest drain.
    Uses projected CRS (EPSG:32643) for metre-based distance.
    """
    grid_gdf     = grid_gdf.to_crs(BENGALURU_CRS)
    drainage_gdf = drainage_gdf.to_crs(BENGALURU_CRS)

    # Calculate distance to nearest drainage line
    drainage_union = drainage_gdf.geometry.union_all()
    grid_gdf['dist_to_drain'] = grid_gdf.geometry.apply(
        lambda x: drainage_union.distance(x)
    )
    return grid_gdf.to_crs(WGS84)


def visualise_drain_proximity(grid_gdf: gpd.GeoDataFrame,
                               drains_gdf: gpd.GeoDataFrame,
                               stations_gdf: gpd.GeoDataFrame | None = None) -> None:
    """Create a 2-panel proximity map."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle('Bengaluru Primary Storm Water Drain Network\nProximity Analysis',
                 fontsize=14, fontweight='bold')

    # Panel 1: Distance heatmap
    ax = axes[0]
    sc = ax.scatter(grid_gdf['longitude'], grid_gdf['latitude'],
                    c=grid_gdf['dist_to_drain_m'], cmap='YlOrRd_r',
                    s=4, alpha=0.7, vmin=0, vmax=2000)
    drains_proj = drains_gdf
    drains_proj.plot(ax=ax, color='blue', linewidth=0.8, label='Drains', zorder=3)
    plt.colorbar(sc, ax=ax, label='Distance to nearest drain (m)')
    ax.set_title('Distance to Drain (m)')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(loc='upper right', fontsize=8)

    # Panel 2: Zone classification
    ax = axes[1]
    zone_colors = {
        'Immediate (<50m)': '#d73027',
        'Adjacent (50–100m)': '#fc8d59',
        'Near (100–500m)': '#fee090',
        'Distant (>500m)': '#2b83ba'
    }
    for zone, color in zone_colors.items():
        sub = grid_gdf[grid_gdf['drain_zone'] == zone]
        if len(sub) > 0:
            ax.scatter(sub['longitude'], sub['latitude'],
                       c=color, s=4, alpha=0.6, label=f'{zone} (n={len(sub):,})')
    drains_gdf.plot(ax=ax, color='black', linewidth=0.8, zorder=5)
    ax.set_title('Drain Influence Zones')
    ax.set_xlabel('Longitude')
    ax.legend(loc='upper right', fontsize=7, markerscale=3)

    if stations_gdf is not None and 'latitude' in stations_gdf.columns:
        for ax_ in axes:
            ax_.scatter(stations_gdf['longitude'], stations_gdf['latitude'],
                        c='black', s=20, marker='*', zorder=6, label='GW Stations')

    plt.tight_layout()
    plt.savefig(FIGURE_OUT, dpi=150, bbox_inches='tight')
    print(f"  ✓ Figure → {FIGURE_OUT}")
    plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("Drain Proximity Analysis Pipeline")
    print("=" * 60)

    # Step 1: Load drains
    print("\n[1] Loading storm water drains...")
    drains = load_drains(DRAINS_GEOJSON)

    # Step 2: Create analysis grid
    print("\n[2] Creating 500m analysis grid...")
    grid = create_analysis_grid(BENGALURU_BBOX, GRID_SPACING_M)

    # Step 3: Compute distances (fast KDTree method)
    print("\n[3] Computing drain distances (KDTree)...")
    grid = calculate_drain_distance_kdtree(grid, drains)

    # Step 4: Zone classification
    print("\n[4] Classifying drain influence zones...")
    grid = classify_drain_zones(grid)
    print(grid['drain_zone'].value_counts().to_string())

    # Step 5: Buffer polygons
    print("\n[5] Generating buffer zones...")
    buffers = create_buffer_polygons(drains, BUFFER_DISTANCES)
    buffers.to_file(BUFFER_OUT, driver='GeoJSON')
    print(f"  ✓ Buffer zones → {BUFFER_OUT}")

    # Step 6: Load stations if available
    stations_gdf = None
    if os.path.exists(STATIONS_CSV):
        df_st = pd.read_csv(STATIONS_CSV)
        if 'longitude' in df_st.columns and 'latitude' in df_st.columns:
            from shapely.geometry import Point as Pt
            stations_gdf = gpd.GeoDataFrame(
                df_st,
                geometry=[Pt(r['longitude'], r['latitude']) for _, r in df_st.iterrows()],
                crs=WGS84
            )

    # Step 7: Visualise
    print("\n[6] Generating visualisation...")
    visualise_drain_proximity(grid, drains, stations_gdf)

    # Step 8: Save outputs
    print("\n[7] Saving outputs...")
    grid.to_file(GRID_OUT_GJ, driver='GeoJSON')
    print(f"  ✓ Grid GeoJSON → {GRID_OUT_GJ}")

    grid[['longitude', 'latitude', 'dist_to_drain_m',
          'drain_zone', 'drain_influence_score']].to_csv(GRID_OUT_CSV, index=False)
    print(f"  ✓ Grid CSV → {GRID_OUT_CSV}")

    # Step 9: Summary statistics
    print("\nSummary:")
    print(f"  Grid points    : {len(grid):,}")
    print(f"  Drain segments : {len(drains):,}")
    mean_d = grid['dist_to_drain_m'].mean()
    pct50  = grid['dist_to_drain_m'].quantile(0.50)
    pct95  = grid['dist_to_drain_m'].quantile(0.95)
    print(f"  Mean distance  : {mean_d:.0f} m")
    print(f"  Median distance: {pct50:.0f} m")
    print(f"  95th pct dist  : {pct95:.0f} m")

    print("\nDone ✓")
