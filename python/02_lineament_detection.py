"""
02_lineament_detection.py
=========================
Automated Lineament / Fracture Detection for Bengaluru using:
  1. NASADEM hillshade (multi-azimuth illumination)
  2. Canny edge detection
  3. Probabilistic Hough Line Transform
  4. Direction-filtered lineament clustering
  5. Export as GeoJSON + heatmap raster

Lineaments are geological fractures / faults that act as
preferential groundwater flow conduits — critical for GWPZ modeling.

Dependencies:
    pip install numpy scipy scikit-image rasterio shapely geopandas matplotlib opencv-python-headless
    pip install earthengine-api requests

Usage:
    python python/02_lineament_detection.py --dem data/nasadem_bengaluru.tif
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio not found — synthetic DEM will be used for demonstration")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] opencv not found — skimage-only mode")

from skimage import feature, filters, morphology, exposure
from skimage.transform import probabilistic_hough_line
from scipy.ndimage import gaussian_filter, sobel
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
BENGALURU_BBOX = (77.40, 12.77, 77.82, 13.18)   # (minlon, minlat, maxlon, maxlat)
OUTPUT_DIR     = Path('data')
OUTPUT_DIR.mkdir(exist_ok=True)

# Canny parameters
CANNY_SIGMA    = 2.0
CANNY_LOW      = 0.05
CANNY_HIGH     = 0.15

# Hough parameters
HOUGH_THRESHOLD  = 10   # minimum votes
HOUGH_LINE_LEN   = 50   # minimum line length in pixels
HOUGH_LINE_GAP   = 10   # maximum gap between line segments
HOUGH_THETA_STEP = np.pi / 360

# Azimuth illumination angles for multi-directional hillshade
AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]


# ── Helper Functions ──────────────────────────────────────────────────────────

def create_synthetic_bengaluru_dem(rows=512, cols=512):
    """
    Generate a synthetic DEM representative of Bengaluru's Deccan Plateau
    (elevation ~900m, gentle undulation, rocky outcrops).
    Used when real NASADEM is unavailable.
    """
    rng = np.random.default_rng(42)
    x = np.linspace(0, 4 * np.pi, cols)
    y = np.linspace(0, 4 * np.pi, rows)
    xx, yy = np.meshgrid(x, y)

    # Base plateau
    dem = 900 + 30 * np.sin(0.3 * xx) + 25 * np.cos(0.2 * yy)
    # Add ridge-like lineaments
    dem += 15 * np.exp(-((yy - 2) ** 2) / 0.5)   # NE-SW ridge
    dem += 12 * np.exp(-((xx - 3) ** 2) / 0.3)   # NW-SE fault
    # Random rocky noise
    dem += rng.normal(0, 3, (rows, cols))
    dem = gaussian_filter(dem, sigma=2)
    return dem.astype(np.float32)


def compute_hillshade(dem: np.ndarray, azimuth_deg: float = 315,
                      altitude_deg: float = 45) -> np.ndarray:
    """Compute hillshade from DEM array."""
    azimuth_rad  = np.radians(360 - azimuth_deg + 90)
    altitude_rad = np.radians(altitude_deg)

    # Compute gradients
    dy, dx = np.gradient(dem)
    slope  = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)

    # Hillshade formula
    hs = (np.cos(altitude_rad) * np.cos(slope) +
          np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    return np.clip(hs, 0, 1)


def multi_directional_hillshade(dem: np.ndarray) -> np.ndarray:
    """Average hillshade across all azimuths for maximum edge revelation."""
    hs_stack = [compute_hillshade(dem, az) for az in AZIMUTHS]
    return np.mean(hs_stack, axis=0)


def detect_edges_canny(image: np.ndarray) -> np.ndarray:
    """Apply Canny edge detection with histogram equalization."""
    # CLAHE contrast enhancement
    image_eq = exposure.equalize_adapthist(image, clip_limit=0.03)
    # Gaussian smoothing
    image_smooth = gaussian_filter(image_eq, sigma=CANNY_SIGMA)
    # Canny edges
    edges = feature.canny(
        image_smooth,
        sigma=CANNY_SIGMA,
        low_threshold=CANNY_LOW,
        high_threshold=CANNY_HIGH
    )
    # Morphological thinning for cleaner lines
    edges = morphology.skeletonize(edges)
    return edges


def detect_lineaments_hough(edges: np.ndarray) -> list:
    """
    Apply probabilistic Hough Transform to detect line segments.
    Returns list of ((x0,y0),(x1,y1)) pixel coordinates.
    """
    lines = probabilistic_hough_line(
        edges,
        threshold=HOUGH_THRESHOLD,
        line_length=HOUGH_LINE_LEN,
        line_gap=HOUGH_LINE_GAP,
        theta=np.linspace(-np.pi/2, np.pi/2, 360)
    )
    return lines


def pixel_to_geo(px: int, py: int, bbox: tuple, shape: tuple) -> tuple:
    """Convert pixel (col, row) to geographic coordinates."""
    rows, cols = shape
    minlon, minlat, maxlon, maxlat = bbox
    lon = minlon + (px / cols) * (maxlon - minlon)
    lat = maxlat - (py / rows) * (maxlat - minlat)   # y-axis flipped
    return lon, lat


def lines_to_geodataframe(lines: list, bbox: tuple, shape: tuple) -> gpd.GeoDataFrame:
    """Convert pixel-space line segments to a GeoDataFrame."""
    records = []
    for i, (p0, p1) in enumerate(lines):
        lon0, lat0 = pixel_to_geo(p0[0], p0[1], bbox, shape)
        lon1, lat1 = pixel_to_geo(p1[0], p1[1], bbox, shape)
        geom = LineString([(lon0, lat0), (lon1, lat1)])

        # Compute azimuth
        dlon = lon1 - lon0
        dlat = lat1 - lat0
        azimuth = np.degrees(np.arctan2(dlon, dlat)) % 180

        records.append({
            'geometry':    geom,
            'azimuth_deg': round(azimuth, 1),
            'length_deg':  round(geom.length, 6),
            'lineament_id': i + 1
        })

    gdf = gpd.GeoDataFrame(records, crs='EPSG:4326')
    # Project to metres for length calculation
    gdf_proj = gdf.to_crs('EPSG:32643')
    gdf['length_m'] = gdf_proj.geometry.length
    return gdf


def classify_lineament_direction(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Classify lineaments by tectonic direction.
    Bengaluru: Primary NNW-SSE (Closepet Granite), Secondary NE-SW (shear zones)
    """
    def direction_class(az):
        if   (az >= 0   and az < 22.5)  or az >= 157.5: return 'N-S'
        elif  az >= 22.5 and az < 67.5:                  return 'NE-SW'
        elif  az >= 67.5 and az < 112.5:                 return 'E-W'
        else:                                             return 'NW-SE'

    gdf['direction'] = gdf['azimuth_deg'].apply(direction_class)
    return gdf


def compute_lineament_density(gdf: gpd.GeoDataFrame, bbox: tuple,
                               resolution: float = 0.005) -> np.ndarray:
    """
    Compute lineament density raster (km of lineament per km²).
    Higher density → better fracture connectivity → better GW recharge.
    """
    minlon, minlat, maxlon, maxlat = bbox
    cols = int((maxlon - minlon) / resolution)
    rows = int((maxlat - minlat) / resolution)

    density = np.zeros((rows, cols), dtype=np.float32)

    for _, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        for j in range(len(coords) - 1):
            lon0, lat0 = coords[j]
            lon1, lat1 = coords[j+1]
            # Rasterize: find grid cells the line passes through
            col0 = int((lon0 - minlon) / resolution)
            row0 = int((maxlat - lat0) / resolution)
            col1 = int((lon1 - minlon) / resolution)
            row1 = int((maxlat - lat1) / resolution)
            # Clamp
            col0 = max(0, min(cols-1, col0))
            row0 = max(0, min(rows-1, row0))
            col1 = max(0, min(cols-1, col1))
            row1 = max(0, min(rows-1, row1))
            density[row0, col0] += 1
            density[row1, col1] += 1

    # Smooth density
    density = gaussian_filter(density, sigma=3)
    return density


def save_density_raster(density: np.ndarray, bbox: tuple, out_path: str) -> None:
    """Save density raster as GeoTIFF."""
    if not HAS_RASTERIO:
        print(f"  [SKIP] rasterio not available — skipping raster save")
        return
    minlon, minlat, maxlon, maxlat = bbox
    rows, cols = density.shape
    transform = from_bounds(minlon, minlat, maxlon, maxlat, cols, rows)
    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=rows, width=cols,
        count=1,
        dtype=density.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform
    ) as dst:
        dst.write(density, 1)
    print(f"  ✓ Density raster → {out_path}")


def visualise_results(dem, hillshade_multi, edges, lines, gdf, bbox):
    """Create a comprehensive 6-panel figure."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Bengaluru Lineament / Fracture Detection\n'
                 'Source: NASADEM + Probabilistic Hough Transform',
                 fontsize=16, fontweight='bold')

    # Panel 1: DEM
    ax = axes[0, 0]
    im = ax.imshow(dem, cmap='terrain', aspect='auto')
    plt.colorbar(im, ax=ax, label='Elevation (m)')
    ax.set_title('NASADEM Elevation')

    # Panel 2: Multi-directional hillshade
    ax = axes[0, 1]
    ax.imshow(hillshade_multi, cmap='gray', aspect='auto')
    ax.set_title('Multi-directional Hillshade\n(8 azimuths)')

    # Panel 3: Canny edges
    ax = axes[0, 2]
    ax.imshow(edges, cmap='gray', aspect='auto')
    ax.set_title(f'Canny Edges (σ={CANNY_SIGMA})')

    # Panel 4: Detected lineaments on hillshade
    ax = axes[1, 0]
    ax.imshow(hillshade_multi, cmap='gray', aspect='auto')
    for p0, p1 in lines:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=0.8, alpha=0.7)
    ax.set_title(f'Detected Lineaments\n({len(lines)} segments)')

    # Panel 5: Azimuth rose diagram
    ax = axes[1, 1]
    ax.set_aspect('equal')
    directions = gdf['azimuth_deg'].values
    bins = np.linspace(0, 180, 13)
    counts, _ = np.histogram(directions, bins=bins)
    theta = np.deg2rad(bins[:-1] + 7.5)
    # Mirror for rose
    theta_full  = np.concatenate([theta, theta + np.pi])
    counts_full = np.concatenate([counts, counts])
    ax2 = fig.add_subplot(2, 3, 5, polar=True)
    ax2.bar(theta_full, counts_full, width=np.deg2rad(14), alpha=0.6, color='steelblue')
    ax2.set_title('Lineament Rose Diagram\n(azimuth distribution)', pad=20)
    axes[1, 1].remove()

    # Panel 6: Direction map
    ax = axes[1, 2]
    colors_map = {'N-S': 'blue', 'NE-SW': 'red', 'E-W': 'green', 'NW-SE': 'orange'}
    for dir_name, color in colors_map.items():
        sub = gdf[gdf['direction'] == dir_name]
        for _, row in sub.iterrows():
            coords = list(row.geometry.coords)
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            # Map geo → pixel
            rows_n, cols_n = dem.shape
            minlon, minlat, maxlon, maxlat = bbox
            pxs = [(x - minlon) / (maxlon - minlon) * cols_n for x in xs]
            pys = [(maxlat - y) / (maxlat - minlat) * rows_n for y in ys]
            ax.plot(pxs, pys, color=color, linewidth=0.8, label=dir_name, alpha=0.8)
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=8)
    ax.set_xlim(0, dem.shape[1])
    ax.set_ylim(dem.shape[0], 0)
    ax.set_title('Lineament Directions\n(geological classification)')

    plt.tight_layout()
    out_fig = str(OUTPUT_DIR / 'lineament_detection_results.png')
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    print(f"  ✓ Figure → {out_fig}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dem_path: str | None = None):
    print("=" * 60)
    print("Bengaluru Lineament Detection")
    print("=" * 60)

    # Step 1: Load DEM
    if dem_path and Path(dem_path).exists() and HAS_RASTERIO:
        print(f"Loading DEM: {dem_path}")
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)
        print(f"  → Shape: {dem.shape}")
    else:
        print("Using synthetic Bengaluru DEM (512×512)...")
        dem = create_synthetic_bengaluru_dem(512, 512)

    # Step 2: Hillshade
    print("Computing multi-directional hillshade...")
    hillshade_multi = multi_directional_hillshade(dem)

    # Step 3: Edge detection
    print("Applying Canny edge detection...")
    edges = detect_edges_canny(hillshade_multi)
    print(f"  → Edge pixels: {edges.sum():,}")

    # Step 4: Hough line detection
    print("Running probabilistic Hough Transform...")
    lines = detect_lineaments_hough(edges)
    print(f"  → Lineament segments detected: {len(lines)}")

    # Step 5: Convert to GeoDataFrame
    print("Converting to geographic coordinates...")
    gdf = lines_to_geodataframe(lines, BENGALURU_BBOX, dem.shape)
    gdf = classify_lineament_direction(gdf)

    print(f"\nLineament Statistics:")
    print(f"  Total segments   : {len(gdf)}")
    print(f"  Total length (m) : {gdf['length_m'].sum():,.0f}")
    print(f"  Mean length (m)  : {gdf['length_m'].mean():.1f}")
    print(f"\nDirection distribution:")
    print(gdf['direction'].value_counts().to_string())

    # Step 6: Save GeoJSON
    geojson_out = str(OUTPUT_DIR / 'lineaments.geojson')
    gdf.to_file(geojson_out, driver='GeoJSON')
    print(f"\n  ✓ Lineaments GeoJSON → {geojson_out}")

    # Step 7: Density raster
    print("Computing lineament density raster...")
    density = compute_lineament_density(gdf, BENGALURU_BBOX, resolution=0.005)
    save_density_raster(density, BENGALURU_BBOX,
                        str(OUTPUT_DIR / 'lineament_density.tif'))

    # Step 8: Statistics CSV
    stats_out = str(OUTPUT_DIR / 'lineament_stats.csv')
    gdf[['lineament_id', 'azimuth_deg', 'direction', 'length_m']].to_csv(
        stats_out, index=False)
    print(f"  ✓ Stats CSV → {stats_out}")

    # Step 9: Visualisation
    print("Generating visualisation...")
    visualise_results(dem, hillshade_multi, edges, lines, gdf, BENGALURU_BBOX)

    print("\nDone ✓")
    return gdf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lineament detection for Bengaluru')
    parser.add_argument('--dem', type=str, default=None,
                        help='Path to NASADEM GeoTIFF (optional)')
    args = parser.parse_args()
    main(args.dem)
