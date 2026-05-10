"""
05_interactive_heatmap.py
=========================
Build a publishable interactive multi-layer heatmap of Bengaluru's
groundwater stress model using Folium + Leaflet.js.

Layers:
  1. GWPZ (Groundwater Potential Zones) — choropleth heatmap
  2. Storm Water Drain Network
  3. Subsidence Hotspots (Sentinel-1 proxy)
  4. LST Urban Heat (Landsat-9 proxy)
  5. WRIS Groundwater Station markers
  6. Lineament traces

Output:
  web/bengaluru_groundwater_heatmap.html   (self-contained, publishable)

Usage:
    pip install folium branca numpy geopandas pandas
    python python/05_interactive_heatmap.py
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
from folium.plugins import HeatMap, MarkerCluster, MeasureControl, MiniMap
from branca.colormap import LinearColormap
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR      = 'data'
WEB_DIR       = 'web'
os.makedirs(WEB_DIR, exist_ok=True)

OUT_HTML      = os.path.join(WEB_DIR, 'bengaluru_groundwater_heatmap.html')

BENGALURU_CENTER = [12.9716, 77.5946]
BENGALURU_BBOX   = [[12.77, 77.40], [13.18, 77.82]]
WGS84            = 'EPSG:4326'
BENGALURU_CRS    = 'EPSG:32643'


# ── Data generators (use real data when available) ───────────────────────────

def load_or_generate_gwpz_grid() -> list:
    """
    Load GWPZ grid from Script 03 output or generate synthetic scores.
    Returns list of [lat, lon, weight] for HeatMap.
    """
    gwpz_csv = os.path.join(DATA_DIR, 'drain_proximity_grid.csv')
    if os.path.exists(gwpz_csv):
        df = pd.read_csv(gwpz_csv)
        # Use drain_influence_score as proxy for GWPZ
        df['weight'] = df.get('drain_influence_score', 1) / 4.0
        return df[['latitude', 'longitude', 'weight']].values.tolist()
    else:
        # Synthetic GWPZ — higher potential in peripheral/green areas
        rng = np.random.default_rng(42)
        n = 2000
        lons = rng.uniform(77.42, 77.80, n)
        lats = rng.uniform(12.79, 13.16, n)
        # Central urban core = low potential; periphery = high
        center_lon, center_lat = 77.5946, 12.9716
        dist_from_center = np.sqrt((lons - center_lon)**2 + (lats - center_lat)**2)
        weights = np.clip(dist_from_center * 10 + rng.normal(0, 0.1, n), 0.1, 1.0)
        return [[lat, lon, w] for lat, lon, w in zip(lats, lons, weights)]


def load_or_generate_subsidence() -> list:
    """Synthetic subsidence hotspot points (replace with S1 export)."""
    rng = np.random.default_rng(7)
    # Known subsidence-prone areas: Koramangala, Whitefield, Bommanahalli
    hotspot_centers = [
        (12.9279, 77.6271, 'Koramangala'),   # IT hub, rampant construction
        (12.9698, 77.7499, 'Whitefield'),    # Heavy groundwater extraction
        (12.8994, 77.6184, 'Bommanahalli'),  # Industrial, drainage issues
        (13.0298, 77.5503, 'Yelahanka'),     # Rapid urbanisation
        (12.8399, 77.6770, 'Electronic City'), # IT cluster
    ]
    points = []
    for lat_c, lon_c, _ in hotspot_centers:
        n = 80
        lats = rng.normal(lat_c, 0.02, n)
        lons = rng.normal(lon_c, 0.02, n)
        weights = rng.uniform(0.4, 1.0, n)
        points.extend([[lat, lon, w] for lat, lon, w in zip(lats, lons, weights)])
    return points


def load_or_generate_lst() -> list:
    """Synthetic LST hotspots (replace with L9 export)."""
    rng = np.random.default_rng(11)
    # Urban core has highest LST
    urban_centers = [
        (12.9716, 77.5946),   # Central Bengaluru (MG Road)
        (12.9352, 77.6245),   # HSR Layout
        (12.9784, 77.6408),   # Indiranagar
        (13.0298, 77.5503),   # Yelahanka
        (12.9099, 77.5963),   # Jayanagar
    ]
    points = []
    for lat_c, lon_c in urban_centers:
        n = 100
        lats = rng.normal(lat_c, 0.025, n)
        lons = rng.normal(lon_c, 0.025, n)
        weights = rng.uniform(0.5, 1.0, n)
        points.extend([[lat, lon, w] for lat, lon, w in zip(lats, lons, weights)])
    return points


def load_or_generate_stations() -> list:
    """Load WRIS stations (synthetic with realistic Bengaluru attributes)."""
    csv_path = os.path.join(DATA_DIR, 'final_training_data.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if {'latitude', 'longitude', 'gw_depth_mbgl'}.issubset(df.columns):
            return df[['latitude', 'longitude', 'gw_depth_mbgl',
                        'station_id', 'aquifer_type']].to_dict('records')

    rng = np.random.default_rng(99)
    stations = []
    station_names = [
        'Hebbal', 'Indiranagar', 'Koramangala', 'Jayanagar', 'Rajajinagar',
        'Malleswaram', 'Whitefield', 'Electronic City', 'Yelahanka', 'Marathahalli',
        'KR Puram', 'Bannerghatta', 'Kengeri', 'Tumkur Road', 'Hosur Road',
        'Sarjapur', 'Domlur', 'Shivajinagar', 'Peenya', 'Yeshwantpur'
    ]
    for name in station_names:
        lat = rng.uniform(12.85, 13.10)
        lon = rng.uniform(77.48, 77.75)
        depth = rng.uniform(8, 55)
        aquifer = rng.choice(['Fractured', 'Alluvial', 'Weathered'])
        stations.append({
            'latitude': round(lat, 4),
            'longitude': round(lon, 4),
            'gw_depth_mbgl': round(depth, 1),
            'station_id': name,
            'aquifer_type': aquifer
        })
    return stations


def load_drain_geojson():
    """Load drain GeoJSON for rendering."""
    path = os.path.join(DATA_DIR, 'stormwater_drains.geojson')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_lineament_geojson():
    """Load lineament GeoJSON."""
    path = os.path.join(DATA_DIR, 'lineaments.geojson')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Map Builder ───────────────────────────────────────────────────────────────

def build_interactive_map() -> folium.Map:
    """Construct the full multi-layer Folium map."""

    # ── Base map ────────────────────────────────────────────────────────────
    m = folium.Map(
        location=BENGALURU_CENTER,
        zoom_start=11,
        tiles=None,
        control_scale=True
    )

    # Base tile layers
    folium.TileLayer('CartoDB positron',    name='CartoDB Light',   show=True).add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark',    show=False).add_to(m)
    folium.TileLayer('OpenStreetMap',       name='OpenStreetMap',   show=False).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Esri Satellite', show=False
    ).add_to(m)

    # ── Layer 1: GWPZ Heatmap ────────────────────────────────────────────────
    gwpz_data = load_or_generate_gwpz_grid()
    HeatMap(
        gwpz_data,
        name='🔵 Groundwater Potential Zone (GWPZ)',
        min_opacity=0.3,
        max_zoom=16,
        radius=20,
        blur=25,
        gradient={0.0: '#2c7bb6', 0.3: '#abd9e9', 0.5: '#ffffbf',
                  0.7: '#fdae61', 1.0: '#d7191c'},
        show=True
    ).add_to(m)

    # ── Layer 2: Subsidence Hotspots ─────────────────────────────────────────
    subs_data = load_or_generate_subsidence()
    subsidence_group = folium.FeatureGroup(
        name='🔴 Land Subsidence Hotspots (S1 Proxy)', show=False)
    HeatMap(
        subs_data,
        min_opacity=0.4,
        radius=18,
        blur=20,
        gradient={0.0: '#fee8c8', 0.5: '#fc8d59', 1.0: '#d73027'}
    ).add_to(subsidence_group)
    subsidence_group.add_to(m)

    # ── Layer 3: LST Urban Heat ──────────────────────────────────────────────
    lst_data = load_or_generate_lst()
    lst_group = folium.FeatureGroup(
        name='🟠 Urban Heat (Landsat-9 LST)', show=False)
    HeatMap(
        lst_data,
        min_opacity=0.4,
        radius=22,
        blur=30,
        gradient={0.0: '#ffffb2', 0.4: '#fecc5c', 0.7: '#fd8d3c', 1.0: '#e31a1c'}
    ).add_to(lst_group)
    lst_group.add_to(m)

    # ── Layer 4: Storm Water Drains ──────────────────────────────────────────
    drain_gj = load_drain_geojson()
    if drain_gj:
        folium.GeoJson(
            drain_gj,
            name='💧 Storm Water Drain Network',
            style_function=lambda f: {
                'color': '#1a73e8', 'weight': 1.5, 'opacity': 0.8
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['drain_length_km'] if 'drain_length_km' in
                        str(drain_gj.get('features', [{}])[0].get('properties', {})) else [],
                aliases=['Length (km):']
            ),
            show=True
        ).add_to(m)
    else:
        # Render synthetic drain lines
        drain_group = folium.FeatureGroup(name='💧 Storm Water Drains (Sample)', show=True)
        sample_lines = [
            [(12.84, 77.53), (12.90, 77.59), (13.00, 77.65)],
            [(12.87, 77.45), (12.91, 77.52), (12.93, 77.58)],
            [(12.82, 77.60), (12.88, 77.63), (12.95, 77.63)],
        ]
        for line in sample_lines:
            folium.PolyLine(line, color='#1a73e8', weight=2, opacity=0.7).add_to(drain_group)
        drain_group.add_to(m)

    # ── Layer 5: Lineaments ──────────────────────────────────────────────────
    lin_gj = load_lineament_geojson()
    if lin_gj:
        folium.GeoJson(
            lin_gj,
            name='🟣 Geological Lineaments / Fractures',
            style_function=lambda f: {
                'color': '#7b2d8b', 'weight': 1.2, 'opacity': 0.6,
                'dashArray': '5, 5'
            },
            show=False
        ).add_to(m)

    # ── Layer 6: WRIS Groundwater Stations ───────────────────────────────────
    stations = load_or_generate_stations()
    station_cluster = MarkerCluster(name='📍 WRIS Groundwater Stations', show=True)

    depth_cmap = LinearColormap(
        ['#1a9850', '#ffffbf', '#d73027'],
        vmin=5, vmax=60,
        caption='Groundwater Depth (mbgl)'
    )
    depth_cmap.add_to(m)

    for s in stations:
        depth = s.get('gw_depth_mbgl', 25)
        color = depth_cmap(depth)
        folium.CircleMarker(
            location=[s['latitude'], s['longitude']],
            radius=8,
            color='black',
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=folium.Tooltip(
                f"<b>{s.get('station_id','Station')}</b><br>"
                f"Depth: {depth:.1f} mbgl<br>"
                f"Aquifer: {s.get('aquifer_type','–')}"
            ),
            popup=folium.Popup(
                f"""<div style='font-family:sans-serif;width:200px'>
                <b>{s.get('station_id','Station')}</b><br>
                <hr style='margin:4px'>
                Depth: <b>{depth:.1f} mbgl</b><br>
                Aquifer: {s.get('aquifer_type','–')}<br>
                Lat: {s['latitude']:.4f}, Lon: {s['longitude']:.4f}
                </div>""",
                max_width=220
            )
        ).add_to(station_cluster)

    station_cluster.add_to(m)

    # ── Layer 7: Bengaluru boundary ──────────────────────────────────────────
    folium.Rectangle(
        bounds=BENGALURU_BBOX,
        color='#333333',
        weight=2,
        fill=False,
        dash_array='10',
        tooltip='Bengaluru Study Area',
        name='📦 Study Area Boundary'
    ).add_to(m)

    # ── Plugins ──────────────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True, tile_layer='CartoDB positron').add_to(m)
    MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m)
    plugins.Fullscreen(position='topright').add_to(m)
    plugins.LocateControl(position='topright').add_to(m)

    # ── Title Panel ──────────────────────────────────────────────────────────
    title_html = """
    <div style="position:fixed;top:10px;left:60px;z-index:1000;
                background:rgba(255,255,255,0.92);padding:12px 18px;
                border-radius:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.3);
                font-family:'Segoe UI',sans-serif;max-width:320px">
      <div style="font-size:14px;font-weight:700;color:#1a1a2e;margin-bottom:4px">
        🌊 Bengaluru Groundwater Stress Model
      </div>
      <div style="font-size:11px;color:#555">
        Sentinel-1 InSAR Proxy · Sentinel-2 NDVI<br>
        Landsat-9 LST · ESRI LULC · NASADEM<br>
        Storm Water Drains · WRIS Stations
      </div>
      <div style="font-size:10px;color:#888;margin-top:6px">
        by Prakash Krishnamachari | 2024
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    return m


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Building Bengaluru Interactive Groundwater Heatmap")
    print("=" * 60)

    m = build_interactive_map()
    m.save(OUT_HTML)

    size_kb = os.path.getsize(OUT_HTML) / 1024
    print(f"\n✓ Map saved → {OUT_HTML} ({size_kb:.0f} KB)")
    print(f"\nTo publish:")
    print(f"  1. Copy {OUT_HTML} to your GitHub Pages branch")
    print(f"  2. Or embed in web/index.html via <iframe>")
    print(f"  3. Or host directly on GitHub Pages from /web folder")
