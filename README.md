# 🌊 Predictive Groundwater Stress Modeling — Bengaluru

> Multi-sensor satellite analysis combining Sentinel-1 SAR, Sentinel-2 NDVI,
> Landsat-9 LST and NASADEM to map recharge zones, subsidence hotspots and
> urban water stress across Bengaluru, Karnataka.

[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?style=flat&logo=google&logoColor=white)](https://earthengine.google.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Map](https://img.shields.io/badge/Live%20Map-View%20Online-orange)](https://prakashkrish-dataGeek.github.io/bengaluru-groundwater-stress/)

---

## 🗺 Live Project Page

**[→ View Interactive Map & Analysis](https://prakashkrish-DataGeek.github.io/bengaluru-groundwater-stress/)**

---

## Overview

Bengaluru's groundwater table has dropped by **15–25 metres** in many areas over the
past decade due to rapid urbanisation, inadequate recharge and climate variability.
This project builds an **open-source, fully reproducible** groundwater stress model
using free satellite data and Google Earth Engine.

### Key Outputs

| Output | Description |
|--------|-------------|
| GWPZ Map | 5-class Groundwater Potential Zone composite (30m) |
| Subsidence Proxy | Sentinel-1 backscatter instability map (20m) |
| LST / UHI Map | Land Surface Temperature + Urban Heat Island intensity (30m) |
| Drain Proximity | 500m grid with distance to nearest primary drain |
| Lineament Map | Probabilistic Hough fracture/lineament detection |

---

## GWPZ Formula

```
GWPZ = (0.30 × LULC_score) + (0.25 × Slope_score)
      + (0.25 × NDVI_score) + (0.20 × Lineament_score)
```

Each input reclassified to 1–5 (Very Low → Very High groundwater potential).
Weights derived from Analytical Hierarchy Process (AHP).

---

## Repository Structure

```
bengaluru-groundwater-stress/
├── gee_scripts/
│   ├── 01_insar_sentinel1_subsidence.js    ← S1 backscatter temporal analysis
│   ├── 02_covariate_extraction.js           ← NDVI, LULC, Slope, Rainfall
│   ├── 03_gwpz_composite.js                 ← GWPZ weighted composite model
│   └── 04_landsat9_lst.js                   ← Landsat-9 LST + UHI
├── python/
│   ├── 01_kml_to_geojson.py                 ← Storm drain KML converter
│   ├── 02_lineament_detection.py            ← Canny + Hough fracture detection
│   ├── 03_covariate_merge.py                ← WRIS stations + GEE spatial join
│   ├── 04_drain_proximity.py                ← Drain distance grid analysis
│   └── 05_interactive_heatmap.py            ← Folium multi-layer map generator
├── web/
│   └── index.html                           ← GitHub Pages project showcase
├── data/
│   └── stormwater_drains.kml                ← BBMP Primary SWD network
├── docs/
│   └── linkedin_article.md                  ← LinkedIn publication article
├── requirements.txt
└── README.md
```

---

## Quick Start

### GEE Scripts (run in [code.earthengine.google.com](https://code.earthengine.google.com))

1. Open any `.js` file from `gee_scripts/`
2. Copy-paste into a new GEE Script
3. Click **Run** — results appear in the map + console
4. Use **Tasks** tab to export rasters to Google Drive

### Python Scripts

```bash
# 1. Clone repo
git clone https://github.com/prakashkrish-DataGeek/bengaluru-groundwater-stress.git
cd bengaluru-groundwater-stress

# 2. Install dependencies
pip install -r requirements.txt

# 3. Convert storm drain KML to GeoJSON
python python/01_kml_to_geojson.py

# 4. Run lineament detection
python python/02_lineament_detection.py

# 5. Merge covariates with WRIS stations
python python/03_covariate_merge.py

# 6. Drain proximity analysis
python python/04_drain_proximity.py

# 7. Build interactive heatmap
python python/05_interactive_heatmap.py
# → Opens: web/bengaluru_groundwater_heatmap.html
```

---

## Data Sources

| Dataset | Source | Resolution | GEE Collection ID |
|---------|--------|------------|-------------------|
| Sentinel-1 GRD | ESA Copernicus | 20m | `COPERNICUS/S1_GRD` |
| Sentinel-2 SR | ESA Copernicus | 10m | `COPERNICUS/S2_SR_HARMONIZED` |
| ESRI LULC 2023 | ESRI / Impact Observatory | 10m | `projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS` |
| NASADEM | NASA JPL | 30m | `NASA/NASADEM_HGT/001` |
| Landsat-9 C2 L2 | USGS / NASA | 30m | `LANDSAT/LC09/C02/T1_L2` |
| CHIRPS Daily | UCSB CHG | ~5.5km | `UCSB-CHG/CHIRPS/DAILY` |
| SWD Network | BBMP / Karnataka | Vector | KML (this repo) |
| WRIS Stations | India-WRIS / CWC | Point | [indiawris.gov.in](https://indiawris.gov.in/wris/#/groundWater) |

---

## True InSAR Note

GEE does not support phase-based InSAR processing. For millimetre-precision
vertical displacement, use:
- **Copernicus EGMS** (European Ground Motion Service): https://egms.land.copernicus.eu/
- **NASA ARIA**: https://aria.jpl.nasa.gov/
- **SNAP + StaMPS/MintPy** with Sentinel-1 SLC data

This project uses SAR backscatter temporal variance as a **free, fast proxy**
suitable for hotspot screening.

---

## Results Summary

- **Lowest GWPZ**: Central Bengaluru urban core (Classes 1–2)
- **Highest GWPZ**: Bannerghatta, Kengeri periphery, Yelahanka (Classes 4–5)
- **Subsidence hotspots**: Whitefield, Bommanahalli, KR Puram corridors
- **Peak LST**: Whitefield, Electronic City (~44–46°C summer, +10°C above rural)
- **Lineament trend**: NNW-SSE (Closepet Granite) — key artificial recharge targets

---

## Author

**Prakash Krishnamachari**
- GitHub: [@prakashkrish-DataGeek](https://github.com/prakashkrish-DataGeek)
- LinkedIn: [linkedin.com/in/prakashkrishnamachari](https://linkedin.com/in/prakashkrishnamachari)
- Email: prakash.krishnamachari@gmail.com

---

## License

MIT License — see [LICENSE](LICENSE) for details.
Satellite imagery data is subject to respective agency terms of use.

---

*Built with ❤️ for open water data science*
