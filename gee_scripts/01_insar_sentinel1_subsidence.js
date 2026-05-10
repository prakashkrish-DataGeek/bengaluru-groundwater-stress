// =============================================================================
// Script 01: Sentinel-1 SAR Backscatter Temporal Analysis
//            Land Subsidence Proxy — Bengaluru, Karnataka
// =============================================================================
// Objective : Detect areas of anomalous surface change using Sentinel-1 GRD
//             backscatter time-series over the last 24 months.
//             True InSAR (phase-based) processing is NOT available natively in
//             GEE. For fully processed InSAR displacement products use:
//               • Copernicus Ground Motion Service (EGMS):
//                 https://egms.land.copernicus.eu/
//               • NASA ARIA processed InSAR (NISAR-era products):
//                 https://aria.jpl.nasa.gov/
//               • TRE-ALTAMIRA / SkyGeo services
//             This script computes:
//               (a) Mean backscatter (σ°) composite
//               (b) Temporal standard deviation — high σ signals likely
//                   land-use change, subsidence, or surface disturbance
//               (c) Z-score anomaly map to identify "hotspot" pixels
// =============================================================================

// ── 1. STUDY AREA ─────────────────────────────────────────────────────────────
var bengaluru = ee.Geometry.Rectangle([77.40, 12.77, 77.82, 13.18]);
Map.centerObject(bengaluru, 11);
Map.addLayer(bengaluru, {color: '000000'}, 'Bengaluru Boundary', false);

// ── 2. DATE RANGE — 24 months ─────────────────────────────────────────────────
var endDate   = ee.Date(Date.now());
var startDate = endDate.advance(-24, 'month');
print('Analysis period:', startDate.format('YYYY-MM-dd'), '→', endDate.format('YYYY-MM-dd'));

// ── 3. LOAD SENTINEL-1 GRD COLLECTION ────────────────────────────────────────
// IW mode, VV polarisation, descending pass (most consistent for subsidence)
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(bengaluru)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select('VV');

print('Sentinel-1 image count:', s1.size());

// ── 4. TEMPORAL STATISTICS ────────────────────────────────────────────────────
var meanBackscatter = s1.mean().clip(bengaluru);
var stdBackscatter  = s1.reduce(ee.Reducer.stdDev()).clip(bengaluru);
var minBs           = s1.min().clip(bengaluru);
var maxBs           = s1.max().clip(bengaluru);

// ── 5. Z-SCORE ANOMALY MAP ────────────────────────────────────────────────────
// High positive z-score → sudden backscatter increase (possible new construction)
// High negative z-score → backscatter decrease (possible subsidence / water logging)
var zScore = meanBackscatter.subtract(minBs)
                            .divide(stdBackscatter.add(1e-6))
                            .rename('z_score');

// ── 6. SUBSIDENCE PROXY INDEX ─────────────────────────────────────────────────
// Combination of high temporal variability + low mean backscatter
// (bare/subsiding soil often shows low and variable returns)
var subsidenceProxy = stdBackscatter.divide(meanBackscatter.abs().add(1e-6))
                                    .rename('subsidence_proxy');

// ── 7. THRESHOLD HOTSPOTS ─────────────────────────────────────────────────────
// Areas where temporal std-dev > 3 dB considered potentially unstable
var hotspots = stdBackscatter.gt(3).selfMask().rename('hotspot');

// ── 8. VISUALISATION ─────────────────────────────────────────────────────────
var visBackscatter = {min: -25, max: 0, palette: ['000000','404040','808080','ffffff']};
var visStd         = {min: 0, max: 6, palette: ['0000ff','00ff00','ffff00','ff8000','ff0000']};
var visProxy       = {min: 0, max: 1, palette: ['#2b83ba','#abdda4','#ffffbf','#fdae61','#d7191c']};
var visHotspot     = {palette: ['ff0000']};

Map.addLayer(meanBackscatter,  visBackscatter, 'S1 Mean Backscatter (VV, dB)');
Map.addLayer(stdBackscatter,   visStd,         'S1 Temporal Std-Dev (Instability)');
Map.addLayer(subsidenceProxy,  visProxy,       'Subsidence Proxy Index');
Map.addLayer(hotspots,         visHotspot,     'Subsidence Hotspots (StdDev > 3 dB)', true);

// ── 9. MONTHLY TIME SERIES CHART ─────────────────────────────────────────────
// Pick a representative point in Central Bengaluru
var centralPoint = ee.Geometry.Point([77.5946, 12.9716]);

var chart = ui.Chart.image.series({
  imageCollection: s1,
  region: centralPoint,
  reducer: ee.Reducer.mean(),
  scale: 30
}).setOptions({
  title: 'Sentinel-1 VV Backscatter — Central Bengaluru (24 months)',
  hAxis: {title: 'Date'},
  vAxis: {title: 'Backscatter (dB)'},
  lineWidth: 2,
  pointSize: 4,
  colors: ['#1a73e8']
});
print(chart);

// ── 10. EXPORT PRODUCTS ──────────────────────────────────────────────────────
Export.image.toDrive({
  image: subsidenceProxy,
  description: 'Bengaluru_Subsidence_Proxy_S1_24months',
  folder: 'GWS_Bengaluru',
  fileNamePrefix: 'subsidence_proxy',
  region: bengaluru,
  scale: 20,
  crs: 'EPSG:32643',
  maxPixels: 1e10
});

Export.image.toDrive({
  image: hotspots,
  description: 'Bengaluru_Subsidence_Hotspots',
  folder: 'GWS_Bengaluru',
  fileNamePrefix: 'subsidence_hotspots',
  region: bengaluru,
  scale: 20,
  crs: 'EPSG:32643',
  maxPixels: 1e10
});

// ── 11. STATISTICS TABLE ─────────────────────────────────────────────────────
var stats = subsidenceProxy.reduceRegion({
  reducer: ee.Reducer.percentile([25, 50, 75, 95]),
  geometry: bengaluru,
  scale: 100,
  maxPixels: 1e9
});
print('Subsidence Proxy Percentiles:', stats);

// ── NOTE ON TRUE InSAR ────────────────────────────────────────────────────────
// For millimetre-precision vertical displacement (InSAR LOS), use:
// 1. Download Sentinel-1 SLC images from Copernicus Open Access Hub
// 2. Process with ESA SNAP + StaMPS / MintPy for time-series InSAR
// 3. OR access Copernicus EGMS (European Ground Motion Service) — covers India
//    via regional service; check https://egms.land.copernicus.eu/
// This backscatter approach is a FREE, fast proxy suitable for hotspot screening.
