// =============================================================================
// Script 02: Multi-Source Covariate Extraction for Bengaluru
//            Sentinel-2 NDVI | ESRI 10m LULC | NASADEM Slope | Rainfall
// =============================================================================
// Outputs a sampled CSV of pixel-level covariates across Bengaluru at a
// regular grid — used as input for the GWPZ model (Script 03) and
// ML-based groundwater depth prediction.
// =============================================================================

// ── 1. STUDY AREA ─────────────────────────────────────────────────────────────
var bengaluru = ee.Geometry.Rectangle([77.40, 12.77, 77.82, 13.18]);
Map.centerObject(bengaluru, 11);

// ── 2. SENTINEL-2 SURFACE REFLECTANCE — NDVI ─────────────────────────────────
// Cloud-masked median composite, Jan 2023 – Dec 2024
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(bengaluru)
  .filterDate('2023-01-01', '2024-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

// Cloud masking using SCL band
function maskS2clouds(image) {
  var scl = image.select('SCL');
  // Keep: vegetation(4), bare soil(5), water(6), unclassified(7), snow(11)
  var mask = scl.eq(4).or(scl.eq(5)).or(scl.eq(6))
               .or(scl.eq(7)).or(scl.eq(11));
  return image.updateMask(mask);
}

var s2Masked = s2.map(maskS2clouds);
var s2Median = s2Masked.median().clip(bengaluru);

// NDVI = (NIR - Red) / (NIR + Red) using B8 and B4
var ndvi = s2Median.normalizedDifference(['B8', 'B4']).rename('NDVI');

// EVI for supplementary vegetation stress analysis
var evi = s2Median.expression(
  '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
    'NIR':  s2Median.select('B8'),
    'RED':  s2Median.select('B4'),
    'BLUE': s2Median.select('B2')
  }).rename('EVI');

// MNDWI (water bodies) = (Green - SWIR1)/(Green + SWIR1)
var mndwi = s2Median.normalizedDifference(['B3', 'B11']).rename('MNDWI');

// ── 3. ESRI 10m LAND USE / LAND COVER 2023 ────────────────────────────────────
// Classes: 1=Water, 2=Trees, 4=Flooded Veg, 5=Crops, 7=Built Area,
//          8=Bare Ground, 9=Snow/Ice, 10=Clouds, 11=Rangeland
var esriLULC = ee.ImageCollection('projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS')
  .filterDate('2023-01-01', '2023-12-31')
  .mosaic()
  .clip(bengaluru)
  .rename('LULC');

// ── 4. NASADEM — ELEVATION + DERIVED SLOPE ────────────────────────────────────
var dem   = ee.Image('NASA/NASADEM_HGT/001').select('elevation').clip(bengaluru);
var slope = ee.Terrain.slope(dem).rename('Slope');
var aspect = ee.Terrain.aspect(dem).rename('Aspect');

// Topographic Wetness Index (TWI) proxy = elevation deviation from local mean
var demSmooth = dem.focal_mean({radius: 500, units: 'meters'});
var twiProxy  = demSmooth.subtract(dem).rename('TWI_proxy');

// ── 5. RAINFALL PROXY — CHIRPS PRECIPITATION ─────────────────────────────────
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterBounds(bengaluru)
  .filterDate('2023-01-01', '2024-12-31');

var annualRainfall = chirps.sum().clip(bengaluru).rename('Annual_Rainfall_mm');
var dryMonthRain   = chirps.filterDate('2024-01-01', '2024-03-31').sum()
                           .clip(bengaluru).rename('DrySeasonRainfall_mm');

// ── 6. COMPOSITE ALL COVARIATES ───────────────────────────────────────────────
var covariates = ee.Image.cat([
  ndvi,
  evi,
  mndwi,
  esriLULC,
  slope,
  aspect,
  twiProxy,
  dem.rename('Elevation_m'),
  annualRainfall,
  dryMonthRain
]);

// ── 7. VISUALISATION ─────────────────────────────────────────────────────────
var ndviVis  = {min: -0.2, max: 0.8, palette: ['#d73027','#f46d43','#fdae61','#fee08b','#d9ef8b','#a6d96a','#66bd63','#1a9850']};
var slopeVis = {min: 0, max: 30, palette: ['#ffffff','#fee8c8','#fdbb84','#e34a33']};
var lulcVis  = {min: 1, max: 11, palette: ['#1a5276','#27ae60','#1abc9c','#f39c12','#e74c3c','#d35400','#ecf0f1','#bdc3c7','#85c1e9','#d7dbdd','#f9e79f']};
var rainVis  = {min: 500, max: 2000, palette: ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#99000d']};

Map.addLayer(ndvi,          ndviVis,  'Sentinel-2 NDVI (2023-24)');
Map.addLayer(esriLULC,      lulcVis,  'ESRI LULC 10m (2023)');
Map.addLayer(slope,         slopeVis, 'Terrain Slope (degrees)');
Map.addLayer(dem,           {min: 800, max: 1000, palette: ['#313695','#4575b4','#74add1','#abd9e9','#e0f3f8','#fee090','#fdae61','#f46d43','#d73027']}, 'NASADEM Elevation');
Map.addLayer(annualRainfall, rainVis, 'CHIRPS Annual Rainfall 2023-24');
Map.addLayer(mndwi,         {min: -0.5, max: 0.5, palette: ['#d73027','#ffffff','#1a9850']}, 'MNDWI (Water Bodies)');

// ── 8. SAMPLE COVARIATES ON A REGULAR GRID ───────────────────────────────────
// Create 500m grid points across Bengaluru for CSV export
var sampleGrid = covariates.sample({
  region:      bengaluru,
  scale:       500,
  projection:  'EPSG:4326',
  numPixels:   20000,
  geometries:  true,
  seed:        42
});

print('Sample count:', sampleGrid.size());
print('First feature:', sampleGrid.first());

// ── 9. EXPORT CSV ─────────────────────────────────────────────────────────────
Export.table.toDrive({
  collection:  sampleGrid,
  description: 'Bengaluru_Covariates_500m_Grid',
  folder:      'GWS_Bengaluru',
  fileFormat:  'CSV'
});

// ── 10. EXPORT COVARIATE RASTERS ─────────────────────────────────────────────
Export.image.toDrive({
  image:           covariates,
  description:     'Bengaluru_Covariates_Stack',
  folder:          'GWS_Bengaluru',
  fileNamePrefix:  'bengaluru_covariates',
  region:          bengaluru,
  scale:           30,
  crs:             'EPSG:32643',
  maxPixels:       1e10
});

// ── 11. BASIC STATISTICS ─────────────────────────────────────────────────────
var ndviStats = ndvi.reduceRegion({
  reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true)
                            .combine(ee.Reducer.percentile([10, 25, 75, 90]), '', true),
  geometry: bengaluru,
  scale: 100,
  maxPixels: 1e9
});
print('NDVI statistics:', ndviStats);

var lulcHist = esriLULC.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: bengaluru,
  scale: 100,
  maxPixels: 1e9
});
print('LULC class distribution:', lulcHist);
