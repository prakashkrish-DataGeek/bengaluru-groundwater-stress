// =============================================================================
// Script 03: Groundwater Potential Zone (GWPZ) Composite Model
//            Bengaluru, Karnataka
// =============================================================================
// Formula: GWPZ = (w1 × LULC_score) + (w2 × Slope_score) +
//                 (w3 × NDVI_score)  + (w4 × Lineament_score)
//
// Default weights (Analytical Hierarchy Process — peer-reviewed values):
//   w1 (LULC)        = 0.30
//   w2 (Slope)       = 0.25
//   w3 (NDVI)        = 0.25
//   w4 (Lineaments)  = 0.20
//
// Each input is reclassified to a 1–5 suitability score:
//   5 = Very High Groundwater Potential
//   1 = Very Low  Groundwater Potential
// =============================================================================

// ── 1. STUDY AREA ─────────────────────────────────────────────────────────────
var bengaluru = ee.Geometry.Rectangle([77.40, 12.77, 77.82, 13.18]);
Map.centerObject(bengaluru, 11);

// ── 2. WEIGHTS (easily adjustable) ────────────────────────────────────────────
var W1_LULC       = 0.30;
var W2_SLOPE      = 0.25;
var W3_NDVI       = 0.25;
var W4_LINEAMENTS = 0.20;

// Validate weights sum to 1.0
var weightSum = W1_LULC + W2_SLOPE + W3_NDVI + W4_LINEAMENTS;
print('Weight sum (should be 1.0):', weightSum);

// ── 3. LAYER 1 — LULC SUITABILITY SCORE ─────────────────────────────────────
// ESRI 10m LULC 2023
// Recharge suitability: Crops/Rangeland=5 (good recharge), Trees=4,
//                       Water=3 (existing), Bare=3, Urban=1 (impervious)
var esriLULC = ee.ImageCollection('projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS')
  .filterDate('2023-01-01', '2023-12-31')
  .mosaic()
  .clip(bengaluru);

// Reclassification: ESRI class → recharge suitability score
// Class 1=Water, 2=Trees, 4=Flooded Veg, 5=Crops, 7=Built, 8=Bare, 11=Rangeland
var lulcScore = esriLULC
  .where(esriLULC.eq(1),  3)   // Water bodies — moderate (not recharge zone)
  .where(esriLULC.eq(2),  4)   // Trees — good (root infiltration)
  .where(esriLULC.eq(4),  5)   // Flooded vegetation — excellent recharge
  .where(esriLULC.eq(5),  5)   // Crops — excellent (open soil, irrigation return)
  .where(esriLULC.eq(7),  1)   // Built-up — very poor (impervious)
  .where(esriLULC.eq(8),  3)   // Bare ground — moderate
  .where(esriLULC.eq(9),  1)   // Snow/ice — not applicable
  .where(esriLULC.eq(11), 5)   // Rangeland — excellent
  .rename('LULC_score')
  .clip(bengaluru);

// ── 4. LAYER 2 — SLOPE SUITABILITY SCORE ─────────────────────────────────────
// Flat terrain → high infiltration potential; steep → high runoff
// Score: 0–2° = 5, 2–5° = 4, 5–10° = 3, 10–20° = 2, >20° = 1
var dem   = ee.Image('NASA/NASADEM_HGT/001').select('elevation').clip(bengaluru);
var slope = ee.Terrain.slope(dem);

var slopeScore = ee.Image(1)
  .where(slope.lte(2),                     5)
  .where(slope.gt(2).and(slope.lte(5)),    4)
  .where(slope.gt(5).and(slope.lte(10)),   3)
  .where(slope.gt(10).and(slope.lte(20)),  2)
  .where(slope.gt(20),                     1)
  .rename('Slope_score')
  .clip(bengaluru);

// ── 5. LAYER 3 — NDVI SUITABILITY SCORE ─────────────────────────────────────
// Dense vegetation → better infiltration and soil moisture retention
// Score: NDVI < 0  = 1, 0–0.2 = 2, 0.2–0.4 = 3, 0.4–0.6 = 4, > 0.6 = 5
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(bengaluru)
  .filterDate('2023-01-01', '2024-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median()
  .clip(bengaluru);

var ndvi = s2.normalizedDifference(['B8', 'B4']);

var ndviScore = ee.Image(1)
  .where(ndvi.lt(0),                          1)
  .where(ndvi.gte(0).and(ndvi.lt(0.2)),       2)
  .where(ndvi.gte(0.2).and(ndvi.lt(0.4)),     3)
  .where(ndvi.gte(0.4).and(ndvi.lt(0.6)),     4)
  .where(ndvi.gte(0.6),                        5)
  .rename('NDVI_score')
  .clip(bengaluru);

// ── 6. LAYER 4 — LINEAMENT / FRACTURE DENSITY SCORE ─────────────────────────
// Lineaments/fractures act as conduits for groundwater recharge.
// Proxy: Terrain ruggedness + directional edge detection on DEM hillshade
// (True lineament mapping requires field survey or optical image classification)

// Hillshade as proxy for surface roughness lineaments
var hillshade = ee.Terrain.hillshade(dem, 315, 45).clip(bengaluru);

// Sobel edge detection (horizontal + vertical gradients)
var sobelH = hillshade.convolve(ee.Kernel.sobel());
var sobelV = hillshade.convolve(ee.Kernel.prewitt());
var edgeMag = sobelH.pow(2).add(sobelV.pow(2)).sqrt().rename('edge_magnitude');

// Normalize edge magnitude to 1–5 score
var edgeMin = edgeMag.reduceRegion({reducer: ee.Reducer.percentile([5]),  geometry: bengaluru, scale: 500, maxPixels: 1e9}).values().get(0);
var edgeMax = edgeMag.reduceRegion({reducer: ee.Reducer.percentile([95]), geometry: bengaluru, scale: 500, maxPixels: 1e9}).values().get(0);

var lineamentScore = edgeMag
  .subtract(ee.Image.constant(edgeMin))
  .divide(ee.Image.constant(edgeMax).subtract(ee.Image.constant(edgeMin)))
  .multiply(4).add(1)
  .clamp(1, 5)
  .rename('Lineament_score')
  .clip(bengaluru);

// ── 7. COMPOSITE GWPZ ────────────────────────────────────────────────────────
var gwpz = lulcScore.multiply(W1_LULC)
  .add(slopeScore.multiply(W2_SLOPE))
  .add(ndviScore.multiply(W3_NDVI))
  .add(lineamentScore.multiply(W4_LINEAMENTS))
  .rename('GWPZ');

// Classify GWPZ into 5 zones
var gwpzClass = ee.Image(1)
  .where(gwpz.lte(1.8),                         1)   // Very Low
  .where(gwpz.gt(1.8).and(gwpz.lte(2.6)),       2)   // Low
  .where(gwpz.gt(2.6).and(gwpz.lte(3.4)),       3)   // Moderate
  .where(gwpz.gt(3.4).and(gwpz.lte(4.2)),       4)   // High
  .where(gwpz.gt(4.2),                           5)   // Very High
  .rename('GWPZ_Class')
  .clip(bengaluru);

// ── 8. VISUALISATION ─────────────────────────────────────────────────────────
var gwpzPalette = ['#d73027','#fc8d59','#fee08b','#d9ef8b','#1a9850'];

Map.addLayer(lulcScore,      {min:1,max:5, palette: gwpzPalette}, 'LULC Score');
Map.addLayer(slopeScore,     {min:1,max:5, palette: gwpzPalette}, 'Slope Score');
Map.addLayer(ndviScore,      {min:1,max:5, palette: gwpzPalette}, 'NDVI Score');
Map.addLayer(lineamentScore, {min:1,max:5, palette: gwpzPalette}, 'Lineament Score');
Map.addLayer(gwpz,           {min:1,max:5, palette: gwpzPalette}, 'GWPZ Composite (continuous)', false);
Map.addLayer(gwpzClass,      {min:1,max:5, palette: gwpzPalette}, 'GWPZ Classified (5 zones)', true);

// ── 9. AREA STATISTICS PER GWPZ CLASS ────────────────────────────────────────
var pixelArea = ee.Image.pixelArea().divide(1e6);  // km²

var areaByClass = gwpzClass.multiply(pixelArea).reduceRegion({
  reducer: ee.Reducer.sum().group({groupField: 0, groupName: 'GWPZ_Class'}),
  geometry: bengaluru,
  scale: 100,
  maxPixels: 1e9
});
print('Area by GWPZ class (km²):', areaByClass);

// ── 10. HOTSPOT OVERLAY ───────────────────────────────────────────────────────
// High potential zones (Class 4 & 5) are groundwater recharge priority areas
var hotspots = gwpzClass.gte(4).selfMask().rename('GW_Hotspot');
Map.addLayer(hotspots, {palette: ['#006400']}, 'High GW Potential Hotspots');

// ── 11. EXPORT ────────────────────────────────────────────────────────────────
Export.image.toDrive({
  image: gwpzClass,
  description: 'Bengaluru_GWPZ_Classified',
  folder: 'GWS_Bengaluru',
  fileNamePrefix: 'gwpz_classified',
  region: bengaluru,
  scale: 30,
  crs: 'EPSG:32643',
  maxPixels: 1e10
});

Export.image.toDrive({
  image: gwpz,
  description: 'Bengaluru_GWPZ_Continuous',
  folder: 'GWS_Bengaluru',
  fileNamePrefix: 'gwpz_continuous',
  region: bengaluru,
  scale: 30,
  crs: 'EPSG:32643',
  maxPixels: 1e10
});

// ── 12. LEGEND (UI Panel) ─────────────────────────────────────────────────────
var legend = ui.Panel({style: {position:'bottom-left', padding:'8px 15px'}});
legend.add(ui.Label('GWPZ Classes', {fontWeight:'bold', fontSize:'14px'}));
var classes = ['Very Low (1)', 'Low (2)', 'Moderate (3)', 'High (4)', 'Very High (5)'];
var colors  = ['#d73027','#fc8d59','#fee08b','#d9ef8b','#1a9850'];
classes.forEach(function(cls, i) {
  var colorBox = ui.Label('', {backgroundColor: colors[i], padding:'8px', margin:'2px'});
  var row = ui.Panel([colorBox, ui.Label(cls)], ui.Panel.Layout.Flow('horizontal'));
  legend.add(row);
});
Map.add(legend);
