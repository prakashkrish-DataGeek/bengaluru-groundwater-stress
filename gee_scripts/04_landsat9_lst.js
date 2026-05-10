// =============================================================================
// Script 04: Land Surface Temperature (LST) from Landsat-9
//            Urban Heat Island Proxy — Bengaluru
// =============================================================================
// Uses Landsat 9 Collection 2 Level-2 (ST_B10 thermal band).
// Scale factors: Multiply × 0.00341802 + 149.0 → Kelvin → subtract 273.15 → °C
// =============================================================================

// ── 1. STUDY AREA ─────────────────────────────────────────────────────────────
var bengaluru = ee.Geometry.Rectangle([77.40, 12.77, 77.82, 13.18]);
Map.centerObject(bengaluru, 11);

// ── 2. LOAD LANDSAT-9 COLLECTION 2 LEVEL-2 ───────────────────────────────────
var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(bengaluru)
  .filterDate('2022-11-01', '2024-12-31');

print('Landsat-9 scene count:', l9.size());

// ── 3. CLOUD MASKING ─────────────────────────────────────────────────────────
function maskL9clouds(image) {
  var qaMask  = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
  var satMask = image.select('QA_RADSAT').eq(0);
  return image.updateMask(qaMask).updateMask(satMask);
}

var l9Masked = l9.map(maskL9clouds);

// ── 4. APPLY LANDSAT SCALE FACTORS ───────────────────────────────────────────
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B10').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
}

var l9Scaled = l9Masked.map(applyScaleFactors);

// ── 5. COMPUTE LST IN CELSIUS ─────────────────────────────────────────────────
function computeLST(image) {
  var lst = image.select('ST_B10').subtract(273.15).rename('LST_C');
  return image.addBands(lst);
}

var l9WithLST = l9Scaled.map(computeLST);

// ── 6. SEASONAL COMPOSITES ────────────────────────────────────────────────────
// Summer (hottest: March–May)
var summerLST = l9WithLST
  .filter(ee.Filter.calendarRange(3, 5, 'month'))
  .select('LST_C')
  .mean()
  .clip(bengaluru)
  .rename('LST_Summer');

// Monsoon (June–September)
var monsoonLST = l9WithLST
  .filter(ee.Filter.calendarRange(6, 9, 'month'))
  .select('LST_C')
  .mean()
  .clip(bengaluru)
  .rename('LST_Monsoon');

// Winter/Post-monsoon (Oct–Feb)
var winterLST = l9WithLST
  .filter(ee.Filter.calendarRange(10, 12, 'month'))
  .select('LST_C')
  .mean()
  .clip(bengaluru)
  .rename('LST_Winter');

// Annual mean
var annualLST = l9WithLST.select('LST_C').mean().clip(bengaluru).rename('LST_Annual');

// ── 7. URBAN HEAT ISLAND INTENSITY ───────────────────────────────────────────
// UHI = LST_urban - LST_rural (where rural is ESRI LULC = Trees/Crops)
var esriLULC = ee.ImageCollection('projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS')
  .filterDate('2023-01-01', '2023-12-31')
  .mosaic()
  .clip(bengaluru);

// Rural reference: Trees (2) and Crops (5)
var ruralMask = esriLULC.eq(2).or(esriLULC.eq(5));
var urbanMask = esriLULC.eq(7);  // Built-up

var lstRural = summerLST.updateMask(ruralMask).reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: bengaluru,
  scale: 100,
  maxPixels: 1e9
}).values().get(0);

var uhiIntensity = summerLST.subtract(ee.Image.constant(lstRural))
                            .rename('UHI_Intensity');

print('Rural reference LST (mean, °C):', lstRural);

// ── 8. LST HOTSPOT CLASSIFICATION ────────────────────────────────────────────
// Classify based on percentile thresholds
var pct = summerLST.reduceRegion({
  reducer: ee.Reducer.percentile([20, 40, 60, 80]),
  geometry: bengaluru,
  scale: 100,
  maxPixels: 1e9
});
print('Summer LST percentiles:', pct);

var lstClass = ee.Image(1)
  .where(summerLST.gt(ee.Image.constant(pct.get('LST_Summer_p20'))), 2)
  .where(summerLST.gt(ee.Image.constant(pct.get('LST_Summer_p40'))), 3)
  .where(summerLST.gt(ee.Image.constant(pct.get('LST_Summer_p60'))), 4)
  .where(summerLST.gt(ee.Image.constant(pct.get('LST_Summer_p80'))), 5)
  .clip(bengaluru)
  .rename('LST_Class');

// ── 9. VISUALISATION ─────────────────────────────────────────────────────────
var lstPalette = ['#313695','#4575b4','#74add1','#abd9e9','#e0f3f8','#ffffbf','#fee090','#fdae61','#f46d43','#d73027','#a50026'];

Map.addLayer(summerLST,  {min: 22, max: 48, palette: lstPalette}, 'Summer LST (Mar-May) °C');
Map.addLayer(monsoonLST, {min: 18, max: 38, palette: lstPalette}, 'Monsoon LST (Jun-Sep) °C', false);
Map.addLayer(winterLST,  {min: 16, max: 35, palette: lstPalette}, 'Winter LST (Oct-Feb) °C', false);
Map.addLayer(annualLST,  {min: 20, max: 42, palette: lstPalette}, 'Annual Mean LST °C', false);
Map.addLayer(uhiIntensity, {min: -5, max: 10, palette: ['#2166ac','#f7f7f7','#d6604d','#b2182b']}, 'UHI Intensity (°C above rural)');
Map.addLayer(lstClass,   {min:1,max:5, palette:['#2c7bb6','#abd9e9','#ffffbf','#fdae61','#d7191c']}, 'LST Heat Classes');

// ── 10. SPATIAL CORRELATION WITH GWPZ ────────────────────────────────────────
// Load GWPZ from Script 03 export or recompute here
// High LST + Low GWPZ = Critical urban stress zone
// This section creates a stress overlay index
// (uncomment and update path after running Script 03 export)
/*
var gwpzExport = ee.Image('users/YOUR_GEE_USERNAME/gwpz_classified');
var stressIndex = lstClass.subtract(gwpzExport).rename('Stress_Index');
Map.addLayer(stressIndex, {min:-4,max:4, palette:['#1a9850','#ffffbf','#d73027']}, 'Urban Water Stress Index');
*/

// ── 11. TIME SERIES CHART FOR BENGALURU CENTRAL ──────────────────────────────
var centralPoint = ee.Geometry.Point([77.5946, 12.9716]);
var lstChart = ui.Chart.image.series({
  imageCollection: l9WithLST.select('LST_C'),
  region: centralPoint,
  reducer: ee.Reducer.mean(),
  scale: 100
}).setOptions({
  title: 'Landsat-9 Land Surface Temperature — Central Bengaluru',
  hAxis: {title: 'Date'},
  vAxis: {title: 'LST (°C)'},
  lineWidth: 2,
  pointSize: 4,
  colors: ['#d73027']
});
print(lstChart);

// ── 12. EXPORT ────────────────────────────────────────────────────────────────
Export.image.toDrive({
  image: summerLST,
  description: 'Bengaluru_LST_Summer_L9',
  folder: 'GWS_Bengaluru',
  fileNamePrefix: 'lst_summer',
  region: bengaluru,
  scale: 30,
  crs: 'EPSG:32643',
  maxPixels: 1e10
});

Export.image.toDrive({
  image: uhiIntensity,
  description: 'Bengaluru_UHI_Intensity',
  folder: 'GWS_Bengaluru',
  fileNamePrefix: 'uhi_intensity',
  region: bengaluru,
  scale: 30,
  crs: 'EPSG:32643',
  maxPixels: 1e10
});
