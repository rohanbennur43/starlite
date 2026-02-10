# MVT Styling Guide

This guide explains how to customize the styling of Point, Polygon, and Line datasets in the MVT viewer. The styling is defined in `view_mvt.html` using MapLibre GL layer configuration.

## Overview

Styling is controlled through `paint` properties in layer definitions. Each geometry type has specific properties you can customize.

## Styling Polygons

Polygons are used for datasets like `TIGER2018_COUNTY`, `TIGER2018_PLACE`, etc.

### Basic Polygon Layer Example

```javascript
{
  id: "place-fill",
  type: "fill",
  source: "local",
  "source-layer": "layer0",
  filter: ["==", "$type", "Polygon"],
  paint: {
    "fill-color": "#cccccc",        // Solid color
    "fill-opacity": 0.65             // Transparency (0-1)
  }
}
```

### Customizing Polygon Colors

#### Solid Color
```javascript
"fill-color": "#ff0000"  // Red
```

#### Data-Driven Color (by property)
Color polygons based on a feature property (e.g., STATEFP for states):

```javascript
"fill-color": [
  "match",
  ["get", "STATEFP"],      // Get the STATEFP property
  "01", "#a6cee3",         // STATEFP = "01" → Light blue (Alabama)
  "06", "#fb9a99",         // STATEFP = "06" → Light red (California)
  "48", "#ffed6f",         // STATEFP = "48" → Yellow (Texas)
  "#cccccc"                // Default color for all others
]
```

#### Interpolated Color (by numeric value)
```javascript
"fill-color": [
  "interpolate",
  ["linear"],
  ["to-number", ["get", "POPULATION"]],  // Convert population to number
  0,        "#ffffcc",    // Min value → Light yellow
  1000000,  "#ff7f00",    // Mid value → Orange
  10000000, "#ff0000"     // Max value → Red
]
```

### Polygon Outline

Add a line layer to outline polygons:

```javascript
{
  id: "place-outline",
  type: "line",
  source: "local",
  "source-layer": "layer0",
  filter: ["==", "$type", "Polygon"],
  paint: {
    "line-color": "#003366",      // Dark blue
    "line-width": 1.3,             // Line thickness
    "line-opacity": 0.9            // Transparency
  }
}
```

## Styling Points

Points are used for datasets like `TIGER2018_POINTLM`, landmarks, etc.

### Basic Point Layer Example

```javascript
{
  id: "pointlm-points",
  type: "circle",
  source: "local",
  "source-layer": "layer0",
  filter: ["==", "$type", "Point"],
  paint: {
    "circle-radius": 5,            // Circle size in pixels
    "circle-color": "#ff0000",     // Red
    "circle-opacity": 0.85,        // Transparency
    "circle-stroke-width": 1,      // Border thickness
    "circle-stroke-color": "#ffffff"  // Border color
  }
}
```

### Customizing Point Colors

#### Data-Driven Color (by category)
Color points based on a categorical property (e.g., MTFCC for landmark type):

```javascript
"circle-color": [
  "match",
  ["get", "MTFCC"],         // Get landmark type
  "K2540", "#ff7f0e",        // Schools → Orange
  "K2543", "#1f77b4",        // Hospitals → Blue
  "K2450", "#2ca02c",        // Airports → Green
  "K1231", "#d62728",        // Summits → Red
  "K3340", "#9467bd",        // Parks → Purple
  "#aaaaaa"                  // Default → Gray
]
```

### Zoom-Responsive Point Sizes

Make point size change with zoom level:

```javascript
"circle-radius": [
  "interpolate",
  ["linear"],
  ["zoom"],         // Based on current zoom level
  4,  2,            // At zoom 4, radius = 2 pixels
  8,  4,            // At zoom 8, radius = 4 pixels
  12, 7,            // At zoom 12, radius = 7 pixels
  14, 10            // At zoom 14, radius = 10 pixels
]
```

### Adding Popups to Points

Show information when hovering over points:

```javascript
const popup = new maplibregl.Popup({
  closeButton: false,
  closeOnClick: false
});

// Show popup on hover
map.on("mousemove", "pointlm-points", (e) => {
  const feature = e.features[0];
  const name = feature.properties.FULLNAME || "Unknown";
  
  map.getCanvas().style.cursor = "pointer";
  
  popup
    .setLngLat(e.lngLat)
    .setHTML(`<b>${name}</b><br>MTFCC: ${feature.properties.MTFCC}`)
    .addTo(map);
});

// Hide popup when mouse leaves
map.on("mouseleave", "pointlm-points", () => {
  map.getCanvas().style.cursor = "";
  popup.remove();
});
```

## Styling Lines

Lines are used for datasets like `TIGER2018_ROADS`, `TIGER2018_RAILS`, etc.

### Basic Line Layer Example

```javascript
{
  id: "roads-lines",
  type: "line",
  source: "local",
  "source-layer": "layer0",
  filter: ["==", "$type", "LineString"],
  paint: {
    "line-color": "#ff6600",       // Orange
    "line-width": 2.0,             // Thickness in pixels
    "line-opacity": 0.9            // Transparency
  }
}
```

### Customizing Line Colors

#### Data-Driven Color (by road type)
```javascript
"line-color": [
  "match",
  ["get", "RTTYP"],         // Get road type
  "M", "#ff0000",            // Major → Red
  "S", "#ffa500",            // Secondary → Orange
  "C", "#cccccc",            // Local → Light gray
  "P", "#ffff00",            // Ramp/Connector → Yellow
  "T", "#8b4513",            // Trail → Brown
  "#0066cc"                  // Default → Blue
]
```

### Zoom-Responsive Line Width

Make lines thicker at higher zoom levels:

```javascript
"line-width": [
  "interpolate",
  ["linear"],
  ["zoom"],
  4,  1.0,     // At zoom 4, width = 1 pixel
  8,  2.0,     // At zoom 8, width = 2 pixels
  12, 4.0,     // At zoom 12, width = 4 pixels
  14, 6.0      // At zoom 14, width = 6 pixels
]
```

### Dashed Lines

Create dashed or dotted line patterns:

```javascript
"line-dasharray": [2, 2]    // Alternates 2px dash, 2px gap
"line-dasharray": [5, 3]    // 5px dash, 3px gap (longer dashes)
```

## Common Paint Properties

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `fill-color` | Color | Polygon fill color | `"#ff0000"` |
| `fill-opacity` | Number | Fill transparency (0-1) | `0.65` |
| `line-color` | Color | Line/outline color | `"#003366"` |
| `line-width` | Number | Line thickness in pixels | `2.0` |
| `line-opacity` | Number | Line transparency (0-1) | `0.9` |
| `circle-radius` | Number | Point radius in pixels | `5` |
| `circle-color` | Color | Point fill color | `"#ff0000"` |
| `circle-opacity` | Number | Point transparency (0-1) | `0.85` |
| `circle-stroke-width` | Number | Point border thickness | `1` |
| `circle-stroke-color` | Color | Point border color | `"#ffffff"` |

## Data-Driven Styling Patterns

### Match Expression (Categorical Data)
Use for categorical attributes like road type, landmark type:

```javascript
["match",
  ["get", "PROPERTY_NAME"],  // Get the property
  "value1", "result1",       // If property == "value1"
  "value2", "result2",       // If property == "value2"
  "default_result"           // Default if no match
]
```

### Interpolate Expression (Numeric Data)
Use for numeric attributes that need continuous scaling:

```javascript
["interpolate",
  ["linear"],               // Interpolation method (linear, exponential, etc.)
  ["zoom"],                 // Can also use numeric properties: ["get", "PROPERTY"]
  0,    "#ff0000",          // At zoom 0 → Red
  10,   "#ffff00",          // At zoom 10 → Yellow
  20,   "#00ff00"           // At zoom 20 → Green
]
```

## Tips & Best Practices

1. **Use opacity for layering**: Combine semi-transparent fills with outlines for better visibility
2. **Zoom-responsive styling**: Use `["zoom"]` to adapt line widths and point sizes at different zoom levels
3. **Color schemes**: Use colorblind-friendly palettes for accessibility
4. **Test in browser**: Use the browser's Developer Tools (F12) to debug styling issues
5. **Filters**: Use the `filter` property to control which features are displayed:
   ```javascript
   filter: ["==", "$type", "Polygon"]  // Only show polygons
   filter: ["!=", "STATEFP", "02"]     // Hide Alaska
   ```

## Modifying Styles in view_mvt.html

To customize styling for your datasets:

1. Open `server/view_mvt.html` in a text editor
2. Find the dataset type section (Polygons, Points, or Lines)
3. Modify the `paint` properties
4. Refresh the browser to see changes immediately

## Example: Custom County Visualization

To highlight specific states with different colors:

```javascript
"fill-color": [
  "match",
  ["get", "STATEFP"],
  "06", "#ff0000",      // California → Red
  "48", "#0000ff",      // Texas → Blue
  "36", "#00ff00",      // New York → Green
  "#cccccc"             // Others → Gray
]
```

## Resources

- [MapLibre GL Documentation](https://maplibre.org/maplibre-gl-js/docs/)
- [Style Spec](https://maplibre.org/maplibre-style-spec/)
- [Paint Properties](https://maplibre.org/maplibre-style-spec/layers/)
