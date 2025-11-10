import argparse
import folium
import geopandas as gpd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Visualise GeoJSON/GeoParquet on a folium map")
    ap.add_argument("--input", required=True, help="Path to GeoJSON/GeoParquet file")
    ap.add_argument("--out", default="tiles_map.html", help="Output HTML file")
    args = ap.parse_args()

    ext = Path(args.input).suffix.lower()
    if ext in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(args.input)
    else:
        gdf = gpd.read_file(args.input)

    if gdf.empty:
        raise SystemExit(f"No features in {args.input}")

    # Ensure WGS84 for folium
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    # Tooltip column: use existing 'id' or synthesize one
    tooltip_col = "id" if "id" in gdf.columns else "_fid"
    if tooltip_col == "_fid":
        gdf["_fid"] = gdf.index.astype(str)

    m = folium.Map(tiles="CartoDB positron")
    # Fit to bounds
    minx, miny, maxx, maxy = gdf.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    folium.GeoJson(
        data=gdf.__geo_interface__,         # avoids an extra to_json() string build
        style_function=lambda _f: {"fillOpacity": 0.05, "weight": 1},
        tooltip=folium.GeoJsonTooltip(fields=[tooltip_col]),
        name="layers"
    ).add_to(m)

    m.save(args.out)
    print(f"Wrote map to {args.out}")

if __name__ == "__main__":
    main()
