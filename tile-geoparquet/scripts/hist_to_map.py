from __future__ import annotations
import argparse, json, math, os, base64
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
# matplotlib is used only for colormaps; install via: pip install matplotlib
from matplotlib import cm
import math

def _merc_m_to_deg(x_m: float, y_m: float):
    R = 6378137.0
    lon = (x_m / R) * 180.0 / math.pi
    lat = (2.0 * math.atan(math.exp(y_m / R)) - math.pi/2.0) * 180.0 / math.pi
    return lat, lon

def _bbox_to_wgs84(minx: float, miny: float, maxx: float, maxy: float):
    # Heuristically detect if already in degrees
    if all(abs(x) <= 360 for x in [minx, maxx]) and all(abs(y) <= 180 for y in [miny, maxy]):
        # Already degrees (lat/lon)
        south, west, north, east = miny, minx, maxy, maxx
    else:
        # Convert from EPSG:3857 meters to lat/lon degrees
        south, west = _merc_m_to_deg(minx, miny)
        north, east = _merc_m_to_deg(maxx, maxy)
        # Clamp latitude to Web Mercator valid range
        lat_min = -85.05112878
        lat_max = 85.05112878
        south = max(min(south, lat_max), lat_min)
        north = max(min(north, lat_max), lat_min)
    return south, west, north, east

HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Histogram viewer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  html,body,#map {{ height:100%; margin:0 }}
  .panel {{ position:absolute; top:10px; left:10px; background:rgba(255,255,255,.9); padding:8px 10px; border-radius:8px; font:14px system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
  .legend {{ margin-top:6px; display:flex; align-items:center; gap:8px }}
  .legend img {{ width:180px; height:12px; }}
  .meta {{ color:#666; font-size:12px; margin-top:4px }}
</style>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map = L.map('map', {{ worldCopyJump:true }}).setView([20,0], 2);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19, attribution: '&copy; OpenStreetMap' }}).addTo(map);

// Layers (injected)
const layers = [];
{LAYER_JS}

// Controls
const pnl = L.control({{position:'topleft'}});
pnl.onAdd = function() {{
  const div = L.DomUtil.create('div','panel');
  div.innerHTML = `
    <div><b>Histogram viewer</b></div>
    <div>Colormap: <code>{COLORMAP}</code> · Scale: log1p (global)</div>
    <div class="legend"><img src="data:image/png;base64,{LEGEND_PNG}"><span class="meta">{MIN_TXT} → {MAX_TXT}</span></div>
    <div>Opacity: <input id="op" type="range" min="0" max="1" step="0.05" value="{OPACITY}" /></div>
    <div class="meta">{COUNT} tile images</div>
  `;
  return div;
}};
pnl.addTo(map);

document.getElementById('op').addEventListener('input', (e)=>{{
  const v = parseFloat(e.target.value);
  layers.forEach(o => o.setOpacity(v));
}});

// Fit bounds if we have any layer
if (layers.length > 0) {{
  const g = L.featureGroup(layers);
  map.fitBounds(g.getBounds().pad(0.05));
}}
</script>
"""

def _read_meta(json_path: Path):
    try:
        j = json.loads(json_path.read_text())
        bbox = j.get("bbox_out") or j.get("bbox")  # [minx,miny,maxx,maxy] in EPSG:3857
        tile_id = j.get("tile_id", json_path.stem)
        lvl = j.get("level", 0)
        return tile_id, lvl, bbox
    except Exception:
        return None, None, None

def _make_colorer(name: str):
    name = (name or "magma").lower()
    # robust set
    cmap = dict(
        viridis=cm.get_cmap("viridis"),
        magma=cm.get_cmap("magma"),
        inferno=cm.get_cmap("inferno"),
        plasma=cm.get_cmap("plasma"),
        cividis=cm.get_cmap("cividis"),
    ).get(name, cm.get_cmap("magma"))

    def colorize(arr: np.ndarray, opacity: float) -> Image.Image:
        # arr in [0,1]; zeros become fully transparent
        rgba = (cmap(arr) * 255).astype(np.uint8)   # (H,W,4)
        # alpha: 0 where arr==0 else opacity*255
        alpha = (arr > 0).astype(np.uint8) * int(round(opacity * 255))
        rgba[..., 3] = alpha
        return Image.fromarray(rgba, mode="RGBA")
    return colorize, name

def _encode_png(img: Image.Image) -> bytes:
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _legend_png(cmap_name: str) -> str:
    w, h = 256, 12
    xs = np.linspace(0,1,w,endpoint=True)
    grad = np.tile(xs, (h,1))
    colorize, _ = _make_colorer(cmap_name)
    png = _encode_png(colorize(grad, opacity=1.0))
    return base64.b64encode(png).decode("ascii")

def main():
    ap = argparse.ArgumentParser(description="Color viewer for histogram pyramids → Leaflet")
    ap.add_argument("hist_dir", help="Directory with *_L*.npy + .json (from hist_pyramid)")
    ap.add_argument("--level", type=int, default=0, help="Which pyramid level to render (default: 0)")
    ap.add_argument("--colormap", default="magma", help="viridis|magma|inferno|plasma|cividis")
    ap.add_argument("--opacity", type=float, default=0.70, help="Overlay opacity (0..1)")
    ap.add_argument("--outfile", default="_web/index.html", help="Output HTML path (inside hist_dir if relative)")
    ap.add_argument("--minmax", nargs=2, type=float, default=None,
                    help="Override global min/max after log1p for normalization, e.g. 0 12")
    args = ap.parse_args()

    hist_dir = Path(args.hist_dir)
    out_html = Path(args.outfile)
    if not out_html.is_absolute():
        out_html = hist_dir / out_html
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # Gather all L{level}.npy + json
    pairs: List[Tuple[Path, Path]] = []
    for npy in hist_dir.glob(f"*__*L{args.level}.npy"):
        j = npy.with_suffix(".json")
        if j.exists(): pairs.append((npy, j))
    # also support earlier naming {tile}_L{level}.npy
    for npy in hist_dir.glob(f"*_L{args.level}.npy"):
        j = npy.with_suffix(".json")
        if j.exists() and (npy, j) not in pairs:
            pairs.append((npy, j))

    if not pairs:
        print(f"[hist_to_map] No hist arrays found at level {args.level} in {hist_dir}")
        return

    # Load arrays, compute global min/max in log1p-space
    logmins, logmaxs = [], []
    metas: Dict[str, dict] = {}
    arrays: Dict[str, np.ndarray] = {}
    sums: Dict[str, float] = {}
    for npy, jpath in pairs:
        tile_id, lvl, bbox = _read_meta(jpath)
        if bbox is None:  # skip if no bbox
            continue
        arr = np.load(npy)  # float64 or float32
        arr = np.nan_to_num(arr, nan=0.0, neginf=0.0, posinf=0.0)
        arrays[npy.stem] = arr
        metas[npy.stem]  = dict(tile_id=tile_id, level=lvl, bbox=bbox)
        sums[npy.stem]   = float(arr.sum())
        logv = np.log1p(arr)
        logmins.append(float(logv.min()))
        logmaxs.append(float(logv.max()))

    if not arrays:
        print("[hist_to_map] No arrays with bbox metadata found.")
        return

    gmin = min(logmins); gmax = max(logmaxs)
    if args.minmax is not None:
        gmin, gmax = float(args.minmax[0]), float(args.minmax[1])
    if not math.isfinite(gmin): gmin = 0.0
    if not math.isfinite(gmax) or gmax <= gmin: gmax = gmin + 1.0

    colorize, cmap_name = _make_colorer(args.colormap)

    # Sort keys so sparser tiles sit on top (less occlusion)
    order = sorted(arrays.keys(), key=lambda k: sums[k])  # low sum last → on top
    layer_js = []
    png_count = 0

    tiles_dir = (out_html.parent / "tiles")
    tiles_dir.mkdir(parents=True, exist_ok=True)

    for key in order:
        arr = arrays[key]
        meta = metas[key]
        tile_id = meta["tile_id"]
        minx, miny, maxx, maxy = meta["bbox"]

        # normalize in log1p space
        norm = (np.log1p(arr) - gmin) / (gmax - gmin)
        norm = np.clip(norm, 0.0, 1.0)
        # zero → transparent (alpha=0), else colormap color with requested opacity
        img = colorize(norm.astype(np.float32), opacity=args.opacity)

        # Save PNG
        png_path = tiles_dir / f"{tile_id}_L{args.level}.png"
        img.save(png_path)
        png_count += 1

        # Convert bbox from EPSG:3857 meters to WGS84 degrees for Leaflet
        south, west, north, east = _bbox_to_wgs84(minx, miny, maxx, maxy)
        print(f"[hist_to_map] {tile_id}: bounds (deg) SW=({south:.4f},{west:.4f}) NE=({north:.4f},{east:.4f})")

        # Build Leaflet ImageOverlay
        layer_js.append(
            f"""
            (function(){{
              const img = L.imageOverlay("{(png_path.relative_to(out_html.parent)).as_posix()}",
                [[{south:.6f}, {west:.6f}], [{north:.6f}, {east:.6f}]],
                {{opacity:{args.opacity}, zIndex: 400}});
              img.addTo(map);
              layers.push(img);
            }})();
            """.strip()
        )

    legend_png_b64 = _legend_png(cmap_name)
    html = HTML_TEMPLATE.format(
        LAYER_JS="\n".join(layer_js),
        COLORMAP=cmap_name,
        LEGEND_PNG=legend_png_b64,
        MIN_TXT=f"{gmin:.2f}",
        MAX_TXT=f"{gmax:.2f}",
        OPACITY=f"{args.opacity:.2f}",
        COUNT=png_count,
    )
    out_html.write_text(html, encoding="utf-8")
    print(f"Built viewer: file://{out_html}")
    print(f"PNG tiles: {png_count} → {tiles_dir}")
    print("Tip: serve the folder to avoid browser CORS issues, e.g.:  python -m http.server -d", out_html.parent)
if __name__ == "__main__":
    main()