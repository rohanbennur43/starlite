import mapbox_vector_tile
import json
from pathlib import Path

# Choose a tile to inspect
tile_path = Path("tiles_out/0/0/0.mvt")  # change if needed

with open(tile_path, "rb") as f:
    data = f.read()

decoded = mapbox_vector_tile.decode(data)

print("\n=== Layers ===")
for layer_name, layer in decoded.items():
    print(f"Layer: {layer_name}")
    print(f"  Number of features: {len(layer['features'])}")
    if layer['features']:
        # show first feature structure
        first = layer['features'][0]
        print("  Geometry type:", first['geometry']['type'])
        print("  Example properties:")
        print(json.dumps(first['properties'], indent=2))