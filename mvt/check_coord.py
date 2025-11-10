import mapbox_vector_tile

with open("tiles_out/0/0/0.mvt", "rb") as f:
    data = mapbox_vector_tile.decode(f.read())

layer = data["layer0"]
print("Features:", len(layer["features"]))
first_geom = layer["features"][0]["geometry"]
coords = first_geom["coordinates"]

# Flatten coordinate list (works for Polygon)
all_x = [pt[0] for ring in coords for pt in ring]
all_y = [pt[1] for ring in coords for pt in ring]

print("X range:", min(all_x), max(all_x))
print("Y range:", min(all_y), max(all_y))