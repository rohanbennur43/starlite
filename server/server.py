from flask import Flask, send_file, Response
from flask_cors import CORS
from tiler.tiler import VectorTiler

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.get("/<dataset>/<int:z>/<int:x>/<int:y>.mvt")
def serve_tile(dataset, z, x, y):
    tiler = VectorTiler(f"../datasets/{dataset}")
    tile_bytes = tiler.get_tile(z, x, y)
    return Response(tile_bytes, mimetype="application/vnd.mapbox-vector-tile")

if __name__ == "__main__":
    app.run(debug=True)
