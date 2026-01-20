from flask import Flask, Response, render_template, send_from_directory
from flask_cors import CORS
from tiler.tiler import VectorTiler
from pathlib import Path
import argparse
import os
import json

# Get the directory where this script is located
SERVER_DIR = Path(__file__).parent

app = Flask(__name__, template_folder=str(SERVER_DIR / 'templates'))
CORS(app, resources={r"/*": {"origins": "*"}})

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=os.environ.get("TILE_ROOT", "datasets"))
args = parser.parse_args()

DATA_ROOT = Path(args.root)

TILER_CACHE = {}

def get_tiler(dataset):
    if dataset not in TILER_CACHE:
        TILER_CACHE[dataset] = VectorTiler(str(DATA_ROOT / dataset))
    return TILER_CACHE[dataset]

@app.get("/<dataset>/<int:z>/<int:x>/<int:y>.mvt")
def serve_tile(dataset, z, x, y):
    tiler = get_tiler(dataset)
    return Response(tiler.get_tile(z, x, y), mimetype="application/vnd.mapbox-vector-tile")

@app.get("/api/datasets")
def list_datasets():
    """Return list of available datasets"""
    datasets = []
    if DATA_ROOT.exists():
        datasets = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
    return json.dumps({"datasets": datasets})

@app.get("/")
def index():
    """Serve the index page with dataset list"""
    return render_template("index.html")

@app.route("/<path:filename>")
def serve_file(filename):
    """Serve static files - this must be LAST to catch everything else"""
    file_path = SERVER_DIR / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory(str(SERVER_DIR), filename)
    return "File not found", 404

if __name__ == "__main__":
    app.run(debug=False)
