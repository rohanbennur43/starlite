from flask import Flask, Response, render_template, send_from_directory, request
from flask_cors import CORS
from tiler.tiler import VectorTiler
from pathlib import Path
import argparse
import os
import json
import signal
import sys
import logging
from download_service import DatasetFeatureService

# Get the directory where this script is located
SERVER_DIR = Path(__file__).parent

app = Flask(__name__, template_folder=str(SERVER_DIR / 'templates'))
CORS(app, resources={r"/*": {"origins": "*"}})

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=os.environ.get("TILE_ROOT", "datasets"))
args = parser.parse_args()

DATA_ROOT = Path(args.root)

TILER_CACHE = {}
feature_service = DatasetFeatureService(DATA_ROOT)

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
    logging.info("Serving index page")
    return render_template("index.html")

@app.route("/<path:filename>")
def serve_file(filename):
    """Serve static files - this must be LAST to catch everything else"""
    file_path = SERVER_DIR / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory(str(SERVER_DIR), filename)
    return "File not found", 404

@app.get("/datasets/<dataset>/features.<format>")
def download_features(dataset, format):
    """
    Download dataset features in specified format with optional spatial filtering.
    
    Query params:
        mbr: Optional Minimum Bounding Rectangle as "minx,miny,maxx,maxy"
    
    Examples:
        GET /datasets/TIGER2018_COUNTY/features.geojson?mbr=-120,30,-100,40
        GET /datasets/TIGER2018_COUNTY/features.geojson  (entire dataset)
        GET /datasets/TIGER2018_POINTLM/features.csv?mbr=-120,30,-100,40
        GET /datasets/TIGER2018_POINTLM/features.csv  (entire dataset)
    """
    try:
        # Get optional MBR from query params
        mbr_string = request.args.get('mbr', default=None)
        
        # Get features stream
        feature_stream = feature_service.get_features_stream(dataset, format, mbr_string)
        
        # Determine mime type and filename
        mime_type = feature_service.get_mime_type(format)
        if mbr_string:
            filename = f"{dataset}_{mbr_string.replace(',', '_')}.{format}"
        else:
            filename = f"{dataset}_full.{format}"
        
        # Stream response
        return Response(
            feature_stream,
            mimetype=mime_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except ValueError as e:
        return {"error": str(e)}, 400
    except FileNotFoundError as e:
        return {"error": str(e)}, 404
    except Exception as e:
        return {"error": f"Internal error: {str(e)}"}, 500

@app.post("/datasets/<dataset>/features.<format>")
def download_features_with_geometry(dataset, format):
    """
    Download dataset features in the specified format with optional spatial filtering.
    Supports GeoJSON payloads for custom geometry queries.

    Query params:
        mbr: Optional Minimum Bounding Rectangle as "minx,miny,maxx,maxy"

    Payload:
        GeoJSON representation of a custom geometry to filter the dataset.
    """
    dataset_path = DATA_ROOT / dataset

    if not dataset_path.exists() or not dataset_path.is_dir():
        return {"error": "Dataset not found"}, 404

    try:
        # Check for GeoJSON payload
        geojson_payload = request.get_json()
        mbr_string = request.args.get("mbr", default=None)

        if geojson_payload:
            # Use GeoJSON geometry for filtering
            geometry = geojson_payload.get("geometry")
            if not geometry:
                return {"error": "Invalid GeoJSON payload: 'geometry' field is required"}, 400

            feature_stream = feature_service.get_features_stream(dataset, format, geometry=geometry)
        else:
            # Use MBR for filtering if no GeoJSON payload
            feature_stream = feature_service.get_features_stream(dataset, format, mbr_string)

        # Determine mime type and filename
        mime_type = feature_service.get_mime_type(format)
        filename = f"{dataset}_filtered.{format}" if geojson_payload else f"{dataset}_mbr.{format}"

        # Stream response
        return Response(
            feature_stream,
            mimetype=mime_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except ValueError as e:
        return {"error": str(e)}, 400
    except FileNotFoundError as e:
        return {"error": str(e)}, 404
    except Exception as e:
        return {"error": f"Internal error: {str(e)}"}, 500

@app.get("/api/datasets/<dataset>/stats")
def get_dataset_stats(dataset):
    """
    Return precomputed attribute statistics for a dataset.
    """
    stats_path = DATA_ROOT / dataset / "stats" / "attributes.json"

    if not stats_path.exists():
        return {"error": "Stats not found for dataset"}, 404

    try:
        with open(stats_path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to load stats: {str(e)}"}, 500

@app.get("/datasets.json")
def search_datasets():
    """
    Searches for datasets that match the given query and returns them.
    Query parameter:
        q: The search query to filter datasets by name or ID.
    """
    query = request.args.get("q", default=None)
    datasets = []

    if DATA_ROOT.exists():
        for d in DATA_ROOT.iterdir():
            if d.is_dir():
                dataset_metadata = {
                    "id": d.name,
                    "name": d.name.replace("_", " ").title(),
                    "size": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()),
                }

                # If a query is provided, filter datasets by name or ID
                if query is None or query.lower() in d.name.lower():
                    datasets.append(dataset_metadata)

    return json.dumps({"datasets": datasets}, indent=2)

@app.get("/datasets/<dataset>.json")
def get_dataset_metadata(dataset):
    """
    Returns detailed metadata for a specific dataset.
    """
    dataset_path = DATA_ROOT / dataset

    if not dataset_path.exists() or not dataset_path.is_dir():
        return {"error": "Dataset not found"}, 404

    try:
        # Example metadata: name, ID, size, and number of files
        metadata = {
            "id": dataset,
            "name": dataset.replace("_", " ").title(),
            "size": sum(f.stat().st_size for f in dataset_path.rglob("*") if f.is_file()),
            "file_count": sum(1 for f in dataset_path.rglob("*") if f.is_file()),
        }
        return json.dumps(metadata, indent=2)
    except Exception as e:
        return {"error": f"Failed to retrieve metadata: {str(e)}"}, 500

@app.get("/datasets/<dataset>.html")
def visualize_dataset(dataset):
    """
    Returns a minimal HTML page for visualizing the dataset.
    """
    dataset_path = DATA_ROOT / dataset

    if not dataset_path.exists() or not dataset_path.is_dir():
        return "<h1>Dataset not found</h1>", 404

    try:
        # Render a simple HTML page for visualization
        return render_template(
            "view_dataset.html",
            dataset_id=dataset,
            dataset_name=dataset.replace("_", " ").title(),
        )
    except Exception as e:
        return f"<h1>Failed to render visualization: {str(e)}</h1>", 500

@app.get("/datasets/<dataset>/features/sample.json")
def get_sample_non_geometry_attributes(dataset):
    """
    Return non-geometry attributes of the first record that matches the given MBR.

    Query params:
        mbr: Minimum Bounding Rectangle as "minx,miny,maxx,maxy"
    """
    dataset_path = DATA_ROOT / dataset

    if not dataset_path.exists() or not dataset_path.is_dir():
        return {"error": "Dataset not found"}, 404

    try:
        # Get MBR from query params
        mbr_string = request.args.get("mbr", default=None)
        if not mbr_string:
            return {"error": "MBR query parameter is required"}, 400

        # Fetch the first matching record
        sample_record = feature_service.get_sample_record(dataset, mbr_string, include_geometry=False)
        if not sample_record:
            return {"error": "No matching record found"}, 404

        return json.dumps(sample_record, indent=2)

    except ValueError as e:
        return {"error": str(e)}, 400
    except FileNotFoundError as e:
        return {"error": str(e)}, 404
    except Exception as e:
        return {"error": f"Internal error: {str(e)}"}, 500

@app.get("/datasets/<dataset>/features/sample.geojson")
def get_sample_with_geometry(dataset):
    """
    Return the first record that matches the given MBR, including geometry.

    Query params:
        mbr: Minimum Bounding Rectangle as "minx,miny,maxx,maxy"
    """
    dataset_path = DATA_ROOT / dataset

    if not dataset_path.exists() or not dataset_path.is_dir():
        return {"error": "Dataset not found"}, 404

    try:
        # Get MBR from query params
        mbr_string = request.args.get("mbr", default=None)
        if not mbr_string:
            return {"error": "MBR query parameter is required"}, 400

        # Fetch the first matching record with geometry
        sample_record = feature_service.get_sample_record(dataset, mbr_string, include_geometry=True)
        if not sample_record:
            return {"error": "No matching record found"}, 404

        return json.dumps(sample_record, indent=2)

    except ValueError as e:
        return {"error": str(e)}, 400
    except FileNotFoundError as e:
        return {"error": str(e)}, 404
    except Exception as e:
        return {"error": f"Internal error: {str(e)}"}, 500


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\n\nShutting down server...")
        sys.exit(0)
    
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # use_reloader=False prevents issues with port still being in use
        # threaded=True allows multiple requests simultaneously
        app.run(debug=False, use_reloader=False, threaded=True, host='127.0.0.1', port=5000)
    except KeyboardInterrupt:
        print("\nServer interrupted")
        sys.exit(0)
