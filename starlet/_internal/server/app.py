"""Flask application factory for the starlet tile server."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional

import json
import logging
import os
import re

from flask import Flask, Response, render_template, send_from_directory, request
from flask_cors import CORS

from .tiler.tiler import VectorTiler
from .download_service import DatasetFeatureService

logger = logging.getLogger(__name__)


def create_app(
    data_dir: str,
    cache_size: int = 256,
    log_level: Optional[str] = None,
) -> Flask:
    """Create and configure a Flask tile server application.

    Parameters
    ----------
    data_dir : str
        Root directory containing dataset subdirectories.
    cache_size : int
        Number of tiles to keep in the in-memory LRU cache.
    log_level : str, optional
        Logging level (e.g. "INFO", "DEBUG"). Defaults to ``LOG_LEVEL`` env
        var or ``"INFO"``.

    Returns
    -------
    Flask
        Configured Flask application ready to be served.
    """
    level = log_level or os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    template_dir = str(Path(__file__).parent / "templates")
    app = Flask(__name__, template_folder=template_dir)
    CORS(app, resources={r"/*": {"origins": "*"}})

    data_root = Path(data_dir)
    tiler_cache: dict[str, VectorTiler] = {}
    feature_service = DatasetFeatureService(data_root)

    def get_tiler(dataset: str) -> VectorTiler:
        if dataset not in tiler_cache:
            tiler_cache[dataset] = VectorTiler(str(data_root / dataset), memory_cache_size=cache_size)
        return tiler_cache[dataset]

    @app.get("/<dataset>/<int:z>/<int:x>/<int:y>.mvt")
    def serve_tile(dataset, z, x, y):
        t0 = perf_counter()
        tiler = get_tiler(dataset)
        data = tiler.get_tile(z, x, y)
        elapsed_ms = (perf_counter() - t0) * 1000
        logger.info("[TileRequest] dataset=%s z=%d x=%d y=%d bytes=%d elapsed=%.1fms",
                    dataset, z, x, y, len(data), elapsed_ms)
        return Response(data, mimetype="application/vnd.mapbox-vector-tile")

    @app.get("/api/datasets")
    def list_datasets():
        datasets = []
        if data_root.exists():
            datasets = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
        return json.dumps({"datasets": datasets})

    @app.get("/")
    def index():
        logger.info("Serving index page")
        return render_template("index.html")

    @app.route("/<path:filename>")
    def serve_file(filename):
        server_dir = Path(__file__).parent
        file_path = server_dir / filename
        if file_path.exists() and file_path.is_file():
            return send_from_directory(str(server_dir), filename)
        return "File not found", 404

    @app.get("/datasets/<dataset>/features.<format>")
    def download_features(dataset, format):
        try:
            mbr_string = request.args.get('mbr', default=None)
            feature_stream = feature_service.get_features_stream(dataset, format, mbr_string)
            mime_type = feature_service.get_mime_type(format)
            if mbr_string:
                filename = f"{dataset}_{mbr_string.replace(',', '_')}.{format}"
            else:
                filename = f"{dataset}_full.{format}"
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
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            geojson_payload = request.get_json()
            mbr_string = request.args.get("mbr", default=None)
            if geojson_payload:
                geometry = geojson_payload.get("geometry")
                if not geometry:
                    return {"error": "Invalid GeoJSON payload: 'geometry' field is required"}, 400
                feature_stream = feature_service.get_features_stream(dataset, format, geometry=geometry)
            else:
                feature_stream = feature_service.get_features_stream(dataset, format, mbr_string)
            mime_type = feature_service.get_mime_type(format)
            filename = f"{dataset}_filtered.{format}" if geojson_payload else f"{dataset}_mbr.{format}"
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
        stats_path = data_root / dataset / "stats" / "attributes.json"
        if not stats_path.exists():
            return {"error": "Stats not found for dataset"}, 404
        try:
            with open(stats_path, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to load stats: {str(e)}"}, 500

    @app.get("/datasets.json")
    def search_datasets():
        query = request.args.get("q", default=None)
        datasets = []
        if data_root.exists():
            for d in data_root.iterdir():
                if d.is_dir():
                    dataset_metadata = {
                        "id": d.name,
                        "name": d.name.replace("_", " ").title(),
                        "size": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()),
                    }
                    if query is None or query.lower() in d.name.lower():
                        datasets.append(dataset_metadata)
        return json.dumps({"datasets": datasets}, indent=2)

    @app.get("/datasets/<dataset>.json")
    def get_dataset_metadata(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
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
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return "<h1>Dataset not found</h1>", 404
        try:
            return render_template(
                "view_dataset.html",
                dataset_id=dataset,
                dataset_name=dataset.replace("_", " ").title(),
            )
        except Exception as e:
            return f"<h1>Failed to render visualization: {str(e)}</h1>", 500

    @app.get("/datasets/<dataset>/features/sample.json")
    def get_sample_non_geometry_attributes(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            mbr_string = request.args.get("mbr", default=None)
            if not mbr_string:
                return {"error": "MBR query parameter is required"}, 400
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
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            mbr_string = request.args.get("mbr", default=None)
            if not mbr_string:
                return {"error": "MBR query parameter is required"}, 400
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

    @app.post("/datasets/<dataset>/styles.json")
    def generate_styles(dataset):
        empty = json.dumps([])
        json_ct = {"Content-Type": "application/json"}

        # Validate dataset
        dataset_path = data_root / dataset
        if not dataset_path.is_dir():
            return empty, 200, json_ct

        # Load attributes.json
        stats_path = dataset_path / "stats" / "attributes.json"
        if not stats_path.exists():
            return empty, 200, json_ct

        try:
            with open(stats_path) as f:
                stats = json.load(f)
        except Exception:
            return empty, 200, json_ct

        attributes = stats.get("attributes", [])
        if not attributes:
            return empty, 200, json_ct

        # Determine mode: targeted (features list provided) vs auto (all attributes)
        body = request.get_json(silent=True) or {}
        requested_features = body.get("features")

        if requested_features:
            attr_subset = [a for a in attributes if a["name"] in requested_features]
            instruction = (
                "Generate styling suggestions for these specific attributes: "
                + ", ".join(requested_features)
            )
        else:
            attr_subset = attributes
            instruction = "Analyze all attributes and suggest the best styling rules for map visualization."

        if not attr_subset:
            return empty, 200, json_ct

        # Load prompt template
        prompt_path = Path(__file__).parent / "llm" / "prompt.md"
        try:
            prompt_template = prompt_path.read_text()
        except Exception:
            return empty, 200, json_ct

        prompt = prompt_template.replace(
            "{{ATTRIBUTES_JSON}}", json.dumps(attr_subset, indent=2)
        ).replace(
            "{{INSTRUCTION}}", instruction
        )

        # Call LLM
        try:
            from .llm.factory import LLMFactory
            provider = LLMFactory.get_default_provider()
            raw = provider.generate_response(prompt)
        except Exception as e:
            logger.error("[Styles] LLM call failed for %s: %s", dataset, e)
            return empty, 200, json_ct

        # Parse JSON array from response
        try:
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if not match:
                return empty, 200, json_ct
            styles = json.loads(match.group())
            if not isinstance(styles, list):
                return empty, 200, json_ct
        except Exception as e:
            logger.error("[Styles] Failed to parse LLM response for %s: %s", dataset, e)
            return empty, 200, json_ct

        return json.dumps(styles), 200, json_ct

    return app
