# Starlet Tile Server

Flask application that serves vector tiles (MVT) and dataset metadata from
pre-processed GeoParquet data.

## Running

```bash
# Install the package
pip install -e .

# Start the server
starlet serve --dir <data_directory> [--host 0.0.0.0] [--port 8765] [--cache-size 256]
```

Or via the Makefile (uses the legacy `server/server.py` entry point):

```bash
make server   # http://127.0.0.1:5000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Interactive dataset selector page |
| `GET` | `/api/datasets` | List all available datasets |
| `GET` | `/datasets.json` | Search datasets by name |
| `GET` | `/datasets/<dataset>.json` | Dataset metadata |
| `GET` | `/datasets/<dataset>.html` | Dataset detail page |
| `GET` | `/<dataset>/<z>/<x>/<y>.mvt` | Mapbox Vector Tile |
| `GET` | `/datasets/<dataset>/features.<fmt>` | Download features (csv/geojson) |
| `POST` | `/datasets/<dataset>/features.<fmt>` | Download with geometry filter |
| `GET` | `/datasets/<dataset>/features/sample.json` | Sample non-geometry attributes |
| `GET` | `/datasets/<dataset>/features/sample.geojson` | Sample record with geometry |
| `GET` | `/api/datasets/<dataset>/stats` | Precomputed attribute statistics |
| `POST` | `/datasets/<dataset>/styles.json` | LLM-generated styling suggestions |

## Styles Endpoint

`POST /datasets/<dataset>/styles.json` sends dataset attribute statistics to a
configured LLM and returns an array of map styling rules.

**Request body** (optional JSON):

```json
{
  "features": ["population", "area"]
}
```

Omit the body (or send `{}`) to let the LLM analyze all attributes.

**Response** — a JSON array:

```json
[
  {
    "attribute": "STATEFP",
    "type": "categorical",
    "fillColor": {"48": "#e31a1c", "13": "#1f78b4"},
    "strokeColor": "#333333",
    "explanation": "Color counties by state FIPS code."
  }
]
```

Returns `[]` on any error (missing dataset, LLM failure, etc.).

### LLM Configuration

The styles endpoint uses whichever LLM provider is selected by the
`LLM_PROVIDER` environment variable. See [`llm/README.md`](llm/README.md) for
full configuration details.

Quick start:

```bash
# Gemini (default)
export GEMINI_API_KEY=your-key-here

# — or — Ollama (local)
export LLM_PROVIDER=ollama
# make sure `ollama serve` is running
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `LLM_PROVIDER` | `gemini` | LLM backend (`gemini` or `ollama`) |
| `GEMINI_API_KEY` | — | Required when using the Gemini provider |
| `OLLAMA_MODEL` | `llama3` | Model name for the Ollama provider |
