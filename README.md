# Geospatial Data Visualiser
This repository provides a simple Makefile driven workflow to generate spatial tiles and Mapbox Vector Tiles from Parquet datasets. All steps run through a few make targets and produce logs automatically.

## Setup

```shell
python -m venv .venv
source .venv/bin/activate
pip install -e ./tile-geoparquet
pip install -r requirements.txt
```

## Make Targets

### `make tiles INPUT=<input-path>`
Runs the tiling pipeline.

- Splits the input dataset into 40 tiles.
- Uses Z order sorting and sampling.
- Saves outputs in `datasets/<dataset_name>/`.
- Writes logs to `logs_<dataset_name>.txt`.

### `make mvt`
Generates Mapbox Vector Tiles.

- Reads tiles from `datasets/<dataset_name>/`.
- Writes MVTs to `mvt_out/<dataset_name>/`.
- Logs are written to the same dataset log file.

### `make all`
Runs `tiles` followed by `mvt` in one command.

### `make server`
Starts a Flask server with interactive dataset viewer.

- Serves the `datasets/` directory as MVT tiles
- Opens `http://127.0.0.1:5000` in your browser
- Shows an interactive dataset selector with search functionality
- Automatically detects dataset type and applies appropriate visualization:
  - **Polygons** (COUNTY, PLACE): State-based coloring with outlines
  - **Points** (POINTLM): MTFCC category coloring with hover popups
  - **Lines** (ROADS, RAILS, PRISEC): Color-coded by road type or generic styling

#### Usage
```bash
make server
```

Or manually:
```bash
source bigdata/bin/activate
python3 server/server.py --root datasets
```

Then visit `http://127.0.0.1:5000`

#### Dynamic Dataset Loading
You can also specify a dataset directly via URL:
```
http://127.0.0.1:5000/view_mvt.html?dataset=TIGER2018_COUNTY
http://127.0.0.1:5000/view_mvt.html?dataset=TIGER2018_POINTLM
http://127.0.0.1:5000/view_mvt.html?dataset=OSM2015_33
```

#### API Endpoints
- `GET /` - Interactive dataset selector page
- `GET /api/datasets` - Returns JSON list of available datasets
- `GET /<dataset>/<z>/<x>/<y>.mvt` - Serves MVT tile for specified dataset
- `GET /view_mvt.html` - Interactive map viewer (use with `?dataset=` param)

### `make clean`
Removes all generated tiles, MVTs, and logs.

## Example
```bash
make all INPUT=../extras/original_datasets/OSM2015_33.parquet
make server
```

Then browse to `http://127.0.0.1:5000` and select a dataset to visualize.

## Features

### Interactive Dataset Selector
- Browse all available datasets in `datasets/` folder
- Search/filter datasets by name
- One-click access to visualization

### Smart Visualization
The viewer automatically detects geometry types and applies appropriate rendering:

- **Polygons**: Colored by state (STATEFP) with semi-transparent fills and outlines
- **Points**: Colored by MTFCC category (landmarks, schools, hospitals, etc.) with hover information
- **Lines**: Styled by type (roads, rails, etc.) with zoom-responsive widths

### Debug Features
- Fetch request logging in browser console
- Tile loading state tracking
- MapLibre GL error reporting
- Network request inspection

## Styling Guide

For detailed instructions on customizing map styling for Points, Polygons, and Lines, see [server/STYLING_GUIDE.md](server/STYLING_GUIDE.md).

### Quick Examples

**Color counties by state (Polygons):**
```javascript
"fill-color": [
  "match",
  ["get", "STATEFP"],
  "06", "#ff0000",  // California
  "48", "#0000ff",  // Texas
  "#cccccc"         // Others
]
```

**Color landmarks by type (Points):**
```javascript
"circle-color": [
  "match",
  ["get", "MTFCC"],
  "K2540", "#ff7f0e",   // Schools
  "K2543", "#1f77b4",   // Hospitals
  "#aaaaaa"             // Default
]
```

**Color roads by type (Lines):**
```javascript
"line-color": [
  "match",
  ["get", "RTTYP"],
  "M", "#ff0000",  // Major roads
  "S", "#ffa500",  // Secondary
  "#0066cc"        // Local
]
```


## Prerequisites

### Unix Systems
- Ensure you have Python 3.6 or higher installed.
- Install `make` if not already available.

### Non-Unix Systems
- Install Python 3.6 or higher.
- Use a compatible `make` alternative or manually execute the commands in the Makefile.

### Activating the Virtual Environment
- Run the following commands to set up and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Dataset Requirement
- Download the required datasets from the UCR STAR repository.
- Place the datasets in the `datasets/` directory following the structure outlined in this repository.