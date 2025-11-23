# Tile Geoparquet Pipeline

This project converts any GeoJSON or GeoParquet file into tiled GeoParquet, then automatically generates MVT vector tiles and histogram grids. It also includes a small development server and web viewer.

## Requirements
Install dependencies:
pip install -r requirements.txt

## Directory Structure
project/
  Makefile
  tile_geoparquet/
  mvt/
  server/
  datasets/          # GeoParquet tiles (auto created)
  mvt_out/           # MVT tiles (auto created)
  requirements.txt

## Full Pipeline
You must run BOTH steps:  
1) GeoParquet tiling  
2) MVT tile generation  

### Step 1. Tiling a Dataset
Run:
make tiles INPUT=path/to/your/data.parquet

Example:
make tiles INPUT=../extras/original_datasets/highways/roads.parquet

This will:
1. Extract dataset name (roads.parquet → roads)
2. Generate GeoParquet tiles in datasets/roads/
3. Generate histogram grids for rendering and analysis
4. Store logs in logs_roads.txt

### Step 2. Generate MVT Tiles (required)
After step 1, run:
make mvt INPUT=path/to/your/data.parquet

This reads tiles from:
datasets/<dataset_name>/
And writes vector tiles to:
mvt_out/<dataset_name>/

These MVT tiles are required for viewing data in the frontend.

## Development Server
To preview tiles:
make server

This:
- Starts the Flask server
- Opens server/view_mvt.html automatically (if supported)

## Cleaning Outputs
make clean

Removes:
- datasets/*
- mvt_out/*
- logs_*.txt

## Summary
✔ Required: Generate GeoParquet tiles  
✔ Required: Generate MVT tiles  
✔ Automatic histogram grids  
✔ Local tile server with viewer  
✔ Simple Makefile workflow
