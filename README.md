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
Starts a small local server to view tiles.

- Serves the `datasets/` directory.
- Opens `server/view_mvt.html` automatically.

### `make clean`
Removes all generated tiles, MVTs, and logs.

## Example
make all INPUT=../extras/original_datasets/OSM2015_33.parquet

