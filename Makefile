# ---------------------------------------------------------
# Makefile for automatic tiling and MVT generation
# Usage:
#   make tiles INPUT=path/to/data.parquet
#   make mvt INPUT=path/to/data.parquet
#   make all INPUT=path/to/data.parquet
#   make server
# ---------------------------------------------------------

PYTHON = python3

# ----------------------------
# INPUT based targets
# ----------------------------

INPUT ?= none
THRESHOLD ?= 0.5

ifneq ($(MAKECMDGOALS),server)
ifneq ($(MAKECMDGOALS),clean)
ifeq ($(INPUT),none)
$(error You must supply INPUT=path/to/file.parquet)
endif
endif
endif

# Extract dataset name from the input file
DATA := $(basename $(notdir $(INPUT)))

TILES = datasets/$(DATA)
MVT_DIR = mvt_out/$(DATA)
LOGFILE = logs_$(DATA).txt

# ----------------------------
# Main targets
# ----------------------------

all: tiles mvt

tiles:
	$(PYTHON) -m tile_geoparquet.cli \
		--input $(INPUT) \
		--outdir $(TILES) \
		--num-tiles 40 \
		--sort-mode zorder \
		--sample-cap 10000 > $(LOGFILE) 2>&1
	
# 	$(PYTHON) mvt/mvt.py \
# 		--dir $(TILES) \
# 		--threshold $(THRESHOLD)

mvt:
	$(PYTHON) mvt/mvt.py \
		--dir $(TILES) \
		--threshold $(THRESHOLD) > $(LOGFILE) 2>&1 \
		|| (echo "MVT generation failed. Check $(LOGFILE) for details."; exit 1)

# ----------------------------
# Server target (no INPUT)
# ----------------------------

server:
	python3 server/server.py --root datasets &
	sleep 1
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open server/view_mvt.html; \
	elif command -v open >/dev/null 2>&1; then \
		open server/view_mvt.html; \
	else \
		echo "Please open server/view_mvt.html manually"; \
	fi

# ----------------------------
# Clean
# ----------------------------

clean:
	rm -rf datasets/*
	rm -rf mvt_out/*
	rm -f logs_*.txt

.PHONY: all tiles mvt server clean
