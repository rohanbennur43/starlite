# Starlite API Documentation

## Base URL

`http://127.0.0.1:5000`

------------------------------------------------------------------------

## 1. Overview

The Vector Dataset Service provides RESTful APIs for:

-   Serving Mapbox Vector Tiles
-   Dataset discovery and search
-   Dataset metadata and statistics
-   Feature download with spatial filtering
-   Sample feature retrieval
-   HTML visualization

Spatial filtering is supported via: - Minimum Bounding Rectangle (MBR) -
Custom GeoJSON geometry (POST requests)

------------------------------------------------------------------------

## 2. Tile Service

### GET `/{dataset}/{z}/{x}/{y}.mvt`

Returns a Mapbox Vector Tile for the specified dataset and tile
coordinates.

**Path Parameters**

  Parameter   Type      Description
  ----------- --------- -------------------
  dataset     string    Dataset name
  z           integer   Zoom level
  x           integer   Tile X coordinate
  y           integer   Tile Y coordinate

**Response** - Content-Type: `application/vnd.mapbox-vector-tile` -
Binary tile data

------------------------------------------------------------------------

## 3. Dataset Discovery

### GET `/api/datasets`

Returns available datasets.

**Example Response**

``` json
{
  "datasets": ["TIGER2018_COUNTY", "TIGER2018_POINTLM"]
}
```

------------------------------------------------------------------------

### GET `/datasets.json?q=<search_term>`

Search datasets by name or ID.

**Query Parameters**

  Parameter   Description
  ----------- --------------------------------
  q           Case-insensitive search string

**Example Response**

``` json
{
  "datasets": [
    {
      "id": "TIGER2018_COUNTY",
      "name": "Tiger2018 County",
      "size": 12345678
    }
  ]
}
```

------------------------------------------------------------------------

## 4. Dataset Metadata & Statistics

### GET `/datasets/{dataset}.json`

Returns metadata including dataset size and file count.

**Example Response**

``` json
{
  "id": "TIGER2018_COUNTY",
  "name": "Tiger2018 County",
  "size": 12345678,
  "file_count": 42
}
```

------------------------------------------------------------------------

### GET `/api/datasets/{dataset}/stats`

Returns precomputed attribute statistics.

------------------------------------------------------------------------

## 5. Feature Download Service

### GET `/datasets/{dataset}/features.{format}`

Downloads dataset features in GeoJSON or CSV format.

Supported formats: - `geojson` - `csv`

Optional query parameter:

  Parameter   Format                Description
  ----------- --------------------- ----------------------
  mbr         minx,miny,maxx,maxy   Spatial bounding box

**Example**

    GET /datasets/TIGER2018_COUNTY/features.geojson?mbr=-120,30,-100,40

Response: - Streamed file download - Content-Disposition header set

------------------------------------------------------------------------

### POST `/datasets/{dataset}/features.{format}`

Accepts a GeoJSON geometry payload for spatial filtering.

**Example Payload**

``` json
{
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[...]]]
  }
}
```

------------------------------------------------------------------------

## 6. Sample Feature Retrieval

### GET `/datasets/{dataset}/features/sample.json?mbr=minx,miny,maxx,maxy`

Returns first matching feature without geometry.

------------------------------------------------------------------------

### GET `/datasets/{dataset}/features/sample.geojson?mbr=minx,miny,maxx,maxy`

Returns first matching feature including geometry.

------------------------------------------------------------------------

## 7. Dataset Visualization

### GET `/datasets/{dataset}.html`

Returns a minimal HTML visualization page for the dataset.

------------------------------------------------------------------------

## 8. Error Handling

The API uses standard HTTP status codes:

  Code   Meaning
  ------ ----------------
  200    Success
  400    Bad Request
  404    Not Found
  500    Internal Error

Errors are returned in JSON format:

``` json
{
  "error": "Dataset not found"
}
```

------------------------------------------------------------------------

## 9. Architectural Notes

-   In-memory tiler cache for performance
-   Streamed responses for large datasets
-   Spatial filtering via MBR and GeoJSON
-   Threaded Flask server for concurrent requests
-   Precomputed dataset statistics

------------------------------------------------------------------------

End of Documentation
