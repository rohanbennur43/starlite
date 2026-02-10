"""
Download API Service for streaming geospatial features in CSV and GeoJSON formats.
Supports spatial filtering using Minimum Bounding Rectangle (MBR).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any
import json
import csv
import pyarrow.parquet as pq
import geopandas as gpd


@dataclass
class BoundingBox:
    """Represents a spatial bounding box with intersection logic."""
    minx: float
    miny: float
    maxx: float
    maxy: float
    
    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this bounding box intersects with another."""
        return not (
            self.maxx < other.minx or 
            self.minx > other.maxx or 
            self.maxy < other.miny or 
            self.miny > other.maxy
        )
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within this bounding box."""
        return self.minx <= x <= self.maxx and self.miny <= y <= self.maxy
    
    @classmethod
    def from_string(cls, mbr_string: str) -> "BoundingBox":
        """Parse MBR from string format: 'x1,y1,x2,y2'."""
        parts = [float(p) for p in mbr_string.split(',')]
        if len(parts) != 4:
            raise ValueError("MBR must be 4 values: minx,miny,maxx,maxy")
        return cls(minx=parts[0], miny=parts[1], maxx=parts[2], maxy=parts[3])


class TileManager:
    """Manages parquet tile discovery and filtering by MBR."""
    
    def __init__(self, dataset_path: Path):
        """Initialize with path to parquet_tiles directory."""
        self.dataset_path = dataset_path
        self.tiles_dir = dataset_path / "parquet_tiles"
        
    def parse_tile_mbr(self, filename: str) -> Optional[BoundingBox]:
        """
        Parse MBR from tile filename.
        Format: tile_XXXXXX__minx_miny_maxx_maxy.parquet
        """
        try:
            # Remove extension and split by __
            base = filename.replace('.parquet', '')
            parts = base.split('__')
            if len(parts) != 2:
                return None
            
            coords = parts[1].split('_')
            # Filter out empty strings from splitting
            coords = [c for c in coords if c]
            
            if len(coords) < 4:
                return None
            
            # Parse coordinates (handling negative values)
            minx = float(coords[0])
            miny = float(coords[1])
            maxx = float(coords[2])
            maxy = float(coords[3])
            
            return BoundingBox(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        except (ValueError, IndexError):
            return None
    
    def find_intersecting_tiles(self, query_mbr: Optional[BoundingBox]) -> List[Path]:
        """Find all parquet tiles that intersect with the query MBR.
        If query_mbr is None, return all tiles."""
        intersecting_tiles = []
        
        if not self.tiles_dir.exists():
            return intersecting_tiles
        
        # If no MBR specified, return all tiles
        if query_mbr is None:
            return sorted(self.tiles_dir.glob("*.parquet"))
        
        # Filter tiles by MBR intersection
        for tile_file in self.tiles_dir.glob("*.parquet"):
            tile_mbr = self.parse_tile_mbr(tile_file.name)
            if tile_mbr and query_mbr.intersects(tile_mbr):
                intersecting_tiles.append(tile_file)
        
        return sorted(intersecting_tiles)


class FormatHandler(ABC):
    """Abstract base class for feature format handlers."""
    
    def __init__(self, output_mbr: Optional[BoundingBox] = None):
        """Initialize handler with optional output MBR for filtering."""
        self.output_mbr = output_mbr
    
    @abstractmethod
    def initialize(self) -> str:
        """Return initialization content (headers, opening tags, etc.)."""
        pass
    
    @abstractmethod
    def write_feature(self, feature: Dict[str, Any]) -> str:
        """Convert a feature to output format string."""
        pass
    
    @abstractmethod
    def finalize(self) -> str:
        """Return finalization content (closing tags, etc.)."""
        pass
    
    def should_include_feature(self, feature: Dict[str, Any]) -> bool:
        """Filter feature by output MBR if specified."""
        if self.output_mbr is None:
            return True
        
        # Check if geometry exists and is a point
        if 'geometry' not in feature:
            return False
        
        geom = feature['geometry']
        if geom is None:
            return False
        
        # Handle Point geometry
        if geom.get('type') == 'Point':
            coords = geom.get('coordinates', [])
            if len(coords) >= 2:
                x, y = coords[0], coords[1]
                return self.output_mbr.contains_point(x, y)
        
        return True


class CSVHandler(FormatHandler):
    """Handles CSV format output."""
    
    def __init__(self, output_mbr: Optional[BoundingBox] = None):
        super().__init__(output_mbr)
        self.fieldnames = None
        self.writer_initialized = False
    
    def initialize(self) -> str:
        return ""  # Headers written with first row
    
    def write_feature(self, feature: Dict[str, Any]) -> str:
        """Convert feature to CSV row."""
        if not self.should_include_feature(feature):
            return ""
        
        props = feature.get('properties', {})
        
        # Initialize fieldnames from first feature
        if self.fieldnames is None:
            self.fieldnames = list(props.keys())
            self.fieldnames.extend(['geometry_type', 'x', 'y'])
            # Write header
            header = ",".join(self.fieldnames) + "\n"
            return header + self._feature_to_csv(props, feature.get('geometry'))
        
        return self._feature_to_csv(props, feature.get('geometry'))
    
    def _feature_to_csv(self, properties: Dict, geometry: Optional[Dict]) -> str:
        """Convert properties dict to CSV row."""
        row = []
        for field in self.fieldnames:
            if field == 'geometry_type' and geometry:
                row.append(geometry.get('type', ''))
            elif field == 'x' and geometry:
                coords = geometry.get('coordinates', [])
                row.append(str(coords[0]) if coords else '')
            elif field == 'y' and geometry:
                coords = geometry.get('coordinates', [])
                row.append(str(coords[1]) if len(coords) > 1 else '')
            else:
                value = properties.get(field, '')
                # Escape quotes and wrap if needed
                if isinstance(value, str) and (',' in value or '"' in value):
                    value = f'"{value.replace(chr(34), chr(34)+chr(34))}"'
                row.append(str(value))
        
        return ",".join(row) + "\n"
    
    def finalize(self) -> str:
        return ""


class GeoJSONHandler(FormatHandler):
    """Handles GeoJSON format output."""
    
    def __init__(self, output_mbr: Optional[BoundingBox] = None):
        super().__init__(output_mbr)
        self.first_feature = True
    
    def initialize(self) -> str:
        return '{"type":"FeatureCollection","features":['
    
    def write_feature(self, feature: Dict[str, Any]) -> str:
        """Convert feature to GeoJSON format."""
        if not self.should_include_feature(feature):
            return ""
        
        if not self.first_feature:
            return "," + json.dumps(feature)
        else:
            self.first_feature = False
            return json.dumps(feature)
    
    def finalize(self) -> str:
        return "]}"


class FeatureStreamer:
    """Streams features from parquet tiles with spatial filtering."""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.tile_manager = TileManager(dataset_path)
    
    def stream_features(
        self, 
        query_mbr: Optional[BoundingBox],
        format_handler: FormatHandler
    ) -> Iterator[str]:
        """
        Stream features that intersect with query MBR in specified format.
        If query_mbr is None, streams all features.
        Yields strings that can be written to response.
        """
        # Initialize output
        yield format_handler.initialize()
        
        # Find intersecting tiles
        tiles = self.tile_manager.find_intersecting_tiles(query_mbr)
        
        if not tiles:
            yield format_handler.finalize()
            return
        
        # Stream features from each tile
        for tile_path in tiles:
            try:
                # Read parquet file
                table = pq.read_table(str(tile_path))
                gdf = table.to_pandas()
                
                # Convert to GeoJSON-like format
                for _, row in gdf.iterrows():
                    feature = self._row_to_feature(row)
                    output = format_handler.write_feature(feature)
                    if output:
                        yield output
            except Exception as e:
                print(f"Error reading tile {tile_path}: {e}")
                continue
        
        # Finalize output
        yield format_handler.finalize()
    
    @staticmethod
    def _row_to_feature(row) -> Dict[str, Any]:
        """Convert a GeoDataFrame row to GeoJSON feature."""
        properties = {}
        geometry = None
        
        for col, value in row.items():
            if col == 'geometry':
                # Handle geometry column
                if hasattr(value, '__geo_interface__'):
                    geometry = value.__geo_interface__
            else:
                # Regular properties
                if value is not None:
                    properties[col] = value
        
        feature = {
            'type': 'Feature',
            'properties': properties,
            'geometry': geometry
        }
        return feature


class DatasetFeatureService:
    """High-level service for dataset feature downloads."""
    
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
    
    def get_features_stream(
        self, 
        dataset_name: str, 
        format: str,
        mbr_string: Optional[str] = None
    ) -> Iterator[str]:
        """
        Get streaming response for features.
        
        Args:
            dataset_name: Name of dataset (e.g., 'TIGER2018_COUNTY')
            format: Output format ('csv' or 'geojson')
            mbr_string: Optional bounding box string 'minx,miny,maxx,maxy'.
                       If None, returns all features.
        
        Yields:
            String chunks for streaming response
        """
        # Parse query MBR if provided
        query_mbr = None
        if mbr_string:
            try:
                query_mbr = BoundingBox.from_string(mbr_string)
            except ValueError as e:
                raise ValueError(f"Invalid MBR format: {e}")
        
        # Get dataset path
        dataset_path = self.data_root / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
        
        # Create appropriate handler
        format_lower = format.lower()
        if format_lower == 'csv':
            handler = CSVHandler(output_mbr=query_mbr)
        elif format_lower == 'geojson':
            handler = GeoJSONHandler(output_mbr=query_mbr)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Stream features
        streamer = FeatureStreamer(dataset_path)
        return streamer.stream_features(query_mbr, handler)
    
    def get_mime_type(self, format: str) -> str:
        """Get MIME type for format."""
        format_lower = format.lower()
        if format_lower == 'csv':
            return 'text/csv'
        elif format_lower == 'geojson':
            return 'application/geo+json'
        return 'application/octet-stream'
