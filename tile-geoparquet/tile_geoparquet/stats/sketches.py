import math
from collections import Counter
from datasketch import HyperLogLog
from shapely import wkb

TOP_K = 20


class SpaceSavingTopK:
    """
    Simple bounded-frequency tracker.
    Not exact, but good enough for visualization hints.
    """
    def __init__(self, k=TOP_K):
        self.k = k
        self.counter = Counter()

    def update(self, values):
        for v in values:
            self.counter[v] += 1

        # keep only top-k
        if len(self.counter) > self.k * 2:
            self.counter = Counter(dict(self.counter.most_common(self.k)))

    def result(self):
        total_count = sum(self.counter.values())
        top_k = self.counter.most_common(self.k)
        top_k_count = sum(count for _, count in top_k)

        # Include top-k only if they represent at least 80% of the data
        if top_k_count / total_count >= 0.8:
            return [
                {"value": v, "count": c}
                for v, c in top_k
            ]
        return []


class NumericSketch:
    def __init__(self):
        self.count = 0
        self.non_null = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = None
        self.max = None
        self.hll = HyperLogLog(p=12)
        self.topk = SpaceSavingTopK()

    def update(self, values):
        for v in values:
            self.count += 1
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue

            self.non_null += 1

            if self.min is None or v < self.min:
                self.min = v
            if self.max is None or v > self.max:
                self.max = v

            # Welford
            delta = v - self.mean
            self.mean += delta / self.non_null
            delta2 = v - self.mean
            self.M2 += delta * delta2

            self.hll.update(str(v).encode("utf-8"))
            self.topk.update([v])

    def finalize(self):
        stddev = math.sqrt(self.M2 / self.non_null) if self.non_null > 1 else 0.0
        return {
            "non_null_count": self.non_null,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "stddev": stddev,
            "approx_distinct": int(self.hll.count()),
            "top_k": self.topk.result(),
        }


class CategoricalSketch:
    def __init__(self):
        self.non_null = 0
        self.hll = HyperLogLog(p=12)
        self.topk = SpaceSavingTopK()

    def update(self, values):
        for v in values:
            if v is None:
                continue
            self.non_null += 1
            s = str(v)
            self.hll.update(s.encode("utf-8"))
            self.topk.update([s])

    def finalize(self):
        return {
            "non_null_count": self.non_null,
            "approx_distinct": int(self.hll.count()),
            "top_k": self.topk.result(),
        }


class TextSketch(CategoricalSketch):
    def __init__(self):
        super().__init__()
        self.total_length = 0
        self.min_length = None
        self.max_length = None

    def update(self, values):
        for v in values:
            if v is None:
                continue
            s = str(v)
            l = len(s)

            self.non_null += 1
            self.total_length += l

            if self.min_length is None or l < self.min_length:
                self.min_length = l
            if self.max_length is None or l > self.max_length:
                self.max_length = l

            self.hll.update(s.encode("utf-8"))
            self.topk.update([s])

    def finalize(self):
        avg_len = (
            self.total_length / self.non_null
            if self.non_null > 0 else 0
        )
        return {
            "non_null_count": self.non_null,
            "approx_distinct": int(self.hll.count()),
            "avg_length": avg_len,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "top_k": self.topk.result(),
        }


class GeometrySketch:
    def __init__(self):
        self.minx = self.miny = None
        self.maxx = self.maxy = None
        self.geom_types = Counter()
        self.total_points = 0

from shapely import wkb

class GeometrySketch:
    def __init__(self):
        self.minx = self.miny = None
        self.maxx = self.maxy = None
        self.geom_types = Counter()
        self.total_points = 0

    def update(self, geoms):
        for g in geoms:
            if g is None:
                continue

            # g is WKB bytes
            try:
                geom = wkb.loads(g)
            except Exception:
                continue

            if geom.is_empty:
                continue

            self.geom_types[geom.geom_type] += 1

            minx, miny, maxx, maxy = geom.bounds
            if self.minx is None:
                self.minx, self.miny, self.maxx, self.maxy = minx, miny, maxx, maxy
            else:
                self.minx = min(self.minx, minx)
                self.miny = min(self.miny, miny)
                self.maxx = max(self.maxx, maxx)
                self.maxy = max(self.maxy, maxy)

            try:
                self.total_points += len(geom.coords)
            except Exception:
                pass


    def finalize(self):
        return {
            "mbr": [self.minx, self.miny, self.maxx, self.maxy],
            "geom_types": dict(self.geom_types),
            "total_points": self.total_points,
        }
