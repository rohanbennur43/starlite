import numpy as np
import logging

logger = logging.getLogger("bucket_mvt")


class HistogramLoader:
    def __init__(self, hist_path):
        self.hist_path = hist_path
        self.prefix = None

    def load(self):
        logger.info(f"Loading histogram from {self.hist_path}")
        hist = np.load(self.hist_path)
        self.prefix = hist.cumsum(axis=0).cumsum(axis=1)
        return self.prefix
