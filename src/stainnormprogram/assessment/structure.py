from stainnormprogram.slide import SlideImage, SlideImageCollection
import stainnormprogram.norm.background as background

import statistics

import numpy as np

class StructurePreservation:

    def __init__(self, reference, query, tile_size=256, background_method="hsv_threshold", accept_perc=10, max_pixels=10_000_000):
        self.reference = reference
        self.query = query
        self.background_method = background_method
        self.accept_perc = accept_perc

        # Tile size
        self.reference.set_tile_size(tile_size)
        self.query.set_tile_size(tile_size)
        
        self.num_pixels = 0
        self.num_tiles = 0
        self.max_pixels = max_pixels

        self.diffs = []
        self.diff_mean = None
        self.diff_std = None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["reference"]
        del state["query"]
        return state

    def __setstate__(self):
        self.__dict__.update(state)
        self.reference = None
        self.query = None

    def compute_tile(self, ref_tile, q_tile):

        if self.background_method == "hsv_threshold":
            fg_mask = background.hsv_threshold_foreground(ref_tile).flatten()
        else:
            raise Exception

        n = np.sum(fg_mask)
        if n / fg_mask.size * 100 < self.accept_perc:
            return None

        ref_canny = ref_tile.canny()
        q_canny = q_tile.canny()

        diff_mask = ref_canny ^ q_canny
        diff_count = np.sum(diff_mask).item()

        self.num_tiles = self.num_tiles + 1
        self.num_pixels = self.num_pixels + n
        
        self.diffs.append(diff_count)

    def get_val(self, val_type=None):
        return (self.diff_mean, self.diff_std)

    def compute(self):
        
        print(f"Computing Structure between {self.reference.identifier} and {self.query.identifier}...")
    
        self.reference.set_mode("random")
        for ref_tile, metadata in self.reference:
            q_tile = self.query.get_tile_using_dict(metadata)
            self.compute_tile(ref_tile, q_tile)
            if self.num_pixels > self.max_pixels:
                break

        self.diff_mean = statistics.mean(self.diffs)
        self.diff_std = statistics.stdev(self.diffs)

        print("Finished computing Structure")
        print(f"Tiles used = {self.num_tiles}")
        print(f"Foreground pixels used = {self.num_pixels}")