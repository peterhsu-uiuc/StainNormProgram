from stainnormprogram.slide import SlideImage, SlideImageCollection
import stainnormprogram.norm.background as background

import statistics

import numpy as np

class Contrast:

    def __init__(self, slide, identifier=None, tile_size=512, background_method="hsv_threshold", accept_perc=10, max_pixels=50_000_000, tile_locs=None):
        self.slide = slide
        self.identifier = identifier if identifier is not None else self.slide.identifier
        self.background_method = background_method
        self.accept_perc = accept_perc

        # Tile size
        self.slide.set_tile_size(tile_size)
        
        self.num_pixels = 0
        self.num_tiles = 0
        self.max_pixels = max_pixels

        self.rms_vals = []
        self.tile_locs = [] if tile_locs is None else tile_locs

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["slide"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.slide = None

    def compute_tile(self, tile, fg_filtering=True):

        if self.background_method == "hsv_threshold":
            fg_mask = background.hsv_threshold_foreground(tile).flatten()
        else:
            raise Exception

        n = np.sum(fg_mask)
        if fg_filtering and n / fg_mask.size * 100 < self.accept_perc:
            return False

        lum = tile.luv()[:,:,0].flatten()
        # lum = lum[fg_mask]
        lum = lum / 100. # Normalize to [0., 1.]
        
        # RMS contrast
        rms = np.std(lum)

        self.num_tiles = self.num_tiles + 1
        self.num_pixels = self.num_pixels + n

        self.rms_vals.append(rms)

        return True

    def compute(self):

        print(f"Computing Contrast for {self.identifier}...")
    
        if len(self.tile_locs) == 0:
            print("Using random iterator...")
            self.slide.set_mode("random")
        else:
            # Use corresponding tile locations
            print("Using custom iterator...")
            self.slide.set_mode("custom")
            self.slide.set_custom_iter_locs(self.tile_locs)

        for tile, metadata in self.slide:
            success = self.compute_tile(tile, fg_filtering=(self.slide.mode == "random"))

            if self.slide.mode == "random" and success:
                self.tile_locs.append(metadata)
                if self.num_pixels > self.max_pixels:
                    break

        print("Finished computing Contrast")
        print(f"Tiles used = {self.num_tiles}")
        print(f"Foreground pixels used = {self.num_pixels}")


class ContrastChange:

    def __init__(self, reference, query):
        self.reference = reference
        self.query = query

        self.changes = []

        self.change_mean = None
        self.change_std = None

    def get_val(self, val_type=None):
        return (self.change_mean, self.change_std)

    def compute(self):
        
        num_vals = len(self.reference.rms_vals)

        if num_vals != len(self.query.rms_vals):
            raise Exception(f"Number of RMS values are not equal for reference and query. Reference = {num_vals}, Query = {len(self.query.rms_vals)}")

        for i in range(num_vals):

            change = self.query.rms_vals[i] - self.reference.rms_vals[i]
            self.changes.append(change)

        self.change_mean = statistics.mean(self.changes)
        self.change_std = statistics.stdev(self.changes)


        
