from stainnormprogram.slide import SlideImage, SlideImageCollection
import stainnormprogram.norm.background as background

from bitarray import bitarray
import numpy as np

class GamutArray:

    def __init__(self):
        self.array = bitarray(256 * 256 * 256)

    @staticmethod
    def colors_to_indices(rgb):
        rgb = rgb.astype(np.uint32)
        indices = rgb[:,0] * 256 * 256 + rgb[:,1] * 256 + rgb[:,2]
        return indices # (N,) np.uint32

    @staticmethod
    def indices_to_colors(indices):
        colors = []
        for idx in indices:
            x = idx % (256*256)
            b = x % 256
            g = x // 256
            r = (idx - b - g * 256) // (256 * 256)
            color = (round(r), round(g), round(b))
            colors.append(color)
        
        colors = np.array(colors)
        assert colors.min() >= 0 and colors.max() <= 255, f"min = {colors.min()}, max = {colors.max()}"
        colors = colors.astype(np.uint8)
        return colors # (N,3) np.ndarray uint8

    def bitwise_or(self, gamut_array):
        right_array = gamut_array.array
        left_array = self.array

        self.array = left_array | right_array

    def add_colors(self, rgb): # (N,3) np.ndarray uint8
        indices = self.colors_to_indices(rgb)
        for idx in indices:
            self.array[idx] = 1

    def has_colors(self, rgb): # (N,3) np.ndarray uint8
        indices = self.colors_to_indices(rgb)
        flags = []
        for position in range(indices.size):
            flags[position] = bool(self.array[indices[position]])
        flags = np.array(flags)
        return flags # (N,) np.ndarray bool

class Gamut:

    def __init__(self, slide, identifier=None, tile_size=32, background_method="hsv_threshold", accept_perc=10, max_pixels=1_000_000):
        
        self.slide = slide
        self.identifier = identifier if identifier is not None else self.slide.identifier
        self.background_method = background_method
        self.accept_perc = accept_perc
        
        # Tile size
        if isinstance(self.slide, (SlideImage, SlideImageCollection)):
            self.slide.set_tile_size(tile_size)
        else:
            raise Exception

        self.num_pixels = 0
        self.num_tiles = 0
        self.max_pixels = max_pixels
        
        self.gamut_array = GamutArray()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["slide"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.slide = None

    def compute_tile(self, tile):

        if self.background_method == "hsv_threshold":
            fg_mask = background.hsv_threshold_foreground(tile).flatten()
        else:
            raise Exception

        n = np.sum(fg_mask)
        if n / fg_mask.size * 100 < self.accept_perc:
            return

        tile_rgb_flat = tile.rgb(as_uint8=True).reshape((-1,3))
        self.gamut_array.add_colors(tile_rgb_flat)

        self.num_tiles = self.num_tiles + 1
        self.num_pixels = self.num_pixels + n

    def compute(self):
    
        print(f"Computing Gamut for {self.identifier}...")
    
        self.slide.set_mode("random")
        for tile, _ in self.slide:
            self.compute_tile(tile)
            if self.num_pixels > self.max_pixels:
                break

        print("Finished computing Gamut")

class GamutComparison:

    def __init__(self, reference, query):
        self.reference = reference
        self.query = query

        # Out-of-gamut count and colors
        self.oog_count = None
        self.oog_colors = None # np.ndarray (N,3) uint8

        self.ig_count = None

    def get_oog_colors(self, oog_mask):

        indices = np.where(oog_mask)[0]
        colors = GamutArray.indices_to_colors(indices)

        return colors

    def get_val(self, val_type):
        if val_type == "oog_count":
            return self.oog_count
        elif val_type == "ig_count":
            return self.ig_count
        else:
            raise Exception

    def compute(self, get_colors=False):

        ref_array = self.reference.gamut_array.array
        ref_array = np.array(ref_array.tolist())
        
        q_array = self.query.gamut_array.array
        q_array = np.array(q_array.tolist())

        # An out-of-gamut color is when query is 1 but ref is 0
        oog_mask = (q_array == 1) & (ref_array == 0)
        self.oog_count = np.sum(oog_mask)

        ig_mask = (q_array == 1) & (ref_array == 1)
        self.ig_count  = np.sum(ig_mask)

        if get_colors:
            self.oog_colors = self.get_oog_colors(oog_mask)

class ReferenceGamut:

    def __init__(self, ref_gamuts, identifier=None):
        self.ref_gamuts = ref_gamuts
        if identifier is not None:
            self.identifier = identifier
        elif isinstance(ref_gamuts, Gamut):
            self.identifier = ref_gamuts.identifier
        elif isinstance(ref_gamuts, list) and len(ref_gamuts) == 1:
            self.identifier = ref_gamuts[0].identifier
        else:
            raise Exception

        if isinstance(self.ref_gamuts, Gamut):
            self.ref_gamuts = [ref_gamuts]

        self.gamut_array = GamutArray()

    def compute(self):

        for g in self.ref_gamuts:
            self.gamut_array.bitwise_or(g.gamut_array)
