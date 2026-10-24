from stainnormprogram.slide import SlideImage, SlideImageCollection
import stainnormprogram.norm.background as background

import itertools
import statistics
from typing import Union

import fast_colorthief
import colour
import numpy as np
from sklearn.cluster import KMeans

class Palette:
    """Generate a color palette using fast_colorthief (Modified Median Cut Quantization)
    """

    def __init__(self, slide, identifier=None, tile_size=32, background_method="hsv_threshold", accept_perc=10, max_pixels=1_000_000, color_count=10):

        self.slide = slide
        self.identifier = identifier if identifier is not None else self.slide.identifier
        self.background_method = background_method
        self.accept_perc = accept_perc
        self.color_count = color_count
        
        # Tile size
        self.slide.set_tile_size(tile_size)
        
        self.num_pixels = 0
        self.num_tiles = 0
        self.max_pixels = max_pixels

        self.palette = None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["slide"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.slide = None

    def extract_pixels(self, tile):
        
        if self.background_method == "hsv_threshold":
            fg_mask = background.hsv_threshold_foreground(tile).flatten()
        else:
            raise Exception

        n = np.sum(fg_mask)
        if n / fg_mask.size * 100 < self.accept_perc:
            return None

        if self.num_tiles % 100 == 0:
            print(f"Processed {self.num_tiles} tiles so far...")

        rgb_tile = tile.rgb(as_uint8=True).reshape((-1,3))
        rgb = rgb_tile[fg_mask, :]

        return rgb # (N,3)

    def compute(self):

        print(f"Computing Palette for {self.identifier}...")

        samples = []

        self.slide.set_mode("random")
        for tile, _ in self.slide:
            pixels = self.extract_pixels(tile)
            if pixels is not None:
                samples.append(pixels)
                self.num_tiles += 1
                self.num_pixels += pixels.shape[0]
            if self.num_pixels > self.max_pixels:
                break
                
        samples = np.concatenate(samples, axis=0) # (N,3)

        # Make into a square image
        img_dim = int(samples.shape[0] ** 0.5)
        samples = samples[:img_dim*img_dim, :]
        samples = samples.reshape((img_dim, img_dim, 3))
        alpha = np.ones((img_dim, img_dim, 1)) * 255
        samples = np.concatenate([samples, alpha], axis=-1)
        samples = samples.astype(np.uint8)

        print(f"Using {samples.shape[0]*samples.shape[1]} pixels to compute palette")

        self.palette = fast_colorthief.get_palette(samples, color_count=self.color_count+1, quality=1)
        self.palette = np.array(self.palette)

        assert self.palette.shape == (self.color_count, 3), self.palette.shape
        
        print("Finished computing Palette")

    def display(self):
        return colour.plotting.plot_multi_colour_swatches(self.palette / 255.)

    def get_palette(self, normalized=False):
        if normalized:
            return self.palette / 255
        return self.palette

class ReferencePalette:

    def __init__(self, palettes, identifier=None, color_count=10):
        self.palettes = palettes
        self.color_count = color_count
        if identifier is not None:
            self.identifier = identifier
        elif isinstance(palettes, Palette):
            self.identifier = palettes.identifier
        elif isinstance(palettes, list) and len(palettes) == 1:
            self.identifier = palettes[0].identifier
        else:
            raise Exception

        self.palette = None

    def display(self):
        return colour.plotting.plot_multi_colour_swatches(self.palette / 255.)

    def get_palette(self, normalized=False):
        if normalized:
            return self.palette / 255
        return self.palette

    def compute(self):

        if isinstance(self.palettes, Palette):
            self.palette = self.palettes.get_palette()
            return

        if not isinstance(self.palettes, list):
            raise Exception
            
        # Accumulate all palette colors into array
        all_colors = []
        for pal in self.palettes:
            all_colors.append(pal.get_palette(normalized=True))
        all_colors = np.concatenate(all_colors, axis=0) # (N,3)

        # Kmeans
        km = KMeans(n_clusters=self.color_count)
        km.fit(all_colors)

        centers = km.cluster_centers_ # (color_count,3)
        centers = np.clip(centers * 255, 0, 255).astype(np.uint8)

        self.palette = centers
            
class PaletteEvaluation:

    def __init__(self, palette: Union[np.ndarray, Palette, ReferencePalette], stain="HE"):
        """Create a Palette evaluation.

        Args:
            palette: Array of RGB values (uint8) representing a color palette; shape should be (N,3). Or a Palette/ReferencePalette.
        """

        if isinstance(palette, np.ndarray):
            self.palette_sRGB = palette / 255
        elif isinstance(palette, (Palette, ReferencePalette)):
            self.palette_sRGB = palette.get_palette(normalized=True)
        else:
            raise Exception
        self.palette_Lab = self.sRGB_to_Lab(self.palette_sRGB)
        self.reference_colors, self.reference_colors_names = self.get_reference_colors()
        self.stain = stain

    def get_reference_colors(self):
        reference_colors = np.array([
            [255, 0, 0], # red
            [255, 192, 203], # pink
            [128, 0, 128], # purple
            [0, 0, 255], # blue
            [255, 125, 0], # orange
            [0, 255, 255], # aqua
            [0, 0, 0] # black
        ]) / 255.
        reference_colors_names = [
            "red",
            "pink",
            "purple",
            "blue",
            "orange",
            "aqua",
            "black"
        ]
        return self.sRGB_to_Lab(reference_colors), reference_colors_names
        
    def sRGB_to_Lab(self, sRGB):
        return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(sRGB))

    def palette_diversity(self):
        """Compute palette diversity. Calculates the delta E between all pairs of colors.
        """
        delta_Es = []
        for c1, c2 in itertools.combinations(self.palette_Lab, 2):

            dE = colour.delta_E(c1, c2, method="CIE 2000")
            delta_Es.append(dE)

        self.palette_diversity = statistics.mean(delta_Es)
        
    def palette_he(self):
        """Compute palette "HE-ness."
        """
        
        # Sort palette into purples, pinks, and others
        palette_colors_names = []
        for i in range(self.palette_Lab.shape[0]):
            deltas_with_references = colour.delta_E(self.reference_colors, self.palette_Lab[i], method="CIE 2000")
            color_name = self.reference_colors_names[np.argmin(deltas_with_references)]
            palette_colors_names.append(color_name)

        pinks_idx = []
        purples_idx = []
        others_idx = []
        for i, color_name in enumerate(palette_colors_names):
            if color_name in ["purple", "blue"]:
                purples_idx.append(i)
            elif color_name in ["pink", "red"]:
                pinks_idx.append(i)
            else:
                others_idx.append(i)

        dEs = []
        for pink_idx, purple_idx in itertools.product(pinks_idx, purples_idx):
            dE = colour.delta_E(self.palette_Lab[pink_idx], self.palette_Lab[purple_idx], method="CIE 2000")
            dEs.append(dE)

        if len(dEs) == 0:
            pink_purple_separation = 0
        else:
            pink_purple_separation = statistics.mean(dEs)

        dEs = []
        for idx1, idx2 in itertools.combinations(pinks_idx, 2):
            dE = colour.delta_E(self.palette_Lab[idx1], self.palette_Lab[idx2], method="CIE 2000")
            dEs.append(dE)

        if len(dEs) == 0:
            pink_diversity = 0
        else:
            pink_diversity = statistics.mean(dEs)

        dEs = []
        for idx1, idx2 in itertools.combinations(purples_idx, 2):
            dE = colour.delta_E(self.palette_Lab[idx1], self.palette_Lab[idx2], method="CIE 2000")
            dEs.append(dE)

        if len(dEs) == 0:
            purple_diversity = 0
        else:
            purple_diversity = statistics.mean(dEs)

        self.pink_purple_separation, self.pink_diversity, self.purple_diversity, self.pinks_idx, self.purples_idx, self.others_idx, self.palette_colors_names = pink_purple_separation, pink_diversity, purple_diversity, pinks_idx, purples_idx, others_idx, palette_colors_names

    def compute(self):

        self.palette_diversity()
        self.palette_he()

class PaletteComparison:

    def __init__(self, reference: PaletteEvaluation, query: PaletteEvaluation, method: str = "pairwise"):
        """Construct a PaletteComparison.

        Args:
            reference: Reference PaletteEvaluation.
            query: Query PaletteEvaluation.
            method: Method of palette comparison.
        """
        self.query = query
        self.reference = reference
        self.method = method

        if len(self.query.palette_Lab) != len(self.reference.palette_Lab):
            raise Exception

        self.pairwise_score = None

        # HE
        self.pink_dE = None
        self.purple_dE = None

    def pairwise(self):

        num_colors = len(self.query.palette_Lab)
        prods = list(itertools.product(list(range(num_colors)), list(range(num_colors))))

        min_dEs = []
        for i in range(0, num_colors * num_colors, num_colors):
            min_dE = 1000
            for q_idx, r_idx in prods[i:i+num_colors]:
                dE = colour.delta_E(self.query.palette_Lab[q_idx], self.reference.palette_Lab[r_idx], method="CIE 2000")
                if dE < min_dE:
                    min_dE = dE
            min_dEs.append(min_dE)

        self.pairwise_score = statistics.mean(min_dEs)

    def get_val(self, val_type):
        if val_type == "compare":
            return self.pairwise_score
        elif val_type == "Hdiv":
            return self.query.purple_diversity
        elif val_type == "Ediv":
            return self.query.pink_diversity
        elif val_type == "HEsep":
            return self.query.pink_purple_separation
        else:
            raise Exception

    def dominant_he(self):

        purple_0_idx = self.query.purples_idx[0]
        purple_1_idx = self.reference.purples_idx[1]

        pink_0_idx = self.query.pinks_idx[0]
        pink_1_idx = self.reference.pinks_idx[1]

        self.pink_dE = colour.delta_E(self.query.palette_Lab[pink_0_idx], self.reference.palette_Lab[pink_1_idx])
        self.purple_dE = colour.delta_E(self.query.palette_Lab[purple_0_idx], self.reference.palette_Lab[purple_1_idx])

    def compute(self):

        if self.method == "pairwise":
            self.pairwise()
        elif self.method == "dominant_he":
            self.dominant_he()
        else:
            raise Exception