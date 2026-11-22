from stainnormprogram.slide import SlideImage, SlideImageCollection
import stainnormprogram.norm.background as background

import itertools
import statistics
from typing import Union
import os
import json

import fast_colorthief
import colour
import numpy as np
from sklearn.cluster import KMeans
import matplotlib

class Palette:

    def __init__(
        self,
        slide: SlideImage | SlideImageCollection,
        identifier: str | None = None,
        tile_size: int = 32,
        background_method: str = "hsv_threshold",
        accept_perc: int = 10,
        max_pixels: int = 1_000_000,
        color_count: int = 10
        ):
        """Generate a color palette using fast_colorthief (Modified Median Cut Quantization).

        Options for ``background_method``:
            * "hsv_threshold": Threshold using saturation and value channels in HSV color space.

        Args:
            slide: The slide to compute the palette of.
            identifier: An ID for this Palette. If None, the identifier of the slide will be used.
            tile_size: Tile size to use.
            background_method: Method for identifying the background/foreground.
            accept_perc: Tiles with less than accept_perc percent of foreground are not used in the computation.
            max_pixels: If the number of pixels used goes over max_pixels, stop the computation.
            color_count: Number of colors for the palette.
        """

        self.slide = slide
        self.identifier = identifier if identifier is not None else self.slide.identifier
        self.background_method = background_method
        self.accept_perc = accept_perc
        self.color_count = color_count
        
        # Tile size
        if self.slide is not None:
            self.slide.set_tile_size(tile_size)
        
        self.num_pixels = 0
        self.num_tiles = 0
        self.max_pixels = max_pixels
        self.tile_size = tile_size

        self.palette = None

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        json_output_path = os.path.join(output_dir, "palette.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "identifier": self.identifier,
                "background_method": self.background_method,
                "accept_perc": self.accept_perc,
                "color_count": self.color_count,
                "tile_size": self.tile_size,
                "num_pixels": self.num_pixels,
                "num_tiles": self.num_tiles,
                "max_pixels": self.max_pixels,
            }, file)

        palette_output_path = os.path.join(output_dir, "palette.npz")
        np.savez(palette_output_path, palette=self.palette, allow_pickle=False)

    @classmethod
    def load(cls, input_dir: str) -> Palette:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A Palette.
        """

        json_input_path = os.path.join(input_dir, "palette.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        pl = cls(None, identifier=json_data["identifier"], tile_size=json_data["tile_size"], background_method=json_data["background_method"], accept_perc=json_data["accept_perc"], max_pixels=json_data["max_pixels"], color_count=json_data["color_count"])
        
        palette_input_path = os.path.join(input_dir, "palette.npz")
        with np.load(palette_input_path, allow_pickle=False) as data:
            pl.palette = data["palette"]

        return pl

    def extract_pixels(self, tile: Tile):
        """Extract foreground pixels from a tile for palette computation.

        Args:
            tile: The tile to extract pixels from.

        Returns:
            A (N,3) RGB np.uint8 array containing foreground pixels, or None if the tile was rejected for having less than accept_perc percent of foreground.
        """
        
        if self.background_method == "hsv_threshold":
            fg_mask = background.hsv_threshold_foreground(tile).flatten()
        else:
            raise Exception(f"{self.background_method} is not a valid background method.")

        n = np.sum(fg_mask).item()
        if n / fg_mask.size * 100 < self.accept_perc:
            return None

        if self.num_tiles % 100 == 0:
            print(f"Processed {self.num_tiles} tiles so far...")

        rgb_tile = tile.rgb(as_uint8=True).reshape((-1,3))
        rgb = rgb_tile[fg_mask, :]

        return rgb # (N,3)

    def compute(self):
        """Compute the Palette.
        """

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

    def display(self) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Display the palette.

        Returns:
            A palette matplotlib figure.
        """
        return colour.plotting.plot_multi_colour_swatches(self.palette / 255.)

    def get_palette(self, normalized: bool = False) -> np.ndarray:
        """Get the palette as a np.ndarray.

        Args:
            normalized: If True, will divide by 255.

        Returns:
            A np.ndarray (np.uint8 RGB) representing the palette with shape (color_count, 3).
        """
        if normalized:
            return self.palette / 255
        return self.palette

class ReferencePalette:

    def __init__(self, palettes: Palette | list[Palette], identifier: str | None = None, color_count: int = 10):
        """A palette for the reference (which may represent multiple reference images).

        Args:
            palettes: One or more Palettes from the reference image(s).
            identifier: An ID for this ReferencePalette. If None, only one Palette must be in palettes, and the ID from this Palette will be used.
            color_count: Number of colors for the ReferencePalette.
        """
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

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        palette_identifiers = []
        for p in self.palettes:
            palette_identifiers.append(p.identifier)

        json_output_path = os.path.join(output_dir, "reference_palette.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "identifier": self.identifier,
                "palette_identifiers": palette_identifiers,
                "color_count": self.color_count,
            }, file)

        palette_output_path = os.path.join(output_dir, "reference_palette.npz")
        np.savez(palette_output_path, palette=self.palette, allow_pickle=False)

    @classmethod
    def load(cls, input_dir: str) -> ReferencePalette:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A ReferencePalette.
        """

        json_input_path = os.path.join(input_dir, "reference_palette.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        rp = cls(json_data["palette_identifiers"], identifier=json_data["identifiers"], color_count=json_data["color_count"])
        
        rp_input_path = os.path.join(input_dir, "reference_palette.npz")
        with np.load(rp_input_path, allow_pickle=False) as data:
            rp.palette = data["palette"]

        return rp

    def display(self) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Display the palette.

        Returns:
            A palette matplotlib figure.
        """
        return colour.plotting.plot_multi_colour_swatches(self.palette / 255.)

    def get_palette(self, normalized: bool = False) -> np.ndarray:
        """Get the palette as a np.ndarray.

        Args:
            normalized: If True, will divide by 255.

        Returns:
            A np.ndarray (np.uint8 RGB) representing the palette with shape (color_count, 3).
        """
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

    def __init__(self, palette: np.ndarray | Palette | ReferencePalette, stain: str = "HE", identifier: str | None = None):
        """Evaluate a single palette.

        This evaluation calculates the following metrics:
            * Palette diversity (attribute: ``diversity``):  Calculates the delta E between all pairs of colors. This is independent of the stain type.
            * HE metrics
                * H&E separability (HEsep; attribute ``pink_purple_separation``): How different the pinks are from the purples.
                * Eosin diversity (Ediv; attribute ``pink_diversity``): How diverse the pink colors are.
                * Hematoxylin diversity (Hdiv; atribute ``purple_diversity``): How diverse the purple colors are.
                
        Other relevant attributes:
            * ``pinks_idx``: The indices of the pink colors in the palette array.
            * ``purples_idx``: The indices of the purple colors in the palette array.
            * ``others_idx``: The indices of the other colors not identified as pink nor purple in the palette array.
            * ``palette_colors_names``: The names of the colors in the palette array. Names supported can be found by calling ``get_reference_colors``.

        Options for ``stain``:
            * "HE": Hematoxylin and eosin.

        Args:
            palette: Array of RGB values (np.uint8) representing a color palette; shape should be (N,3). Or a Palette/ReferencePalette.
            stain: The stain type.
            identifier: The ID for this PaletteEvaluation. If None, the ID from the Palette or ReferencePalette is taken.
        """
        if identifier is not None:
            self.identifier = identifier
        elif isinstance(palette, (Palette, ReferencePalette)):
            self.identifier = palette.identifier
        else:
            Exception("Set an identifier.")

        if isinstance(palette, np.ndarray):
            self.palette_sRGB = palette / 255
        elif isinstance(palette, (Palette, ReferencePalette)):
            self.palette_sRGB = palette.get_palette(normalized=True)
        else:
            raise Exception("palette must be a np.ndarray, Palette, or ReferencePalette.")

        self.palette_Lab = self.sRGB_to_Lab(self.palette_sRGB)
        self.reference_colors, self.reference_colors_names = self.get_reference_colors()
        self.stain = stain
        self.pink_purple_separation = None
        self.pink_diversity = None
        self.purple_diversity = None
        self.pinks_idx = None
        self.purples_idx = None
        self.others_idx = None
        self.palette_colors_names = None
        self.diversity = None

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        json_output_path = os.path.join(output_dir, "palette_evaluation.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "identifier": self.identifier,
                "stain": self.stain,
                "pink_purple_separation": self.pink_purple_separation,
                "pink_diversity": self.pink_diversity,
                "purple_diversity": self.purple_diversity,
                "pinks_idx": self.pinks_idx,
                "purples_idx": self.purples_idx,
                "others_idx": self.others_idx,
                "palette_colors_names": self.palette_colors_names,
                "diversity": self.diversity,
            }, file)

    @classmethod
    def load(cls, input_dir: str, palette: np.ndarray | Palette | ReferencePalette | None) -> PaletteEvaluation:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.
            palette: The palette. Can be None if computing the PaletteEvaluation in the future is not desired and the object is just being used to store the already computed metrics.

        Returns:
            A PaletteEvaluation.
        """

        json_input_path = os.path.join(input_dir, "palette_evaluation.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        pe = cls(palette, stain=json_data["stain"], identifier=json_data["identifier"])
        pe.pink_purple_separation = json_data["pink_purple_separation"]
        pe.pink_diversity = json_data["pink_diversity"]
        pe.purple_diversity = json_data["purple_diversity"]
        pe.pinks_idx = json_data["pinks_idx"]
        pe.purples_idx = json_data["purples_idx"]
        pe.others_idx = json_data["others_idx"]
        pe.palette_colors_names = json_data["palette_colors_names"]
        pe.diversity = json_data["diversity"]

        return pe

    def get_reference_colors(self) -> tuple[np.ndarray, list[str]]:
        """Get the reference colors used to identify the "names" of the colors in the palette.

        Returns:
            A tuple containing (the reference color RGB values as a np.ndarray normalized by 255., the reference color names as a list of str).
        """
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
        
    def sRGB_to_Lab(self, sRGB: np.ndarray) -> np.ndarray:
        """Convert sRGB values into CIELAB.

        Args:
            sRGB: An array of sRGB values.

        Returns:
            A np.ndarray of the colors converted to CIELAB.
        """
        return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(sRGB))

    def palette_diversity(self):
        """Compute palette diversity. Calculates the delta E between all pairs of colors.
        """
        delta_Es = []
        for c1, c2 in itertools.combinations(self.palette_Lab, 2):

            dE = colour.delta_E(c1, c2, method="CIE 2000")
            delta_Es.append(dE)

        self.diversity = statistics.mean(delta_Es)
        
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
        """Compute the PaletteEvaluation.
        """

        self.palette_diversity()
        if self.stain == "HE":
            self.palette_he()

class PaletteComparison:

    def __init__(self, reference: PaletteEvaluation, query: PaletteEvaluation, method: str = "pairwise"):
        """Compare two palettes.

        Options for ``method``:
            * "pairwise": Compare the two palettes using pairwise comparison.
            * "dominant_he": Compare the two palettes using only their dominant pink color and dominant purple color. Assumes that the palettes are ordered such that the more dominant colors have lower index values.

        Args:
            reference: Reference PaletteEvaluation.
            query: Query PaletteEvaluation.
            method: Method of palette comparison.
        """
        self.query = query
        self.reference = reference
        self.method = method

        if len(self.query.palette_Lab) != len(self.reference.palette_Lab):
            raise Exception("The number of colors are different between the query and reference palettes.")

        self.pairwise_score = None

        # HE
        self.pink_dE = None
        self.purple_dE = None

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        json_output_path = os.path.join(output_dir, "palette_comparison.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "reference": self.reference.identifier,
                "query": self.query.identifier,
                "method": self.method,
                "pairwise_score": self.pairwise_score,
                "pink_dE": self.pink_dE,
                "purple_dE": self.purple_dE,
            }, file)

    @classmethod
    def load(cls, input_dir: str) -> PaletteComparison:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A PaletteComparison.
        """

        json_input_path = os.path.join(input_dir, "palette_comparison.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        pc = cls(json_data["reference"], json_data["query"])
        pc.dist = json_data["dist"]

        return pc

    def pairwise(self):
        """Compute a pairwise comparison between the query and reference palettes.
        """

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

    def get_val(self, val_type: str) -> float:
        """Get a metric/statistic.

        Options for ``val_type``:
            * "compare": Pairwise comparison, independent of the stain type.
            * "Hdiv": Hematoxylin diversity.
            * "Ediv": Eosin diversity.
            * "HEsep": Hematoxylin and eosin separability.

        Args:
            val_type: The name of the metric/statistic to get.

        Returns:
            The requested metric/statistic.
        """
        if val_type == "compare":
            return self.pairwise_score
        elif val_type == "Hdiv":
            return self.query.purple_diversity
        elif val_type == "Ediv":
            return self.query.pink_diversity
        elif val_type == "HEsep":
            return self.query.pink_purple_separation
        else:
            raise Exception(f"{val_type} is not a valid metric.")

    def dominant_he(self):
        """Compute the difference between the query and reference palettes by comparing their dominant pink color with each other and their dominant purple color with each other.
        """

        purple_0_idx = self.query.purples_idx[0]
        purple_1_idx = self.reference.purples_idx[1]

        pink_0_idx = self.query.pinks_idx[0]
        pink_1_idx = self.reference.pinks_idx[1]

        self.pink_dE = colour.delta_E(self.query.palette_Lab[pink_0_idx], self.reference.palette_Lab[pink_1_idx])
        self.purple_dE = colour.delta_E(self.query.palette_Lab[purple_0_idx], self.reference.palette_Lab[purple_1_idx])

    def compute(self):
        """Compute the PaletteComparison.
        """

        if self.method == "pairwise":
            self.pairwise()
        elif self.method == "dominant_he":
            self.dominant_he()
        else:
            raise Exception(f"{self.method} is not a valid method for computing the PaletteComparison.")
            