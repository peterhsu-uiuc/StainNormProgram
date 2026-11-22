from stainnormprogram.slide import SlideImage, SlideImageCollection
import stainnormprogram.norm.background as background

import os
import json

from bitarray import bitarray
import numpy as np

class GamutArray:

    def __init__(self):
        """A wrapper for a bitarray that keeps track of which colors are in the gamut.
        """
        self.array = bitarray(256 * 256 * 256)

    @staticmethod
    def colors_to_indices(rgb: np.ndarray) -> np.ndarray:
        """Colors to array indices.

        Args:
            rgb: A (N,3) array containing RGB colors as uint8.
        
        Returns:
            Indices as a (N,) np.uint32 array.
        """
        rgb = rgb.astype(np.uint32)
        indices = rgb[:,0] * 256 * 256 + rgb[:,1] * 256 + rgb[:,2]
        return indices # (N,) np.uint32

    @staticmethod
    def indices_to_colors(indices: np.ndarray | list) -> np.ndarray:
        """Array indices to colors.

        Args:
            indices: Array indices.

        Returns:
            Colors as a RGB (N,3) np.uint8 array.
        """
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

    def bitwise_or(self, gamut_array: bitarray.bitarray):
        """Perform bitwise OR.

        Args:
            gamut_array: The bitarray to perform bitwise OR with.
        """
        right_array = gamut_array.array
        left_array = self.array

        self.array = left_array | right_array

    def add_colors(self, rgb: np.ndarray): # (N,3) np.ndarray uint8
        """Add colors to the GamutArray.

        Args:
            rgb: Colors as a (N,3) np.uint8 array.
        """
        indices = self.colors_to_indices(rgb)
        for idx in indices:
            self.array[idx] = 1

    def has_colors(self, rgb: np.ndarray) -> np.ndarray: # (N,3) np.ndarray uint8
        """Check if the GamutArray as certain colors.

        Args:
            rgb: Colors to check as a (N,3) np.uint8 array.

        Returns:
            A (N,) bool np.ndarray where True means that the color is in the gamut and False otherwise.
        """
        indices = self.colors_to_indices(rgb)
        flags = []
        for position in range(indices.size):
            flags[position] = bool(self.array[indices[position]])
        flags = np.array(flags)
        return flags # (N,) np.ndarray bool

class Gamut:

    def __init__(
        self,
        slide: SlideImage | SlideImageCollection,
        identifier: str | None = None,
        tile_size: int = 32,
        background_method: str = "hsv_threshold",
        accept_perc: int = 10,
        max_pixels: int = 1_000_000
        ):
        """Compute the Gamut for an image.

        Options for ``background_method``:
            * "hsv_threshold": Threshold using saturation and value channels in HSV color space.

        Args:
            slide: The slide to calculate the gamut of.
            identifier: An ID for this Gamut. If None, the identifier of the slide will be used.
            tile_size: Tile size to use.
            background_method: Method for identifying the background/foreground.
            accept_perc: Tiles with less than accept_perc percent of foreground are not used in the computation.
            max_pixels: If the number of pixels used goes over max_pixels, stop the computation.
        """
        
        self.slide = slide
        self.identifier = identifier if identifier is not None else self.slide.identifier
        self.background_method = background_method
        self.accept_perc = accept_perc
        
        # Tile size
        if isinstance(self.slide, (SlideImage, SlideImageCollection)):
            self.slide.set_tile_size(tile_size)

        self.num_pixels = 0
        self.num_tiles = 0
        self.max_pixels = max_pixels
        self.tile_size = tile_size
        
        self.gamut_array = GamutArray()

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        json_output_path = os.path.join(output_dir, "gamut.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "identifier": self.identifier,
                "background_method": self.background_method,
                "accept_perc": self.accept_perc,
                "num_pixels": self.num_pixels,
                "num_tiles": self.num_tiles,
                "max_pixels": self.max_pixels,
                "tile_size": self.tile_size,
            }, file)

        gamut_output_path = os.path.join(output_dir, "gamut.bin")
        with open(gamut_output_path, "wb") as f:
            self.gamut_array.tofile(f)

    @classmethod
    def load(cls, input_dir: str) -> Gamut:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A Gamut.
        """

        json_input_path = os.path.join(input_dir, "gamut.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        gamut = cls(None, json_data["identifier"], json_data["tile_size"], json_data["background_method"], json_data["accept_perc"], json_data["max_pixels"])
        
        gamut_input_path = os.path.join(input_dir, "gamut.bin")
        with open(gamut_input_path, "rb") as f:
            gamut.gamut_array.fromfile(f)

        return gamut

    def compute_tile(self, tile: Tile):
        """Add colors from the image tile to the gamut.

        Args:
            tile: Image tile.
        """

        if self.background_method == "hsv_threshold":
            fg_mask = background.hsv_threshold_foreground(tile).flatten()
        else:
            raise Exception(f"{self.background_method} is not a valid background estimation method.")

        n = np.sum(fg_mask).item()
        if n / fg_mask.size * 100 < self.accept_perc:
            return

        tile_rgb_flat = tile.rgb(as_uint8=True).reshape((-1,3))
        self.gamut_array.add_colors(tile_rgb_flat)

        self.num_tiles = self.num_tiles + 1
        self.num_pixels = self.num_pixels + n

    def compute(self):
        """Compute the Gamut.
        """
    
        print(f"Computing Gamut for {self.identifier}...")
    
        self.slide.set_mode("random")
        for tile, _ in self.slide:
            self.compute_tile(tile)
            if self.num_pixels > self.max_pixels:
                break

        print("Finished computing Gamut")

class GamutComparison:

    def __init__(self, reference: ReferenceGamut, query: Gamut):
        """Compare a query Gamut with a ReferenceGamut.

        Args:
            reference: The ReferenceGamut.
            query: The query Gamut.
        """
        self.reference = reference
        self.query = query

        # Out-of-gamut count and colors
        self.oog_count = None
        self.oog_colors = None # np.ndarray (N,3) uint8

        self.ig_count = None

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        json_output_path = os.path.join(output_dir, "gamut_comparison.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "reference": self.reference.identifier,
                "query": self.query.identifier,
                "oog_count": self.oog_count,
                "ig_count": self.ig_count,
            }, file)

        oog_output_path = os.path.join(output_dir, "gamut_oog_colors.npz")
        if self.oog_colors is not None:
            np.savez(oog_output_path, oog_colors=self.oog_colors, allow_pickle=False)


    @classmethod
    def load(cls, input_dir: str) -> GamutComparison:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A GamutComparison.
        """

        json_input_path = os.path.join(input_dir, "gamut_comparison.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        gc = cls(json_data["reference"], json_data["query"])
        
        oog_input_path = os.path.join(output_dir, "gamut_oog_colors.npz")
        if os.path.exists(oog_input_path):
            with np.load(oog_input_path, allow_pickle=False) as data:
                gc.oog_colors = data["oog_colors"]

        return gc

    def get_oog_colors(self, oog_mask: np.ndarray) -> np.ndarray:
        """Get the out-of-gamut colors.

        Args:
            oog_mask: A bool np.ndarray where True indicates an out-of-gamut color. Each index corresponds to a color (as defined by GamutArray).

        Returns:
            Colors as a RGB (N,3) np.uint8 array.
        """

        indices = np.where(oog_mask)[0]
        colors = GamutArray.indices_to_colors(indices)

        return colors

    def get_val(self, val_type: str) -> int:
        """Get a metric computed by this GamutComparison.

        Options for ``val_type``:
            * "oog_count": Out-of-gamut color count.
            * "ig_count": In-gamut color count.

        Returns:
            The requested metric.
        """
        if val_type == "oog_count":
            return self.oog_count
        elif val_type == "ig_count":
            return self.ig_count
        else:
            raise Exception(f"{val_type} is not a valid metric that can be requested.")

    def compute(self, get_colors: bool = False):
        """Compute the GamutComparison.

        Args:
            get_colors: If True, attribute ``oog_colors`` is set to the out-of-gamut RGB colors. Nothing is returned.
        """

        ref_array = self.reference.gamut_array.array
        ref_array = np.array(ref_array.tolist())
        
        q_array = self.query.gamut_array.array
        q_array = np.array(q_array.tolist())

        # An out-of-gamut color is when query is 1 but ref is 0
        oog_mask = (q_array == 1) & (ref_array == 0)
        self.oog_count = np.sum(oog_mask).item()

        ig_mask = (q_array == 1) & (ref_array == 1)
        self.ig_count  = np.sum(ig_mask).item()

        if get_colors:
            self.oog_colors = self.get_oog_colors(oog_mask)

class ReferenceGamut:

    def __init__(self, ref_gamuts: Gamut | list[Gamut], identifier: str | None = None):
        """A gamut for the reference distribution (i.e., a single gamut to represent multiple reference images).

        Args:
            ref_gamuts: Either a single Gamut or a list of Gamuts that represent the reference.
            identifier: An ID for this reference.
        """
        self.ref_gamuts = ref_gamuts
        if identifier is not None:
            self.identifier = identifier
        elif isinstance(ref_gamuts, Gamut):
            self.identifier = ref_gamuts.identifier
        elif isinstance(ref_gamuts, list) and len(ref_gamuts) == 1:
            self.identifier = ref_gamuts[0].identifier
        else:
            raise Exception("Set an identifier when more than one Gamut is in ref_gamuts.")

        if isinstance(self.ref_gamuts, Gamut):
            self.ref_gamuts = [ref_gamuts]

        self.gamut_array = GamutArray()

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        ref_gamut_identifiers = []
        for rg in self.ref_gamuts:
            ref_gamut_identifiers.append(rg)

        json_output_path = os.path.join(output_dir, "reference_gamut.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "identifier": self.identifier,
                "ref_gamut_identifiers": ref_gamut_identifiers,
            }, file)

        gamut_output_path = os.path.join(output_dir, "reference_gamut.bin")
        with open(gamut_output_path, "wb") as f:
            self.gamut_array.tofile(f)

    @classmethod
    def load(cls, input_dir: str) -> ReferenceGamut:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A ReferenceGamut.
        """

        json_input_path = os.path.join(input_dir, "reference_gamut.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        gamut = cls(json_data["ref_gamut_identifiers"], identifier=json_data["identifier"])
        
        gamut_input_path = os.path.join(input_dir, "reference_gamut.bin")
        with open(gamut_input_path, "rb") as f:
            gamut.gamut_array.fromfile(f)

        return gamut

    def compute(self):
        """Compute the ReferenceGamut.
        """

        for g in self.ref_gamuts:
            self.gamut_array.bitwise_or(g.gamut_array)
