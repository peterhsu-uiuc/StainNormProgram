from stainnormprogram.slide import SlideImage, SlideImageCollection
import stainnormprogram.norm.background as background

import statistics
import os
import json

import numpy as np

class Contrast:

    def __init__(
        self,
        slide: SlideImage | SlideImageCollection,
        identifier: str | None = None,
        tile_size: int = 512,
        background_method: str = "hsv_threshold",
        accept_perc: int = 10,
        max_pixels: int = 50_000_000,
        tile_locs: list[dict[str, str | int]] | None = None
        ):
        """``Contrast`` estimates the root mean square (RMS) contrast of an image.

        Options for ``background_method``:
            * "hsv_threshold": Threshold using saturation and value channels in HSV color space.

        If you want to compare the contrast between two images and use the same tiles for both, first compute the Contrast for one image, then get the ``tile_locs`` attribute from that Contrast and give it to the second image's Contrast using the ``tile_locs`` parameter.

        Args:
            slide: Image to perform contrast estimation on.
            identifier: An identifier. If None, the identifier is taken from the slide.
            tile_size: Tile size to use.
            background_method: Method for identifying the background/foreground.
            accept_perc: Tiles with less than accept_perc percent of foreground are not used in the computation.
            max_pixels: Max number of pixels to use.
            tile_locs: Tile locations to visit. Useful when you want to compare the contrast between two images and use the same tiles for both. Should be a list of locations, each location is a dict with "x", "y", "w", "h", and (if SlideImageCollection) "identifier" keys.
        """
        self.slide = slide
        self.identifier = identifier if identifier is not None else self.slide.identifier
        self.background_method = background_method
        self.accept_perc = accept_perc

        # Tile size
        if self.slide is not None:
            self.slide.set_tile_size(tile_size)
        
        self.num_pixels = 0
        self.num_tiles = 0
        self.max_pixels = max_pixels
        self.tile_size = tile_size

        self.rms_vals = []
        self.tile_locs = [] if tile_locs is None else tile_locs

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        json_output_path = os.path.join(output_dir, "contrast.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "identifier": self.identifier,
                "background_method": self.background_method,
                "accept_perc": self.accept_perc,
                "tile_size": self.tile_size,
                "num_pixels": self.num_pixels,
                "num_tiles": self.num_tiles,
                "max_pixels": self.max_pixels,
            }, file)

        contrast_output_path = os.path.join(input_dir, "contrast_vals.npz")
        np.savez(contrast_output_path, rms_vals=self.rms_vals, tile_locs=self.tile_locs, allow_pickle=False)

    @classmethod
    def load(cls, input_dir: str) -> Contrast:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A Contrast.
        """

        json_input_path = os.path.join(input_dir, "contrast.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        contrast = cls(None, identifier=json_data["identifier"], tile_size=json_data["tile_size"], background_method=json_data["background_method"], accept_perc=json_data["accept_perc"], max_pixels=json_data["max_pixels"])
        
        contrast_input_path = os.path.join(input_dir, "contrast_vals.npz")
        with np.load(contrast_input_path, allow_pickle=False) as data:
            contrast.rms_vals, contrast.tile_locs = data["rms_vals"], data["tile_locs"]

        return contrast

    def compute_tile(self, tile: Tile, fg_filtering: bool = True) -> bool:
        """Compute the RMS contrast of an image tile.

        Args:
            tile: A Tile.
            fg_filtering: If True, tiles with less than accept_perc percent of foreground pixels will be rejected.

        Returns:
            A bool indicating whether the tile was accepted for computation or not.
        """

        if self.background_method == "hsv_threshold":
            fg_mask = background.hsv_threshold_foreground(tile).flatten()
        else:
            raise Exception(f"{self.background_method} is not a valid method.")

        n = np.sum(fg_mask).item()
        if fg_filtering and n / fg_mask.size * 100 < self.accept_perc:
            return False

        lum = tile.luv()[:,:,0].flatten()
        lum = lum / 100. # Normalize to [0., 1.]
        
        # RMS contrast
        rms = np.std(lum)

        self.num_tiles = self.num_tiles + 1
        self.num_pixels = self.num_pixels + n

        self.rms_vals.append(rms)

        return True

    def compute(self):
        """Compute the Contrast.
        """

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

            if success and self.num_tiles % 10 == 0:
                print(f"Processed {self.num_tiles} tiles so far...")

            if self.slide.mode == "random" and success:
                self.tile_locs.append(metadata)
                if self.num_pixels > self.max_pixels:
                    break

        print("Finished computing Contrast")
        print(f"Tiles used = {self.num_tiles}")
        print(f"Foreground pixels used = {self.num_pixels}")


class ContrastChange:

    def __init__(self, reference: Contrast, query: Contrast):
        """Compute the contrast change between a reference and query.

        Args:
            reference: The Contrast of a reference image.
            query: The Contrast of a query image.
        """
        self.reference = reference
        self.query = query

        self.changes = []

        self.change_mean = None
        self.change_std = None

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        json_output_path = os.path.join(output_dir, "contrast_change.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "reference": self.reference.identifier,
                "query": self.query.identifier,
                "change_mean": self.change_mean,
                "change_std": self.change_std,
            }, file)

        contrast_output_path = os.path.join(input_dir, "contrast_change_vals.npz")
        np.savez(contrast_output_path, changes=self.changes, allow_pickle=False)

    @classmethod
    def load(cls, input_dir: str) -> ContrastChange:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A ContrastChange.
        """

        json_input_path = os.path.join(input_dir, "contrast_change.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        cc = cls(json_data["reference"], json_data["query"])
        cc.change_mean = json_data["change_mean"]
        cc.change_std = json_data["change_std"]

        contrast_input_path = os.path.join(input_dir, "contrast_change_vals.npz")
        with np.load(contrast_input_path, allow_pickle=False) as data:
            cc.changes = data["changes"].tolist()

        return cc

    def get_val(self) -> tuple[float, float]:
        """Get the contrast change metric.

        If the change is positive, the query increased in contrast relative to the reference.
        If the change is negative, the query decreased in contrast relative to the reference.

        Returns:
            A tuple with (contrast change mean, contrast change standard deviation).
        """
        return (self.change_mean, self.change_std)

    def compute(self):
        """Compute the ContrastChange.
        """
        
        num_vals = len(self.reference.rms_vals)

        if num_vals != len(self.query.rms_vals):
            raise Exception(f"Number of RMS values are not equal for reference and query. Reference = {num_vals}, Query = {len(self.query.rms_vals)}")

        for i in range(num_vals):

            change = self.query.rms_vals[i] - self.reference.rms_vals[i]
            self.changes.append(change)

        self.change_mean = statistics.mean(self.changes)
        self.change_std = statistics.stdev(self.changes)

