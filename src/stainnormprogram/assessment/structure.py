from stainnormprogram.slide import SlideImage, SlideImageCollection
import stainnormprogram.norm.background as background

import statistics
import os
import json

import numpy as np

class StructurePreservation:

    def __init__(
        self,
        reference: SlideImage | SlideImageCollection,
        query: SlideImage | SlideImageCollection,
        tile_size: int = 256,
        background_method: str = "hsv_threshold",
        accept_perc: int = 10,
        max_pixels: int = 10_000_000
        ):
        """Assess the structure preservation between the original and stain normalized versions of an image.

        Structure preservation is defined here as the preservation of edges as identified by Canny edge detection.

        Options for ``background_method``:
            * "hsv_threshold": Threshold using saturation and value channels in HSV color space.

        Args:
            reference: The reference image.
            query: The query image.
            tile_size: Tile size to use.
            background_method: Method for identifying the background/foreground.
            accept_perc: Tiles with less than accept_perc percent of foreground are not used in the computation.
            max_pixels: If the number of pixels used goes over max_pixels, stop the computation.
        """
        self.reference = reference
        self.query = query
        self.background_method = background_method
        self.accept_perc = accept_perc

        # Tile size
        if reference is not None:
            self.reference.set_tile_size(tile_size)
        if query is not None:
            self.query.set_tile_size(tile_size)
        
        self.num_pixels = 0
        self.num_tiles = 0
        self.max_pixels = max_pixels
        self.tile_size = tile_size

        self.diffs = []
        self.diff_mean = None
        self.diff_std = None

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        json_output_path = os.path.join(output_dir, "structure_preservation.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "reference": self.reference.identifier,
                "query": self.query.identifier,
                "background_method": self.background_method,
                "accept_perc": self.accept_perc,
                "num_pixels": self.num_pixels,
                "num_tiles": self.num_tiles,
                "max_pixels": self.max_pixels,
                "tile_size": self.tile_size,
                "diff_mean": self.diff_mean,
                "diff_std": self.diff_std,
            }, file)

        structure_output_path = os.path.join(output_dir, "structure_preservation_diffs.npz")
        np.savez(structure_output_path, diffs=self.diffs, allow_pickle=False)

    @classmethod
    def load(cls, input_dir: str) -> StructurePreservation:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A StructurePreservation.
        """

        json_input_path = os.path.join(input_dir, "structure_preservation.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        sp = cls(json_data["reference"], json_data["query"], json_data["tile_size"], json_data["background_method"], json_data["accept_perc"], json_data["max_pixels"])
        
        structure_input_path = os.path.join(input_dir, "structure_preservation_diffs.npz")
        with np.load(structure_input_path, allow_pickle=False) as data:
            sp.diffs = data["diffs"]

        return sp

    def compute_tile(self, ref_tile: Tile, q_tile: Tile):
        """Compute the structure preservation between two tiles.

        Args:
            ref_tile: The reference tile.
            q_tile: The query tile.
        """

        if self.background_method == "hsv_threshold":
            fg_mask = background.hsv_threshold_foreground(ref_tile).flatten()
        else:
            raise Exception

        n = np.sum(fg_mask).item()
        if n / fg_mask.size * 100 < self.accept_perc:
            return None

        ref_canny = ref_tile.canny()
        q_canny = q_tile.canny()

        diff_mask = ref_canny ^ q_canny
        diff_count = np.sum(diff_mask).item()

        self.num_tiles = self.num_tiles + 1
        self.num_pixels = self.num_pixels + n
        
        self.diffs.append(diff_count)

    def get_val(self) -> tuple[float, float]:
        """Get the value of the metric.

        Returns:
            A tuple containing (difference mean, difference standard deviation).
        """
        return (self.diff_mean, self.diff_std)

    def compute(self):
        """Compute the StructurePreservation.
        """
        
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