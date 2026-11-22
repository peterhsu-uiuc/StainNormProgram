import stainnormprogram.norm.background as background
from stainnormprogram.slide import SlideImage, SlideImageCollection

import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter

class ColorHistogramEdges:

    def __init__(self,
                 rgb_bins: tuple[int, int, int] = (51,51,51),
                 h_bins: int = 200,
                 s_bins: int = 100,
                 v_bins: int = 100,
                 oklab_bins: tuple[int, int, int] = (200,160,160),
                 oklab_a_range: tuple[float, float] = (-0.4,0.4),
                 oklab_b_range: tuple[float, float] = (-0.4,0.4),
                 uv_bins: tuple[int, int] = (100,100),
                 uv_u_range: tuple[float, float] = (0.0,0.6),
                 uv_v_range: tuple[float, float] = (0.0,0.6),
        ):
        """Shared store for bins and edges of color histograms.

        Args:
            rgb_bins: (red, green, blue) bin count.
            h_bins: Number of hue (HSV) bins.
            s_bins: Number of saturation (HSV) bins.
            v_bins: Number of value/brightness (HSV) bins.
            oklab_bins: OKLAB (l, a, b) bin count.
            oklab_a_range: OKLAB a range.
            oklab_b_range: OKLAB b range.
            uv_bins: Number of CIE 1976 UCS chromaticity u'v' bins as (u' bins, v' bins).
            uv_u_range: u'v' u range.
            uv_v_range: u'v' v range.
        """

        self.rgb_bins = rgb_bins
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.oklab_bins = oklab_bins
        self.oklab_a_range = oklab_a_range
        self.oklab_b_range = oklab_b_range
        self.uv_bins = uv_bins
        self.uv_u_range = uv_u_range
        self.uv_v_range = uv_v_range

        self.hsv_h_hist_edges = np.linspace(0, 1, num=self.h_bins+1)
        self.hsv_s_hist_edges = np.linspace(0, 1, num=self.s_bins+1)
        self.hsv_v_hist_edges = np.linspace(0, 1, num=self.v_bins+1)
        self.rgb_r_hist_edges = np.linspace(0, 255, num=self.rgb_bins[0]+1)
        self.rgb_g_hist_edges = np.linspace(0, 255, num=self.rgb_bins[1]+1)
        self.rgb_b_hist_edges = np.linspace(0, 255, num=self.rgb_bins[2]+1)
        self.oklab_l_hist_edges = np.linspace(0, 1, num=self.oklab_bins[0]+1)
        self.oklab_a_hist_edges = np.linspace(self.oklab_a_range[0], self.oklab_a_range[1], num=self.oklab_bins[1]+1)
        self.oklab_b_hist_edges = np.linspace(self.oklab_b_range[0], self.oklab_b_range[1], num=self.oklab_bins[2]+1)
        self.uv_u_hist_edges = np.linspace(self.uv_u_range[0], self.uv_u_range[1], num=self.uv_bins[0]+1)
        self.uv_v_hist_edges = np.linspace(self.uv_v_range[0], self.uv_v_range[1], num=self.uv_bins[1]+1)

    def __eq__(self, other: ColorHistogramEdges) -> bool:
        """Compare the equality between this and other ColorHistogramEdges objects.

        Returns:
            A bool indicating whether the two objects have the same edges or not.
        """

        return self.rgb_bins == other.rgb_bins and self.h_bins == other.h_bins \
        and self.s_bins == other.s_bins and self.v_bins == other.v_bins \
        and self.oklab_bins == other.oklab_bins and self.oklab_a_range == other.oklab_a_range \
        and self.oklab_b_range == other.oklab_b_range and self.uv_bins == other.uv_bins \
        and self.uv_u_range == other.uv_u_range and self.uv_v_range == other.uv_v_range \
        and np.array_equal(self.hsv_h_hist_edges, other.hsv_h_hist_edges) and np.array_equal(self.hsv_s_hist_edges, other.hsv_s_hist_edges) \
        and np.array_equal(self.hsv_v_hist_edges, other.hsv_v_hist_edges) and np.array_equal(self.rgb_r_hist_edges, other.rgb_r_hist_edges) \
        and np.array_equal(self.rgb_g_hist_edges, other.rgb_g_hist_edges) and np.array_equal(self.rgb_b_hist_edges, other.rgb_b_hist_edges) \
        and np.array_equal(self.oklab_l_hist_edges, other.oklab_l_hist_edges) and np.array_equal(self.oklab_a_hist_edges, other.oklab_a_hist_edges) \
        and np.array_equal(self.oklab_b_hist_edges, other.oklab_b_hist_edges) and np.array_equal(self.uv_u_hist_edges, other.uv_u_hist_edges) \
        and np.array_equal(self.uv_v_hist_edges, other.uv_v_hist_edges)

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        edges_output_path = os.path.join(output_dir, "color_histogram_edges.npz")
        np.savez(edges_output_path, hsv_h_hist_edges=self.hsv_h_hist_edges,
            hsv_s_hist_edges=self.hsv_s_hist_edges, hsv_v_hist_edges=self.hsv_v_hist_edges, rgb_r_hist_edges=self.rgb_r_hist_edges,
            rgb_g_hist_edges=self.rgb_g_hist_edges, rgb_b_hist_edges=self.rgb_b_hist_edges, oklab_l_hist_edges=self.oklab_l_hist_edges,
            oklab_a_hist_edges=self.oklab_a_hist_edges, oklab_b_hist_edges=self.oklab_b_hist_edges, uv_u_hist_edges=self.uv_u_hist_edges,
            uv_v_hist_edges=self.uv_v_hist_edges, allow_pickle=False
        )

        json_output_path = os.path.join(output_dir, "color_histogram_edges.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "rgb_bins": self.rgb_bins,
                "h_bins": self.h_bins,
                "s_bins": self.s_bins,
                "v_bins": self.v_bins,
                "oklab_bins": self.oklab_bins,
                "oklab_a_range": self.oklab_a_range,
                "oklab_b_range": self.oklab_b_range,
                "uv_bins": self.uv_bins,
                "uv_u_range": self.uv_u_range,
                "uv_v_range": self.uv_v_range,
            }, file)

    @classmethod
    def load(cls, input_dir: str) -> ColorHistogramEdges:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A ColorHistogramEdges.
        """

        json_input_path = os.path.join(input_dir, "color_histogram_edges.json")
        if not os.path.exists(json_input_path):
            return None
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        edges = cls(rgb_bins=json_data["rgb_bins"],
            h_bins=json_data["h_bins"],
            s_bins=json_data["s_bins"],
            v_bins=json_data["v_bins"],
            oklab_bins=json_data["oklab_bins"],
            oklab_a_range=json_data["oklab_a_range"],
            oklab_b_range=json_data["oklab_b_range"],
            uv_bins=json_data["uv_bins"],
            uv_u_range=json_data["uv_u_range"],
            uv_v_range=json_data["uv_v_range"]
        )

        edges_input_path = os.path.join(input_dir, "color_histogram_edges.npz")
        if not os.path.exists(edges_input_path):
            return None
        with np.load(edges_input_path, allow_pickle=False) as data:
            edges.hsv_h_hist_edges, edges.hsv_s_hist_edges, edges.hsv_v_hist_edges, edges.rgb_r_hist_edges, edges.rgb_g_hist_edges, edges.rgb_b_hist_edges, edges.oklab_l_hist_edges, edges.oklab_a_hist_edges, edges.oklab_b_hist_edges, edges.uv_u_hist_edges, edges.uv_v_hist_edges = \
            data["hsv_h_hist_edges"], data["hsv_s_hist_edges"], data["hsv_v_hist_edges"], data["rgb_r_hist_edges"], data["rgb_g_hist_edges"], data["rgb_b_hist_edges"], data["oklab_l_hist_edges"], data["oklab_a_hist_edges"], data["oklab_b_hist_edges"], data["uv_u_hist_edges"], data["uv_v_hist_edges"]

        return edges

class ColorSample:

    def __init__(self,
                 slide: SlideImage | SlideImageCollection,
                 edges: ColorHistogramEdges,
                 background_method: str = "hsv_threshold",
                 sampling_method: str = "random",
                 max_pixels: int = 10_000_000,
                 tile_size: int = 256,
                 identifier: int | None = None,
        ):
        """ColorSample creates color histograms of the slide.

        Options for ``background_method``:
            * "hsv_threshold": Threshold using saturation and value channels in HSV color space.

        Options for ``sampling_method``:
            * "sequential": Row major order.
            * "random": Random selection from a grid.
            * "custom": Custom order. You should specify this custom order by calling set_custom_iter_locs on your slide before constructing the ColorSample.

        Args:
            slide: The slide to compute the ColorSample of.
            edges: Histogram edges.
            background_method: Method for identifying the background/foreground.
            sampling_method: Method for sampling tiles from the image.
            max_pixels: If the number of pixels used goes over max_pixels, stop the computation.
            tile_size: Tile size to use.
            identifier: An ID for this ColorSample. If None, the identifier of the slide will be used.
        """
        
        self.slide = slide
        self.edges = edges
        self.background_method = background_method
        self.sampling_method = sampling_method
        self.max_pixels = max_pixels
        self.identifier = identifier if identifier is not None else self.slide.identifier
        self.tile_size = tile_size
        
        # Setup histograms
        self.rgb_hist = np.zeros(self.edges.rgb_bins, dtype=np.uint64)
        self.h_hist = np.zeros((self.edges.h_bins,), dtype=np.uint64)
        self.s_hist = np.zeros((self.edges.s_bins,), dtype=np.uint64)
        self.v_hist = np.zeros((self.edges.v_bins,), dtype=np.uint64)
        self.oklab_hist = np.zeros(self.edges.oklab_bins, dtype=np.uint64)
        self.uv_hist = np.zeros(self.edges.uv_bins, dtype=np.uint64)

        # Counters
        self.num_tiles = 0
        self.num_pixels = 0

        # Tile size
        if slide is not None:
            self.slide.set_tile_size(self.tile_size)

    def save(self, output_dir: str, include_edges: bool = True):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
            include_edges: Save the histogram edges as well (I recommend you set this to False if you already have the edges saved somewhere else to avoid wasting disk memory).
        """

        json_output_path = os.path.join(output_dir, "color_sample.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "background_method": self.background_method,
                "sampling_method": self.sampling_method,
                "max_pixels": self.max_pixels,
                "identifier": self.identifier,
                "num_tiles": self.num_tiles,
                "num_pixels": self.num_pixels,
                "tile_size": self.tile_size,
            }, file)
        
        hist_output_path = os.path.join(output_dir, "color_sample_histograms.npz")
        np.savez(hist_output_path, rgb_hist=self.rgb_hist, h_hist=self.h_hist, s_hist=self.s_hist, v_hist=self.v_hist, oklab_hist=self.oklab_hist, uv_hist=self.uv_hist, allow_pickle=False)

        if include_edges:
            self.edges.save(output_dir)

    @classmethod
    def load(cls, input_dir: str, edges: ColorHistogramEdges | None = None) -> ColorSample:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.
            edges: Color histogram edges. If None, the edges will be loaded from input_dir.

        Returns:
            A ColorSample.
        """

        json_input_path = os.path.join(input_dir, "color_sample.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        if edges is None:
            edges = ColorHistogramEdges.load(input_dir)

        cs = cls(None, edges, background_method=json_data["background_method"], sampling_method=json_data["sampling_method"], max_pixels=json_data["max_pixels"], tile_size=json_data["tile_size"], identifier=json_data["identifier"])

        hist_input_path = os.path.join(input_dir, "color_sample_histograms.npz")
        with np.load(hist_input_path, allow_pickle=False) as data:
            cs.rgb_hist, cs.h_hist, cs.s_hist, cs.v_hist, cs.oklab_hist, cs.uv_hist = data["rgb_hist"], data["h_hist"], data["s_hist"], data["v_hist"], data["oklab_hist"], data["uv_hist"]

        cs.num_tiles = json_data["num_tiles"]
        cs.num_pixels = json_data["num_pixels"]

        return cs

    def get_h_hist(self, density: bool = False) -> np.ndarray:
        """Get the hue (HSV) histogram.

        Args:
            density: Return as a density.

        Returns:
            A np.ndarray for the histogram.
        """
        if density:
            out = self.h_hist / np.sum(self.h_hist)
            return out
        return self.h_hist

    def get_s_hist(self, density: bool = False) -> np.ndarray:
        """Get the saturation (HSV) histogram.

        Args:
            density: Return as a density.

        Returns:
            A np.ndarray for the histogram.
        """
        if density:
            out = self.s_hist / np.sum(self.s_hist)
            return out
        return self.s_hist

    def get_v_hist(self, density: bool = False) -> np.ndarray:
        """Get the value (HSV) histogram.

        Args:
            density: Return as a density.

        Returns:
            A np.ndarray for the histogram.
        """
        if density:
            out = self.v_hist / np.sum(self.v_hist)
            return out
        return self.v_hist

    def get_rgb_hist(self, density: bool = False) -> np.ndarray:
        """Get the RGB histogram.

        Args:
            density: Return as a density.

        Returns:
            A np.ndarray for the histogram.
        """
        if density:
            out = self.rgb_hist / np.sum(self.rgb_hist)
            return out
        return self.rgb_hist

    def get_oklab_hist(self, density: bool = False) -> np.ndarray:
        """Get the OKLAB histogram.

        Args:
            density: Return as a density.

        Returns:
            A np.ndarray for the histogram.
        """
        if density:
            out = self.oklab_hist / np.sum(self.oklab_hist)
            return out
        return self.oklab_hist

    def get_uv_hist(self, density: bool = False) -> np.ndarray:
        """Get the u'v' histogram.

        Args:
            density: Return as a density.

        Returns:
            A np.ndarray for the histogram.
        """
        if density:
            out = self.uv_hist / np.sum(self.uv_hist)
            return out
        return self.uv_hist

    def compute_tile(self, tile: Tile):
        """Compute the histogram bar counts from a tile.

        Args:
            tile: The tile to compute counts from.
        """

        if self.background_method == "hsv_threshold":
            fg_mask = background.hsv_threshold_foreground(tile).flatten()
        else:
            raise Exception(f"Background method {self.background_method} not supported.")

        n = np.sum(fg_mask).item()
        if n == 0:
            return

        if self.num_tiles % 10 == 0:
            print(f"Processed {self.num_tiles} tiles so far...")

        rgb_tile = tile.rgb(as_uint8=True).reshape((-1,3))
        hsv_tile = tile.hsv().reshape((-1,3))
        oklab_tile = tile.oklab().reshape((-1,3))
        uv_u_tile, uv_v_tile = tile.u_prime_v_prime()
        uv_u_tile = uv_u_tile.flatten()
        uv_v_tile = uv_v_tile.flatten()

        rgb = rgb_tile[fg_mask, :]
        hsv_h = hsv_tile[:,0][fg_mask]
        hsv_s = hsv_tile[:,1][fg_mask]
        hsv_v = hsv_tile[:,2][fg_mask]
        oklab = oklab_tile[fg_mask, :]
        uv_u = uv_u_tile[fg_mask]
        uv_v = uv_v_tile[fg_mask]
        uv_not_nan = ~(np.isnan(uv_u) | np.isnan(uv_v))
        uv_u = uv_u[uv_not_nan]
        uv_v = uv_v[uv_not_nan]
        
        self.h_hist += np.histogram(hsv_h, bins=self.edges.hsv_h_hist_edges)[0].astype(np.uint64)
        self.s_hist += np.histogram(hsv_s, bins=self.edges.hsv_s_hist_edges)[0].astype(np.uint64)
        self.v_hist += np.histogram(hsv_v, bins=self.edges.hsv_v_hist_edges)[0].astype(np.uint64)

        r_idx = (rgb[:,0] / 255 * (self.edges.rgb_bins[0] - 1)).astype(int)
        g_idx = (rgb[:,1] / 255 * (self.edges.rgb_bins[1] - 1)).astype(int)
        b_idx = (rgb[:,2] / 255 * (self.edges.rgb_bins[2] - 1)).astype(int)

        np.add.at(self.rgb_hist, (r_idx, g_idx, b_idx), 1)

        l_idx = (oklab[:,0] * (self.edges.oklab_bins[0] - 1)).astype(int) 
        a_idx = ((oklab[:,1] - self.edges.oklab_a_range[0]) / (self.edges.oklab_a_range[1] - self.edges.oklab_a_range[0]) * (self.edges.oklab_bins[1] - 1)).astype(int)
        b_idx = ((oklab[:,2] - self.edges.oklab_b_range[0]) / (self.edges.oklab_b_range[1] - self.edges.oklab_b_range[0]) * (self.edges.oklab_bins[2] - 1)).astype(int)
        
        np.add.at(self.oklab_hist, (l_idx, a_idx, b_idx), 1)

        u_idx = ((uv_u - self.edges.uv_u_range[0]) / (self.edges.uv_u_range[1] - self.edges.uv_u_range[0]) * (self.edges.uv_bins[0] - 1)).astype(int)
        v_idx = ((uv_v - self.edges.uv_v_range[0]) / (self.edges.uv_v_range[1] - self.edges.uv_v_range[0]) * (self.edges.uv_bins[1] - 1)).astype(int)

        np.add.at(self.uv_hist, (u_idx, v_idx), 1)

        self.num_tiles = self.num_tiles + 1
        self.num_pixels = self.num_pixels + n
    
    def compute(self):
        """Compute the ColorSample.
        """

        print(f"Computing ColorSample for {self.identifier}...")

        self.slide.set_mode(self.sampling_method)
        for tile, _ in self.slide:
            self.compute_tile(tile)
            if self.num_pixels > self.max_pixels:
                break
            
        print("Finished computing ColorSample")
    

class ColorDistribution:

    def __init__(self, samples: ColorSample | list[ColorSample], identifier: str | None = None, equally_weighted: bool = False):
        """Color distribution is a Gaussian smoothed version of one or more ColorSamples to represent a color distribution.

        Args:
            samples: One or more ColorSamples.
            identifier: An ID for the ColorDistribution. If None, only one ColorSample should be given in samples, and the ID will be taken from that ColorSample.
            equally_weighted: Equally weight each ColorSample in the distribution.
        """
        
        self.samples = samples
        if identifier is not None:
            self.identifier = identifier
        elif isinstance(samples, ColorSample):
            self.identifier = samples.identifier
        elif (isinstance(samples, list) and len(samples) == 1):
            self.identifier = samples[0].identifier
        else:
            raise Exception("Specify an identifier for this ColorDistribution.")
        self.equally_weighted = equally_weighted

        if samples is not None:
            self.set_edges(samples)

        # Setup histograms
        hist_dtype = np.float64 if self.equally_weighted else np.uint64
        self.rgb_hist = np.zeros(self.edges.rgb_bins, dtype=hist_dtype)
        self.h_hist = np.zeros((self.edges.h_bins,), dtype=hist_dtype)
        self.s_hist = np.zeros((self.edges.s_bins,), dtype=hist_dtype)
        self.v_hist = np.zeros((self.edges.v_bins,), dtype=hist_dtype)
        self.oklab_hist = np.zeros(self.edges.oklab_bins, dtype=hist_dtype)
        self.uv_hist = np.zeros(self.edges.uv_bins, dtype=hist_dtype)

    def set_edges(self, samples: ColorSample | list[ColorSample]):
        """Set the edges of this ColorDistribution from one or more ColorSamples.

        If multiple ColorSamples are given, this function will check that all ColorSamples have the same edges.

        Args:
            samples: One or more ColorSamples.
        """
        
        if isinstance(samples, list):
            init_sample = samples[0]
        if isinstance(samples, ColorSample):
            init_sample = samples

        self.edges = init_sample.edges

        if isinstance(samples, list):
            for sample in samples:
                if self.edges != sample.edges:
                    raise Exception("The bin counts and ranges must match for all ColorSamples.")

    def save(self, output_dir: str, include_edges: bool = True):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
            include_edges: Save the histogram edges as well (I recommend you set this to False if you already have the edges saved somewhere else to avoid wasting disk memory).
        """

        sample_identifiers = []
        if isinstance(self.samples, ColorSample):
            sample_identifiers.append(self.samples.identifier)
        else:
            for sample in self.samples:
                sample_identifiers.append(sample.identifier)

        if include_edges:
            self.edges.save(output_dir)

        json_output_path = os.path.join(output_dir, "color_distribution.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "identifier": self.identifier,
                "equally_weighted": self.equally_weighted,
                "sample_identifiers": sample_identifiers,
            }, file)

        hist_output_path = os.path.join(output_dir, "color_distribution_histograms.npz")
        np.savez(hist_output_path, rgb_hist=self.rgb_hist, h_hist=self.h_hist, v_hist=self.v_hist, oklab_hist=self.oklab_hist, uv_hist=self.uv_hist, allow_pickle=False)

    @classmethod
    def load(cls, input_dir: str, edges: ColorHistogramEdges | None = None):
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.
            edges: Color histogram edges. If None, the edges will be loaded from input_dir.

        Returns:
            A ColorDistribution.
        """

        if edges is None:
            edges = ColorHistogramEdges.load(input_dir)

        json_input_path = os.path.join(input_dir, "color_distribution.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        cd = cls(None, identifier=json_data["identifier"], equally_weighted=json_data["equally_weighted"])
        cd.edges = edges

        hist_input_path = os.path.join(input_dir, "color_distribution_histograms.npz")
        with np.load(hist_input_path, allow_pickle=False) as data:
            cd.rgb_hist, cd.h_hist, cd.s_hist, cd.v_hist, cd.oklab_hist, cd.uv_hist = data["rgb_hist"], data["h_hist"], data["v_hist"], data["oklab_hist"], data["uv_hist"]

        return cd

    def get_h_hist(self) -> np.ndarray:
        """Get the hue (HSV) histogram.

        Returns:
            A np.ndarray histogram (returns as a density instead of counts).
        """
        return self.h_hist

    def get_s_hist(self) -> np.ndarray:
        """Get the saturation (HSV) histogram.

        Returns:
            A np.ndarray histogram (returns as a density instead of counts).
        """
        return self.s_hist

    def get_v_hist(self) -> np.ndarray:
        """Get the value (HSV) histogram.

        Returns:
            A np.ndarray histogram (returns as a density instead of counts).
        """
        return self.v_hist

    def get_rgb_hist(self) -> np.ndarray:
        """Get the RGB histogram.

        Returns:
            A np.ndarray histogram (returns as a density instead of counts).
        """
        return self.rgb_hist

    def get_oklab_hist(self) -> np.ndarray:
        """Get the OKLAB histogram.

        Returns:
            A np.ndarray histogram (returns as a density instead of counts).
        """
        return self.oklab_hist

    def get_uv_hist(self) -> np.ndarray:
        """Get the u'v' histogram.

        Returns:
            A np.ndarray histogram (returns as a density instead of counts).
        """
        return self.uv_hist

    def compute(self):
        """Compute the ColorDistribution.
        """

        samples = self.samples
        if isinstance(self.samples, ColorSample):
            samples = [self.samples]
        
        for sample in samples:

            self.h_hist += sample.get_h_hist(density=self.equally_weighted)
            self.s_hist += sample.get_s_hist(density=self.equally_weighted)
            self.v_hist += sample.get_v_hist(density=self.equally_weighted)
            self.rgb_hist += sample.get_rgb_hist(density=self.equally_weighted)
            self.oklab_hist += sample.get_oklab_hist(density=self.equally_weighted)
            self.uv_hist += sample.get_uv_hist(density=self.equally_weighted)
                
        self.h_hist = self.h_hist / np.sum(self.h_hist)
        self.s_hist = self.s_hist / np.sum(self.s_hist)
        self.v_hist = self.v_hist / np.sum(self.v_hist)
        self.rgb_hist = self.rgb_hist / np.sum(self.rgb_hist)
        self.oklab_hist = self.oklab_hist / np.sum(self.oklab_hist)
        self.uv_hist = self.uv_hist / np.sum(self.uv_hist)

        # Smooth
        self.h_hist = gaussian_filter(self.h_hist, sigma=2.0)
        self.h_hist = self.h_hist / np.sum(self.h_hist)

        self.s_hist = gaussian_filter(self.s_hist, sigma=2.0)
        self.s_hist = self.s_hist / np.sum(self.s_hist)

        self.v_hist = gaussian_filter(self.v_hist, sigma=2.0)
        self.v_hist = self.v_hist / np.sum(self.v_hist)

        self.rgb_hist = gaussian_filter(self.rgb_hist, sigma=0.5)
        self.rgb_hist = self.rgb_hist / np.sum(self.rgb_hist)

        self.oklab_hist = gaussian_filter(self.oklab_hist, sigma=0.5)
        self.oklab_hist = self.oklab_hist / np.sum(self.oklab_hist)

        self.uv_hist = gaussian_filter(self.uv_hist, sigma=0.5)
        self.uv_hist = self.uv_hist / np.sum(self.uv_hist)

class ColorComparison:

    def __init__(self, reference: ColorDistribution | str, query: ColorSample | str):
        """Compute a comparison between a ColorDistribution and a ColorSample:

        Args:
            reference: The reference ColorDistribution. This parameter can be set to a str if the only use of this object is to store attributes and not to compute.
            query: The query ColorSample. This parameter can be set to a str if the only use of this object is to store attributes and not to compute.
        """
        
        if not isinstance(reference, ColorDistribution) and not isinstance(reference, str):
            raise Exception("reference should be a ColorDistribution")
        if not isinstance(query, ColorSample) and not isinstance(query, str):
            raise Exception("query should be a ColorSample")

        self.reference = reference
        self.query = query

        self.h_dist = None
        self.s_dist = None
        self.v_dist = None
        self.rgb_dist = None
        self.oklab_dist = None

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        json_output_path = os.path.join(output_dir, "color_comparison.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "reference": self.reference.identifier,
                "query": self.query.identifier,
                "h_dist": self.h_dist,
                "s_dist": self.s_dist,
                "v_dist": self.v_dist,
                "rgb_dist": self.rgb_dist,
                "oklab_dist": self.oklab_dist,
            }, file)

    @classmethod
    def load(cls, input_dir: str) -> ColorComparison:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A ColorComparison.
        """

        json_input_path = os.path.join(input_dir, "color_comparison.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        cc = cls(json_data["reference"], json_data["query"])
        cc.h_dist, cc.s_dist, cc.v_dist, cc.rgb_dist, cc.oklab_dist = json_data["h_dist"], json_data["s_dist"], json_data["v_dist"], json_data["rgb_dist"], json_data["oklab_dist"]

        return cc

    def get_val(self, val_type: str) -> float:
        """Get a metric/statistic.

        Options for ``val_type``:
            * "h": Hue (HSV) distance.
            * "s": Saturation (HSV) distance.
            * "v": Value (HSV) distance.
            * "rgb": RGB distance.
            * "oklab": OKLAB distance.

        Args:
            val_type: The name of the metric/statistic to get.

        Returns:
            The requested metric/statistic.
        """
        if val_type == "h":
            return self.h_dist
        elif val_type == "s":
            return self.s_dist
        elif val_type == "v":
            return self.v_dist
        elif val_type == "rgb":
            return self.rgb_dist
        elif val_type == "oklab":
            return self.oklab_dist
        else:
            raise Exception(f"{val_type} is not a valid value type.")

    @property
    def query_identifier(self) -> str:
        """
        Returns:
            The ID of the query.
        """
        return self.query.identifier

    @property
    def reference_identifier(self) -> str:
        """
        Returns:
            The ID of the reference.
        """
        return self.reference.identifier

    def dist_metric(self, reference_hist: np.ndarray, query_hist: np.ndarray) -> float:
        """Compute the distance between two histograms.

        Args:
            reference_hist: The reference histogram.
            query_hist: The query histogram.

        Returns:
            The distance between two histograms.
        """
        return self.bhattacharyya(reference_hist, query_hist)

    def bhattacharyya(self, reference_hist: np.ndarray, query_hist: np.ndarray) -> float:
        """Compute the Bhattacharyya distance between two histograms.

        Args:
            reference_hist: The reference histogram.
            query_hist: The query histogram.

        Returns:
            The Bhattacharyya distance between two histograms.
        """
        BC = np.sum(np.sqrt(reference_hist * query_hist))
        Db = -1.0 * np.log(BC)
        return Db

    def compute(self):
        """Compute the ColorComparison.
        """
        
        print(f"Comparing {self.query.identifier} to {self.reference.identifier}...")
        
        self.h_dist = self.dist_metric(self.reference.get_h_hist(), self.query.get_h_hist(density=True))
        self.s_dist = self.dist_metric(self.reference.get_s_hist(), self.query.get_s_hist(density=True))
        self.v_dist = self.dist_metric(self.reference.get_v_hist(), self.query.get_v_hist(density=True))
        self.rgb_dist = self.dist_metric(self.reference.get_rgb_hist(), self.query.get_rgb_hist(density=True))
        self.oklab_dist = self.dist_metric(self.reference.get_oklab_hist(), self.query.get_oklab_hist(density=True))
