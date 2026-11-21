from stainnormprogram.slide import Tile, SlideImage

import pyvips
import numpy as np
import colour
from fastdigest import TDigest
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

def otsu_threshold_foreground(tile: Tile, flatten: bool = False):
    """Compute a foreground tissue mask using Otsu's thresholding.

    Args:
        tile: Image tile.
        flatten: Output a flattened version of the tile.

    Returns:
        A np.ndarray for the foreground mask (True = tissue).
    """

    gray_tile = tile.gray()
    if flatten:
        gray_tile = gray_tile.reshape(-1)
    thresh = threshold_otsu(gray_tile)
    fg_mask = gray_tile < thresh
    return fg_mask

def hsv_threshold_foreground(tile: Tile, s_threshold: float = 0.05, v_threshold: float = 0.15, flatten: bool = False):
    """Compute the foreground tissue mask using HSV thresholds.

    HSV pixels with saturation greater than s_threshold and value (brightness) greater than v_threshold are considered foreground.
    
    Args:
        tile: Image tile.
        s_threshold: HSV saturation threshold [0., 1.] where 0. is no saturation.
        v_threshold: HSV value threshold [0., 1.] where 0. is black.
        flatten: Output a flattened version of the tile.

    Returns:
        A np.ndarray for the foreground mask (True = tissue).
    """
    hsv_tile = tile.hsv()
    if flatten:
        hsv_tile = hsv_tile.reshape(-1,3)
    s = hsv_tile[...,1]
    v = hsv_tile[...,2]
    fg_mask = (s > s_threshold) & (v > v_threshold)
    return fg_mask

class HSVBackgroundEstimation:

    def __init__(self, slide: SlideImage, s_threshold: float = 0.05, v_threshold: float = 0.5):
        """Estimate the background color using HSV thresholds.
        
        HSV pixels with saturation less than s_threshold and value (brightness) greater than v_threshold are used in the background estimation.

        Args:
            slide: Image to estimate the background of.
            s_threshold: HSV saturation threshold [0., 1.] where 0. is no saturation.
            v_threshold: HSV value threshold [0., 1.] where 0. is black.
        """

        self.image = slide
        self.s_threshold = s_threshold
        self.v_threshold = v_threshold

    def compute(self, method: str, options: dict[str, Any] = {}):
        """Perform the background color estimation.

        Args:
            method: "tiled" for estimating from tiles, "pyvips" for estimating from the entire image via pyvips, and "thumbnail" for estimating from a thumbnail.
            options: Options for the specific compute method.

        Returns:
            A dictionary containing the requested computations.
        """
        print(f"Computing background estimation for {self.image.identifier}...")

        if method == "tiled":
            return self.tiled_compute(**options)
        elif method == "pyvips":
            return self.pyvips_compute(**options)
        elif method == "thumbnail":
            return self.thumbnail_compute(**options)
        else:
            raise Exception(f"{method} is not a valid method.")

    def thumbnail_compute(self, height: int = 4096, smooth: bool = True, use_otsu: bool = False):
        """Compute the background using a thumbnail.

        Args:
            height: Max height of the thumbnail.
            smooth: Allow for interpolation that performs pixel smoothing.
            use_otsu: Use Otsu's thresholding to identify the background.

        Returns:
            A dictionary containing the median background color.
        """
        pixels = self.image.get_thumbnail(height=height, smooth=smooth) # (H,W,3) RGB uint8

        if use_otsu:
            otsu_mask = self.image.get_thumbnail_mask(otsu_threshold_foreground) # (H,W) True = foreground
            candidate_bgd_pixels = pixels[~otsu_mask] # (N,3)
        else:
            candidate_bgd_pixels = pixels.reshape((-1,3))

        hsv_pixels = rgb2hsv(candidate_bgd_pixels / 255.)
        hsv_mask = (hsv_pixels[:,1] < self.s_threshold) & (hsv_pixels[:,2] > self.v_threshold) # (N,) True = background

        bgd_pixels = candidate_bgd_pixels[hsv_mask]

        I0 = np.median(bgd_pixels, axis=0)

        return {"median": I0}

    def tiled_compute(self, percentile: int = 50, max_pixels: int = 10_000_000, tile_size: int = 256, stability_check_interval: int = 10, stability_window: int = 3):
        """Compute the background using full resolution tiles.

        Args:
            percentile: Background percentile to compute (50 = median).
            max_pixels: Max number of pixels to use.
            tile_size: Tile size to use.
            stability_check_interval: Check for stability every ``stability_check_interval`` number of updates to the running background color.
            stability_window: Number of stability checks that must pass before finishing the computation.

        Returns:   
            A dictionary containing the requested background color.
        """

        # Tile size
        self.image.set_tile_size(tile_size)

        digests = [TDigest(), TDigest(), TDigest()]
        self.image.set_mode("random")
        n_pixels = 0
        n_updates = 0
        n_stable = 0

        I0 = [None, None, None]

        for i, (tile, _) in enumerate(self.image):

            if i % 10 == 0:
                print(f"Processed {i} tiles so far...")

            tile_rgb = tile.rgb(as_uint8=True).reshape(-1,3) # (N,3)
            tile_hsv = tile.hsv().reshape(-1,3)

            tile_s = tile_hsv[:,1]
            tile_v = tile_hsv[:,2]
            mask = (tile_s < self.s_threshold) & (tile_v > self.v_threshold)

            filtered = tile_rgb[mask] # (M,3)

            if filtered.size == 0:
                continue

            n_pixels += filtered.shape[0]

            for i in range(3):
                digests[i].batch_update(filtered[:,i].tolist())
            n_updates += 1

            # Absolute stopping condition
            if n_pixels > max_pixels:
                break

            # Check for convergence
            if n_updates % stability_check_interval == 0:
                same = True
                for i in range(3):
                    same = same and (I0[i] == round(digests[i].percentile(percentile)))
                    I0[i] = round(digests[i].percentile(percentile))
                if same:
                    n_stable += 1
                else:
                    n_stable = 0
                if n_stable >= stability_window:
                    break

        for i in range(3):
            I0[i] = round(digests[i].percentile(percentile))
        I0 = np.array(I0).clip(0, 255).astype(np.uint8)

        if percentile == 50:
            return {"median": I0}
        else:
            return {str(percentile): I0}

    def _compute_percentile(self, masked_rgb: pyvips.Image, percentile: int):
        """Compute the RGB value at a certain percentile for a pyvips image.

        Args:
            masked_rgb: Masked (non-background should be 0, background should be the original RGB value) pyvips image.
            percentile: Percentile to compute (50 = median).

        Returns:
            A np.ndarray of the computed RGB pixel value.
        """
        
        percentile_rgb = []
        for i in range(3):
            channel_hist = masked_rgb[i].hist_find().numpy()
            channel_hist = np.squeeze(channel_hist)
            channel_hist[0] = 0
            cdf = 100.0 * np.cumsum(channel_hist) / np.sum(channel_hist)
            val = np.searchsorted(cdf, percentile)
            percentile_rgb.append(val.item())
        percentile_rgb = np.array(percentile_rgb)
        return percentile_rgb

    def pyvips_compute(self, measures: list[str | int] = ["median"], downsample_factor: float = 1.0, downsample_kernel: str = "nearest", **kwargs):
        """Compute the background using a pyvips image.

        Args:
            measures: Measures to compute ("median", "mean", "iqr", "std", or an int for percentile).
            downsample_factor: Factor to downsample the image by. Must be in the interval (0.0, 1.0].
            downsample_kernel: Kernel to use for downsampling ("nearest", "linear", "cubic", or any other pyvips kernel).

        Returns:
            A dictionary containing the requested computations.
        """

        image = self.image.get_pyvips()

        if not isinstance(measures, list):
            measures = [measures]

        # Downsample
        if downsample_factor <= 0.0 or downsample_factor > 1.0:
            raise Exception("downsample_factor must be in the interval (0.0, 1.0]")
        if downsample_factor < 1.0:
            image = image.resize(downsample_factor, kernel=downsample_kernel)
        
        # Convert to HSV
        hsv_image = image.colourspace("hsv")

        # Threshold
        s_image = hsv_image[1]
        v_image = hsv_image[2]

        mask = (s_image < self.s_threshold * 255) & (v_image > self.v_threshold * 255)
        masked_rgb = mask.ifthenelse(image, 0)
        area = mask.height * mask.width
        mask_sum = mask.avg() * area

        results = {}
        for measure in measures:
            if measure == "mean":
                mean_rgb = []
                for i in range(3):
                    mean_val = masked_rgb[i].avg() * area / mask_sum * 255
                    mean_rgb.append(mean_val)
                mean_rgb = np.array(mean_rgb)
                results["mean"] = mean_rgb
            elif measure == "std":
                std_rgb = []
                for i in range(3):
                    std_val = (masked_rgb[i].deviate() ** 2 * area / mask_sum) ** 0.5
                    std_rgb.append(std_val)
                std_rgb = np.array(std_rgb)
                results["std"] = std_rgb
            elif measure == "median":
                results["median"] = self._compute_percentile(masked_rgb, 50)
            elif measure == "iqr":
                results["iqr"] = self._compute_percentile(masked_rgb, 75) - self._compute_percentile(masked_rgb, 25)
            elif isinstance(measure, int):
                results[measure] = self._compute_percentile(masked_rgb, measure)
            else:
                raise Exception(f"{measure} is not a valid measure")
            
        return results
        