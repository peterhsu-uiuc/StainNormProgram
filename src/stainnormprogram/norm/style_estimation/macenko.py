from stainnormprogram.slide import Tile
from stainnormprogram.norm.utils import nnls
from stainnormprogram.norm.background import hsv_threshold_foreground

import math
import time
import multiprocessing as mp
import queue
import os
import random

import pyvips
import numpy as np
from sklearn.linear_model import LinearRegression
import colour
from skimage.color import rgb2hsv

DAB_HUE = 30.0 / 360.0 # orange
HTX_HUE = 240.0 / 360.0 # blue

class MacenkoStyleEstimation:

    def __init__(self,
                slide: SlideImage,
                I0: np.ndarray,
                stain_balancing: bool = False,
                mode: str = "tiled_parallel",
                alpha: float = 1.0,
                beta: float = 0.15,
                use_nnls: bool = True,
                maxworkers: int | None = None,
                options: dict[str, Any] = {}
                ):
        """Estimate the stain style using Macenko's method.

        Args:
            slide: Slide to perform estimation on.
            I0: Background intensity.
            stain_balancing: Whether to perform stain balancing (True is recommended for DAB).
            mode: Estimation mode ("tiled", "tiled_parallel", or "thumbnail").
            alpha: Macenko's alpha.
            beta: Macenko's beta.
            use_nnls: Use non-negative least squares when calculating concentrations.
            maxworkers: Max workers for style estimation (None calculates this for you).
            options: Additional options to pass to the estimation methods.
        """
        self.slide = slide
        self.I0 = I0
        self.stain_balancing = stain_balancing
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.use_nnls = use_nnls
        self.maxworkers = maxworkers
        self.options = options
        
        if "tiled" in self.mode and "threshold" not in options:
            if not stain_balancing:
                self.options["threshold"] = 200_000
            else:
                self.options["threshold"] = (50_000, 50_000)

        self.matrix_avg = None
        self.maxC_avg = None

    def compute_matrix(self):
        """Compute the stain matrix.
        """
        if self.mode == "tiled_parallel":
            self.compute_tiled_parallel(quantity="matrix", **self.options)
        elif self.mode == "thumbnail":
            self.compute_matrix_thumbnail(**self.options)
        elif self.mode == "tiled":
            self.compute_tiled(quantity="matrix", **self.options)
        else:
            raise Exception(f"{self.mode} is not a valid mode.")

    def compute_maxC(self):
        """Compute maxC.
        """
        if self.mode == "tiled_parallel":
            self.compute_tiled_parallel(quantity="maxC", **self.options)
        elif self.mode == "thumbnail":
            self.compute_maxC_thumbnail(**self.options)
        elif self.mode == "tiled":
            self.compute_tiled(quantity="maxC", **self.options)
        else:
            raise Exception(f"{self.mode} is not a valid mode.")

    def set_matrix_avg(self, matrix_avg: np.ndarray):
        """Set the average stain matrix (useful if you want to just calculate maxC).
        """
        self.matrix_avg = matrix_avg

    @staticmethod
    def get_buffer_assignment(pixels: np.ndarray) -> np.ndarray:
        """Assign pixels to either the hematoxylin (False) or DAB (True) buffers.

        Args:
            pixels: (N,3) in HSV

        Returns:
            A np.ndarray (N,) where hematoyxlin pixels are False and DAB pixels are True.
        """
        hues = pixels[:,0]

        dab_abs_dists = np.abs(hues - DAB_HUE)
        dab_dists = np.minimum(dab_abs_dists, 1.0 - dab_abs_dists)

        htx_abs_dists = np.abs(hues - HTX_HUE)
        htx_dists = np.minimum(htx_abs_dists, 1.0 - htx_abs_dists)

        return dab_dists < htx_dists

    @staticmethod
    def convert_od_and_filter(tile: Tile, I0: np.ndarray, beta: float, return_mask: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Convert RGB pixels into optical density (OD) pixels, then filter out pixels using the beta threshold.

        Args:
            tile: Tile image.
            I0: Initial intensity.
            beta: Macenko's beta.
            return_mask: Return the filter mask. True pixels in the mask correspond to pixels that were accepted.

        Returns:
            A np.ndarray with accepted OD pixels, along with a np.ndarray bool mask if return_mask is True.
        """

        tile_od = tile.od(I0).reshape(-1,3)

        beta_mask = np.any(tile_od > beta, axis=1)
        hsv_mask = hsv_threshold_foreground(tile, flatten=True)
        mask = beta_mask & hsv_mask

        filtered = tile_od[mask]
        
        if return_mask:
            return filtered, mask

        return filtered

    @staticmethod
    def svd(ODhat: np.ndarray, alpha: float) -> np.ndarray:
        """Compute Macenko's singular value decomposition.

        Code taken from https://github.com/slideflow/slideflow/blob/master/slideflow/norm/macenko.py.

        Args:
            ODhat: Filtered OD pixel values.
            alpha: Macenko's alpha.

        Returns:
            A np.ndarray representing the stain matrix.
        """

        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        That = ODhat.dot(eigvecs[:, 1:3])
    
        phi = np.arctan2(That[:, 1], That[:, 0])
    
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
    
        vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
        if vMin[0] > vMax[0]:
            matrix = np.array((vMin[:, 0], vMax[:, 0])).T
        else:
            matrix = np.array((vMax[:, 0], vMin[:, 0])).T

        return matrix

    def get_thumbnail_pixels(self, height: int = 4096, smooth: bool = True) -> np.ndarray:
        """Get filtered OD pixels from the thumbnail image.

        Args:
            height: Max height of the thumbnail.
            smooth: Allow for interpolation that involves pixel smoothing.

        Returns:
            A np.ndarray with accepted OD pixels.
        """
        image = self.slide.get_thumbnail(height=height, smooth=smooth)
        tile = Tile(image)
        filtered = self.convert_od_and_filter(tile, self.I0, self.beta)

        if filtered.shape[0] == 0:
            raise Exception("All pixels were filtered out during style estimation. Adjust your parameters or check this isn't a blank image.")

        return filtered

    def compute_matrix_thumbnail(self, height: int = 4096, smooth: bool = True, **kwargs):
        """Compute the stain matrix from a thumbnail of the whole slide image.

        Args:
            height: Max height of the thumbnail.
            smooth: Allow for interpolation that involves pixel smoothing.
        """
        filtered = self.get_thumbnail_pixels(height=height, smooth=smooth)
        self.matrix_avg = self.svd(filtered, self.alpha)

    def compute_maxC_thumbnail(self, height: int = 4096, smooth: bool = True, **kwargs):
        """Compute maxC from a thumbnail of the whole slide image.

        Args:
            height: Max height of the thumbnail.
            smooth: Allow for interpolation that involves pixel smoothing.
        """

        filtered = self.get_thumbnail_pixels(height=height, smooth=smooth)

        vals = filtered.transpose() # (3,N)
        if self.use_nnls:
            C = nnls(self.matrix_avg, vals) # (2,N)
        else:
            C = np.linalg.lstsq(self.matrix_avg, vals, rcond=None)[0]
        maxC = np.array([np.percentile(C[0, :], 100 - self.alpha), np.percentile(C[1, :], 100 - self.alpha)])

        self.maxC_avg = maxC

    @staticmethod
    def compute_task(
                    quantity: str,
                    slide: SlideImage,
                    coord_queue: mp.Queue,
                    quantity_queue: mp.Queue,
                    I0: np.ndarray,
                    alpha: float,
                    beta: float,
                    threshold: int | tuple[int, int],
                    remainder_queue: mp.Queue,
                    stop_event: mp.Event,
                    use_nnls: bool = True,
                    matrix_avg: np.array | None = None,
                    ):
        """Perform either stain matrix or maxC computation. This is a task function given to workers for parallel processing.

        Args:
            quantity: Either "matrix" or "maxC" depending on the quantity of interest to calculate.
            slide: SlideImage to perform the computation on.
            coord_queue: Queue containing tile coordinates (upper left corner) to process.
            quantity_queue: Queue to put output computations into.
            I0: Initial intensity.
            alpha: Macenko's alpha.
            beta: Macenko's beta.
            threshold: Number of pixels to accumulate before executing the computation. Tuple of two ints (HTX, DAB) if stain balancing.
            remainder_queue: Queue to put leftover pixels into if the worker is at the end of life and doesn't have enough pixels to do another computation.
            stop_event: Stop event called when workers should stop.
            use_nnls: Use non-negative least squares for calculating concentrations.
            matrix_avg: Computed stain matrix (if maxC is being computed).
        """

        stain_balancing = isinstance(threshold, tuple)

        if stain_balancing:
            pixels = [0, 0]
            buffer = [[], []]
        else:
            pixels = 0
            buffer = []
        
        slide.load_slide()

        while not stop_event.is_set():
            try:
                y, x, tile_h, tile_w = coord_queue.get_nowait()
            except queue.Empty:
                break

            tile = slide.get_tile(y, x, tile_h, tile_w)

            filtered, mask = MacenkoStyleEstimation.convert_od_and_filter(tile, I0, beta, return_mask=True)

            if filtered.shape[0] > 0:

                if stain_balancing:
                    tile_hsv = tile.hsv().reshape(-1,3)
                    filtered_hsv = tile_hsv[mask]
                    buffer_assignment = MacenkoStyleEstimation.get_buffer_assignment(filtered_hsv)
                    htx = filtered[~buffer_assignment]
                    dab = filtered[buffer_assignment]
                    if len(htx.shape) != 2:
                        htx = htx[np.newaxis, :]
                    if len(dab.shape) != 2:
                        dab = dab[np.newaxis, :]
                    if htx.shape[0] > 0 and pixels[0] <= threshold[0]:
                        buffer[0].append(htx)
                        pixels[0] += htx.shape[0]
                    if dab.shape[0] > 0 and pixels[1] <= threshold[1]:
                        buffer[1].append(dab)
                        pixels[1] += dab.shape[0]
                else:
                    buffer.append(filtered)
                    pixels += filtered.shape[0]

            if not stain_balancing and pixels > threshold:
                    
                vals = np.concatenate(buffer, axis=0)
                buffer = []
                pixels = 0
                if quantity == "matrix":
                    matrix = MacenkoStyleEstimation.svd(vals, alpha)
                    quantity_queue.put(matrix)
                else:
                    vals = vals.transpose()
                    if use_nnls:
                        C = nnls(matrix_avg, vals) # (2,N)
                    else:
                        C = np.linalg.lstsq(matrix_avg, vals, rcond=None)[0]
                    maxC = np.array([np.percentile(C[0, :], 100 - alpha), np.percentile(C[1, :], 100 - alpha)])
                    quantity_queue.put(maxC)

            if stain_balancing and pixels[0] > threshold[0] and pixels[1] > threshold[1]:

                htx_vals = np.concatenate(buffer[0], axis=0) # (N_htx,3)
                dab_vals = np.concatenate(buffer[1], axis=0) # (N_dab,3)
                vals = np.concatenate([htx_vals, dab_vals], axis=0) # (N,3)
                pixels = [0, 0]
                buffer = [[], []]
                if quantity == "matrix":
                    matrix = MacenkoStyleEstimation.svd(vals, alpha)
                    quantity_queue.put(matrix)
                else:
                    vals = vals.transpose()
                    if use_nnls:
                        C = nnls(matrix_avg, vals) # (2,N)
                    else:
                        C = np.linalg.lstsq(matrix_avg, vals, rcond=None)[0]
                    maxC = np.array([np.percentile(C[0, :], 100 - alpha), np.percentile(C[1, :], 100 - alpha)])
                    quantity_queue.put(maxC)

        remainder_queue.put(buffer)
        quantity_queue.put(None)

    def compute_tiled_parallel(self, threshold: int | tuple[int, int], quantity: str, num_tiles: int = 500, tile_size: int = 512, detect_fg: bool = False, maxC_mode: str = "mean", **kwargs):
        """Compute stain matrix or maxC using tiles, parallelized.

        Args:
            threshold: Min number of pixels to accumulate before calculating the quantity. If int, there is one threshold. If tuple of two ints, the first is for hematoxylin and second for DAB.
            quantity: Either "matrix" or "maxC" depending on the quantity of interest to calculate.
            num_tiles: Max number of tiles to use for estimation.
            tile_size: Tile size to use.
            detect_fg: Detect where the tissue is using the thumbnail.
            maxC_mode: Type of statistic to calculate over maxC values ("mean" or "percentile" for 100 - alpha percentile of maxC values).
        """

        print(f"Computing stain {quantity}...")
        print(f"Min pixels to accumulate before calculating = {threshold}")
        print(f"Max tiles to use = {num_tiles}")
        print(f"Tile size = {tile_size}")
        print(f"Detect foreground from thumbnail first = {detect_fg}")
        if quantity == "maxC":
            print(f"maxC estimates are aggregated via = {maxC_mode}")

        # Input checks
        if not (isinstance(threshold, int) or (isinstance(threshold, tuple) and len(threshold) == 2)):
            raise Exception("threshold needs to be an int or tuple of two ints")
        if not (quantity == "matrix" or quantity == "maxC"):
            raise Exception("quantity needs to be either matrix or maxC")
        if quantity == "maxC" and self.matrix_avg is None:
            raise Exception("Stain matrix needs to be computed before maxC")

        # Set tile size
        self.slide.set_tile_size(tile_size)

        # Set number of workers based on CPU cores
        num_workers = 1
        if self.maxworkers is None:
            num_workers = max(1, (os.process_cpu_count() or 1) - 1)
        else:
            num_workers = self.maxworkers
        print(f"Using {num_workers} workers...")

        # Get coordinates to visit
        coords = self.slide.compute_random_sequence()
        if detect_fg:
            coords = self.slide.filter_coords_using_thumbnail(hsv_threshold_foreground, coords)

        # Put coordinates into a queue
        coord_queue = mp.Queue()
        for c in coords:
            coord_queue.put(c)

        # Make slide pickleable
        self.slide.pickleable()

        quantity_queue = mp.Queue() # Queue for storing computed stain quantities
        remainder_queue = mp.Queue() # Queue for storing leftover pixels
        stop_event = mp.Event()

        # Give workers tasks
        workers = []
        for i in range(num_workers):
            p = mp.Process(target=self.compute_task, args=(quantity, self.slide, coord_queue, quantity_queue, self.I0, self.alpha, self.beta, threshold, remainder_queue, stop_event, self.use_nnls, self.matrix_avg))
            p.start()
            workers.append(p)

        # Accumulate matrices
        quantity_buffer = []
        tiles = 0
        finished_workers = 0
        try:
            while finished_workers < num_workers and len(quantity_buffer) < num_tiles:
                q = quantity_queue.get()
                if q is None:
                    finished_workers += 1
                else:
                    quantity_buffer.append(q)
                    tiles += 1
                    if tiles % 100 == 0:
                        print(f"Processed {tiles} tiles so far...")
        finally:
            stop_event.set()

            # Handle any remaining pixels
            stain_balancing = isinstance(threshold, tuple)
            if stain_balancing:
                remainder_buffer = [[], []]
                pixels = [0, 0]
            else:
                remainder_buffer = []
                pixels = 0
            used_remainder = False
            for i in range(num_workers):
                try:
                    remainder = remainder_queue.get(timeout=2.0)

                    if stain_balancing:

                        if len(remainder[0]) == 0 and len(remainder[1]) == 0:
                            continue
                        remainder[0] = np.concatenate(remainder[0], axis=0)
                        remainder[1] = np.concatenate(remainder[1], axis=0)
                        pixels[0] += remainder[0].shape[0]
                        pixels[1] += remainder[1].shape[0]
                        remainder_buffer[0].append(remainder[0])
                        remainder_buffer[1].append(remainder[1])

                        if (pixels[0] > threshold[0] and pixels[1] > threshold[1]) or (i + 1 == num_workers and len(remainder_buffer[0]) > 0 and len(remainder_buffer[1]) > 0 and not used_remainder):
                            htx_vals = np.concatenate(remainder_buffer[0], axis=0) # (N_htx,3)
                            dab_vals = np.concatenate(remainder_buffer[1], axis=0) # (N_dab,3)
                            vals = np.concatenate([htx_vals, dab_vals], axis=0) # (N,3)
                            remainder_buffer = [[], []]
                            pixels = [0, 0]
                            if quantity == "matrix":
                                matrix = MacenkoStyleEstimation.svd(vals, self.alpha)
                                quantity_buffer.append(matrix)
                            else:
                                vals = vals.transpose()
                                if self.use_nnls:
                                    C = nnls(self.matrix_avg, vals) # (2,N)
                                else:
                                    C = np.linalg.lstsq(self.matrix_avg, vals, rcond=None)[0]
                                maxC = np.array([np.percentile(C[0, :], 100 - self.alpha), np.percentile(C[1, :], 100 - self.alpha)])
                                quantity_buffer.append(maxC)
                            used_remainder = True
                    
                    else:

                        if len(remainder) == 0:
                            continue
                        remainder = np.concatenate(remainder, axis=0)
                        pixels += remainder.shape[0]
                        remainder_buffer.append(remainder)

                        if pixels > threshold or (i + 1 == num_workers and len(remainder_buffer) > 0 and not used_remainder):
                            vals = np.concatenate(remainder_buffer, axis=0)
                            remainder_buffer = []
                            pixels = 0
                            if quantity == "matrix":
                                matrix = MacenkoStyleEstimation.svd(vals, self.alpha)
                                quantity_buffer.append(matrix)
                            else:
                                vals = vals.transpose()
                                if self.use_nnls:
                                    C = nnls(self.matrix_avg, vals) # (2,N)
                                else:
                                    C = np.linalg.lstsq(self.matrix_avg, vals, rcond=None)[0]
                                maxC = np.array([np.percentile(C[0, :], 100 - self.alpha), np.percentile(C[1, :], 100 - self.alpha)])
                                quantity_buffer.append(maxC)
                            used_remainder = True

                except queue.Empty:
                    pass

            for w in workers:
                w.join(timeout=1.0)
                if w.is_alive():
                    w.terminate()
            
            if quantity == "matrix":
                self.matrix_buffer = quantity_buffer
                self.matrix_avg = np.mean(quantity_buffer, axis=0)
            else:
                self.maxC_buffer = quantity_buffer
                if maxC_mode == "mean":
                    self.maxC_avg = np.mean(quantity_buffer, axis=0)
                elif maxC_mode == "percentile":
                    self.maxC_avg = np.percentile(quantity_buffer, 100 - self.alpha, axis=0)

            # Re-load slide
            self.slide.load_slide()

        # Close queues
        coord_queue.close()
        coord_queue.cancel_join_thread()
        quantity_queue.close()
        quantity_queue.cancel_join_thread()
        remainder_queue.close()
        remainder_queue.cancel_join_thread()


    def compute_tiled(self, threshold: int | tuple[int, int], quantity: str, num_tiles: int = 500, tile_size: int = 512, detect_fg: bool = False, maxC_mode: str = "mean", **kwargs):
        """Compute stain matrix or maxC using tiles (but not in parallel).

        Args:
            threshold: Min number of pixels to accumulate before calculating the quantity. If int, there is one threshold. If tuple of two ints, the first is for hematoxylin and second for DAB.
            quantity: Either "matrix" or "maxC" depending on the quantity of interest to calculate.
            num_tiles: Max number of tiles to use for estimation.
            tile_size: Tile size to use.
            detect_fg: Detect where the tissue is using the thumbnail.
            maxC_mode: Type of statistic to calculate over maxC values ("mean" or "percentile" for 100 - alpha percentile of maxC values).
        """

        print(f"Computing stain {quantity}...")
        print(f"Min pixels to accumulate before calculating = {threshold}")
        print(f"Max tiles to use = {num_tiles}")
        print(f"Tile size = {tile_size}")
        print(f"Detect foreground from thumbnail first = {detect_fg}")
        if quantity == "maxC":
            print(f"maxC estimates are aggregated via = {maxC_mode}")

        # Input checks
        if not (isinstance(threshold, int) or (isinstance(threshold, tuple) and len(threshold) == 2)):
            raise Exception("threshold needs to be an int or tuple of two ints")
        if not (quantity == "matrix" or quantity == "maxC"):
            raise Exception("quantity needs to be either matrix or maxC")
        if quantity == "maxC" and self.matrix_avg is None:
            raise Exception("Stain matrix needs to be computed before maxC") 

        # Set tile size
        self.slide.set_tile_size(tile_size)

        # Foreground detection
        if detect_fg:
            coords = self.slide.compute_random_sequence()
            self.slide.filter_coords_using_thumbnail(hsv_threshold_foreground, coords)
            self.slide.set_mode("custom")
        else:
            self.slide.set_mode("random")

        stain_balancing = isinstance(threshold, tuple)

        if stain_balancing:
            buffer = [[], []]
            pixels = [0, 0]
        else:
            buffer = []
            pixels = 0

        tiles = 0
        quantity_buffer = []

        iter_len = len(self.slide)

        for i, (tile, _) in enumerate(self.slide):

            filtered = self.convert_od_and_filter(tile, self.I0, self.beta)

            if filtered.shape[0] > 0:

                if stain_balancing:
                    tile_hsv = tile.hsv().reshape(-1,3)
                    filtered_hsv = tile_hsv[mask]
                    buffer_assignment = MacenkoStyleEstimation.get_buffer_assignment(filtered_hsv)
                    htx = filtered[~buffer_assignment]
                    dab = filtered[buffer_assignment]
                    if len(htx.shape) != 2:
                        htx = htx[np.newaxis, :]
                    if len(dab.shape) != 2:
                        dab = dab[np.newaxis, :]
                    if htx.shape[0] > 0 and pixels[0] <= threshold[0]:
                        buffer[0].append(htx)
                        pixels[0] += htx.shape[0]
                    if dab.shape[0] > 0 and pixels[1] <= threshold[1]:
                        buffer[1].append(dab)
                        pixels[1] += dab.shape[0]
                else:
                    buffer.append(filtered)
                    pixels += filtered.shape[0]

            if not stain_balancing and (pixels > threshold or (i + 1 == iter_len and len(quantity_buffer) == 0)):
                    
                vals = np.concatenate(buffer, axis=0)
                buffer = []
                pixels = 0
                if quantity == "matrix":
                    matrix = MacenkoStyleEstimation.svd(vals, self.alpha)
                    quantity_buffer.append(matrix)
                else:
                    vals = vals.transpose()
                    if self.use_nnls:
                        C = nnls(self.matrix_avg, vals) # (2,N)
                    else:
                        C = np.linalg.lstsq(self.matrix_avg, vals, rcond=None)[0]
                    maxC = np.array([np.percentile(C[0, :], 100 - self.alpha), np.percentile(C[1, :], 100 - self.alpha)])
                    quantity_buffer.append(maxC)
                tiles += 1

                if tiles % 100 == 0:
                    print(f"Processed {tiles} tiles so far...")

            elif stain_balancing and ((pixels[0] > threshold[0] and pixels[1] > threshold[1]) or (i + 1 == iter_len and len(quantity_buffer) == 0)):

                htx_vals = np.concatenate(buffer[0], axis=0) # (N_htx,3)
                dab_vals = np.concatenate(buffer[1], axis=0) # (N_dab,3)
                vals = np.concatenate([htx_vals, dab_vals], axis=0) # (N,3)
                pixels = [0, 0]
                buffer = [[], []]
                if quantity == "matrix":
                    matrix = MacenkoStyleEstimation.svd(vals, self.alpha)
                    quantity_buffer.append(matrix)
                else:
                    vals = vals.transpose()
                    if self.use_nnls:
                        C = nnls(self.matrix_avg, vals) # (2,N)
                    else:
                        C = np.linalg.lstsq(self.matrix_avg, vals, rcond=None)[0]
                    maxC = np.array([np.percentile(C[0, :], 100 - self.alpha), np.percentile(C[1, :], 100 - self.alpha)])
                    quantity_buffer.append(maxC)
                tiles += 1

                if tiles % 100 == 0:
                    print(f"Processed {tiles} tiles so far...")

            if tiles > num_tiles:
                break

        if quantity == "matrix":
            self.matrix_buffer = quantity_buffer
            self.matrix_avg = np.mean(quantity_buffer, axis=0)
        else:
            self.maxC_buffer = quantity_buffer
            if maxC_mode == "mean":
                self.maxC_avg = np.mean(quantity_buffer, axis=0)
            elif _mode == "percentile":
                self.maxC_avg = np.percentile(quantity_buffer, 100 - self.alpha, axis=0)
