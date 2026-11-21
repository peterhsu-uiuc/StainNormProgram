from stainnormprogram.norm.adjustments import Adjustment

import os
import random
import math
import time
import copy
from collections.abc import Callable

import openslide
import numpy as np
from skimage.color import rgb2hsv, rgb2luv, rgb2xyz, rgb2gray
from skimage.filters import threshold_otsu
import skimage.feature
import tifffile
import zarr
import colour
import pyvips
from PIL import Image
from bioio import BioImage
import bioio_bioformats

class Tile:

    def __init__(self, rgb_array: np.ndarray, adjustment: Adjustment | None = None):
        """Wrapper around an image tile. Caches calculated color space conversions.

        Args:
            rgb_array: RGB (H,W,3) np.ndarray uint8 [0,255]
            adjustment: Adjustment to apply.
        """
        if adjustment is not None:
            rgb_array = adjustment.apply(rgb_array)
        self.rgb_uint8 = rgb_array
        self.rgb_tile = rgb_array / 255.0 
        self.hsv_tile = None
        self.luv_tile = None
        self.oklab_tile = None
        self.u_prime = None
        self.v_prime = None
        self.od_tile = None
        self.I0 = None
        self.clip_min = None
        self.gray_tile = None
        self.canny_tile = None
        self.sigma = None
        self.lab_tile = None

    def __repr__(self):
        return f"Tile(H={self.rgb_tile.shape[0]}, W={self.rgb_tile.shape[1]}, C={self.rgb_tile.shape[2]})"

    def rgb(self, as_uint8=False) -> np.ndarray:
        """Get pixels in RGB (red, green, blue) color space.

        Args:
            as_uint8: When True, returns as RGB uint8 [0,255]. When False, values are scaled within the interval [0.,1.].

        Returns:
            A np.ndarray of RGB pixels.
        """
        return self.rgb_uint8 if as_uint8 else self.rgb_tile

    def hsv(self) -> np.ndarray:
        """Get pixels in HSV (hue, saturation, value) color space.

        Expected output ranges:
        * H 0.0 to 1.0
        * S 0.0 to 1.0
        * V 0.0 to 1.0

        Returns:
            A np.ndarray of HSV pixels.
        """
        if self.hsv_tile is None:
            self.hsv_tile = rgb2hsv(self.rgb_tile)
        return self.hsv_tile

    def luv(self) -> np.ndarray:
        """Get pixels in CIELUV color space.

        Expected output ranges:
        * L 0.0 to 100.0
        * U -134.0 to 220.0 (Theoretically unbounded)
        * V -140.0 to 122.0 (Theoretically unbounded)

        Returns:
            A np.ndarray of CIELUV pixels.
        """
        if self.luv_tile is None:
            self.luv_tile = rgb2luv(self.rgb_tile)
        return self.luv_tile

    def oklab(self) -> np.ndarray:
        """Get pixels in OKLAB color space.

        Expected output ranges:
        * L 0.0 to 1.0
        * a -0.5 to 0.5 (Theoretically unbounded)
        * b -0.5 to 0.5 (Theoretically unbounded)

        Returns:
            A np.ndarray of OKLAB pixels.
        """
        if self.oklab_tile is None:
            self.oklab_tile = colour.convert(self.rgb_tile, "sRGB", "Oklab")
        return self.oklab_tile

    def u_prime_v_prime(self) -> tuple[np.ndarray, np.ndarray]:
        """Get pixels in CIE 1976 u'v' chromaticity coordinates.

        This function can return np.ndarrays containing np.nan when the denominator is 0.

        Expected output ranges:
        * u' 0.0 to 0.7 (Theoretically unbounded)
        * v' 0.0 to 0.7 (Theoretically unbounded)

        Returns:
            A tuple of np.ndarrays, each (H,W). The first is u' and the second is v'.
        """
        if self.u_prime is None or self.v_prime is None:
            xyz = rgb2xyz(self.rgb_tile)
            X = xyz[:,:,0]
            Y = xyz[:,:,1]
            Z = xyz[:,:,2]
            denominator = X + (15 * Y) + (3 * Z)
            self.u_prime = np.divide(4 * X, denominator, out=np.full_like(X, np.nan, dtype=np.float64), where=denominator!=0)
            self.v_prime = np.divide(9 * Y, denominator, out=np.full_like(Y, np.nan, dtype=np.float64), where=denominator!=0)
        return self.u_prime, self.v_prime

    def od(self, I0: np.ndarray, clip_min: int = 20) -> np.ndarray:
        """Get pixels in Beer-Lambert's optical density (OD) space.

        Args:
            I0: Initial intensity.
            clip_min: Min clamp [0, 255].

        Returns:
            A np.ndarray of OD pixels.
        """
        if self.od_tile is None or (not np.array_equal(self.I0, I0) or self.clip_min != clip_min):
            tile = np.clip(self.rgb_uint8, clip_min, 255)
            tile = np.log(tile / I0) * -1.0
            tile = np.clip(tile, 0, None)
            self.od_tile = tile
            self.I0 = I0
            self.clip_min = clip_min
        return self.od_tile

    def gray(self) -> np.ndarray:
        """Get pixels in grayscale (uses scikit-image's rgb2gray, which is a perceptual grayscale instead of simple average).

        Returns:
            A np.ndarray of grayscale pixels.
        """
        if self.gray_tile is None:
            self.gray_tile = rgb2gray(self.rgb_tile)
        return self.gray_tile

    def canny(self, sigma: float = 1.0) -> np.ndarray:
        """Get Canny edge detection.

        Args:
            sigma: Gaussian filter sigma.

        Returns:
            A np.ndarray of a Canny edge detection bool mask. The output shape is (H,W).
        """
        if self.canny_tile is None or (self.sigma != sigma):
            self.canny_tile = skimage.feature.canny(self.gray(), sigma=sigma)
            self.sigma = sigma
        return self.canny_tile

    def lab(self) -> np.ndarray:
        """Get pixels in CIELAB color space.

        Expected output ranges:
        * L 0.0 to 100.0
        * A -128.0 to 127.0 (Theoretically unbounded)
        * B -128.0 to 127.0 (Theoretically unbounded)

        Returns:
            A np.ndarray of CIELAB pixels.
        """
        if self.lab_tile is None:
            self.lab_tile = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(self.rgb_tile))
        return self.lab_tile

class SequentialSlideImageIterator:

    def __init__(self, slide: SlideImage, tile_size: int = 512):
        """Iterator to go through tiles in a SlideImage sequentially (row major order).

        Args:
            slide: The SlideImage to iterate over.
            tile_size: Tile size to use.
        """
        self.slide = slide
        self.tile_size = tile_size
        self.y = 0
        self.x = 0
        self.tile = None
        
    def __next__(self) -> tuple[Tile, dict[str, int]]:

        if self.x >= self.slide.shape[1]:
            self.x = 0
            self.y += self.tile_size
        if self.y >= self.slide.shape[0]:
            raise StopIteration
        tile_w = self.tile_size
        tile_h = self.tile_size
        if self.y + tile_h > self.slide.shape[0]:
            tile_h = self.slide.shape[0] - self.y
        if self.x + tile_w > self.slide.shape[1]:
            tile_w = self.slide.shape[1] - self.x

        self.tile = self.slide.get_tile(self.y, self.x, tile_h, tile_w)

        y = self.y
        x = self.x
        self.x += self.tile_size

        return self.tile, {"y": y, "x": x, "h": tile_h, "w": tile_w} 

class RandomSlideImageIterator:

    def __init__(self, slide: SlideImage, tile_size: int = 128):
        """Iterator to go through tiles in a SlideImage randomly (chosen using a grid).

        Args:
            slide: The SlideImage to iterate over.
            tile_size: Tile size to use.
        """
        self.slide = slide
        self.tile_size = tile_size
        self.tile = None
        self.coords = self._compute_random_sequence()
        self.next_idx = 0

    def _compute_random_sequence(self) -> list[tuple[int, int, int, int]]:
        """Compute a random sequence of tile locations.

        Returns:
            A list containing tuples of y, x, tile height, tile width ints.
        """
        h, w = self.slide.shape
        coords = []
        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                tile_h = self.tile_size
                tile_w = self.tile_size
                if y + self.tile_size > h:
                    tile_h = h - y
                if x + self.tile_size > w:
                    tile_w = w - x
                coords.append((y, x, tile_h, tile_w))

        random.shuffle(coords)

        return coords
        
    def __next__(self) -> tuple[Tile, dict[str, int]]:

        if self.next_idx > len(self.coords) - 1:
            raise StopIteration

        y, x, tile_h, tile_w = self.coords[self.next_idx]
        self.tile = self.slide.get_tile(y, x, tile_h, tile_w)
        self.next_idx = self.next_idx + 1
        
        return self.tile, {"y": y, "x": x, "h": tile_h, "w": tile_w} 

class CustomSlideImageIterator:

    def __init__(self, slide: SlideImage, locs: list[dict[str, int] | tuple[int, int, int, int]]):
        """Iterator to go over tiles in a SlideImage using a custom order of locations to visit.

        Args:
            slide: The SlideImage to iterate over.
            locs: A list of tile locations to visit in their preferred order. Each item is a dictionary with keys "y", "x", "h", and "w" and values as ints specifying the tile location. Each item can also be a tuple of the ints in y, x, h, w order.
        """
        self.slide = slide
        self.locs = locs
        self.tile = None
        self.next_idx = 0

    def __next__(self) -> tuple[Tile, dict[str, int] | tuple[int, int, int, int]]:

        if self.next_idx > len(self.locs) - 1:
            raise StopIteration

        loc = self.locs[self.next_idx]
        if isinstance(loc, dict):
            y, x, tile_h, tile_w = loc["y"], loc["x"], loc["h"], loc["w"]
        else:
            y, x, tile_h, tile_w = loc
        self.tile = self.slide.get_tile(y, x, tile_h, tile_w)
        self.next_idx = self.next_idx + 1

        return self.tile, loc

class CustomSlideImageCollectionIterator:

    def __init__(self, slides: list[SlideImage], locs: list[dict[str, int | str]]):
        """Iterator to go over tiles in a list of SlideImages using a custom order of locations to visit.

        Args:
            slides: The list of SlideImages to iterate over.
            locs: A list of tile locations to visit in their preferred order. Each item is a dictionary with keys "identifier", "y", "x", "h", and "w" and values specifying the tile location.
        """
        self.slides = slides
        self.locs = locs
        self.tile = None
        self.next_idx = 0
        
    def __next__(self) -> tuple[Tile, dict[str, int | str]]:
        
        if self.next_idx > len(self.locs) - 1:
            raise StopIteration

        loc = self.locs[self.next_idx]
        identifier = loc["identifier"]
        slide = self.slides[identifier]

        y, x, tile_h, tile_w = loc["y"], loc["x"], loc["h"], loc["w"]
        
        self.tile = slide.get_tile(y, x, tile_h, tile_w)
        self.next_idx = self.next_idx + 1

        return self.tile, loc

class RandomSlideImageCollectionIterator:

    def __init__(self, slides: list[SlideImage], tile_size: int = 512):
        """Iterator to go over tiles in a list of SlideImages randomly.

        Args:
            slides: The list of SlideImages to iterate over.
            tile_size: Tile size to use.
        """
        self.slides = slides
        self.tile_size = tile_size

        for s in self.slides:
            s.set_mode("random")
            s.set_tile_size(tile_size)

        self.iterators = {}
        self.slide_keys = self.slides.keys()
        self.slide_keys_copy = []

    def __next__(self) -> tuple[Tile, dict[str, int | str]]:
        
        while True:
            
            if len(self.slide_keys_copy) == 0:
                
                if len(self.slide_keys) == 0:
                    raise StopIteration

                self.slide_keys_copy = self.slide_keys.copy()
                random.shuffle(self.slide_keys_copy)

            slide_key = self.slide_keys_copy.pop(0)
            s = self.slides[slide_key]
            if s.identifier not in self.iterators:
                self.iterators[s.identifier] = iter(s)
            
            try:
                tile, metadata = next(self.iterators[s.identifier])
                metadata["identifier"] = s.identifier
                return tile, metadata
            except StopIteration as si:
                self.slide_keys.remove(slide_key)

class SlideImageCollection:

    def __init__(self, identifier: str, tile_size: int = 512, mode: str = "random", custom_iter_locs: list[dict[str, str | int]] | None = None):
        """A wrapper to collect multiple SlideImages together.

        Args:
            identifier: A str to identify this collection.
            tile_size: The tile size to use.
            mode: Sampling mode. Either "random" or "custom".
            custom_iter_locs: If mode is "custom", then you should set custom_iter_locs. This should be a list of tile locations to visit in their preferred order. Each item is a dictionary with keys "identifier", "y", "x", "h", and "w" and values specifying the tile location.
        """
        self.identifier = identifier
        self.tile_size = tile_size

        self.slides = {}
        self.set_mode(mode)
        self.set_custom_iter_locs(custom_iter_locs)

    def append(self, path: str, identifier: str | None = None):
        """Append an image to this collection.

        Args:
            path: The file path to the image.
            identifier: An identifer for the image. If None, the identifier is taken from the filename.
        """
        slide = SlideImage(path, identifier=identifier, tile_size=self.tile_size)
        identifier = slide.identifier
        self.slides[identifier] = slide
    
    def __repr__(self):
        return f"SlideImageCollection(\n \
        identifier={self.identifier}\n \
        tile_size={self.tile_size}\n \
        slides={self.slides}\n)"

    def set_mode(self, mode: str):
        """Set the sampling mode.

        Args:
            mode: Either "random" or "custom".
        """
        if mode not in ["random", "custom"]:
            raise Exception(f"{mode} is not a valid mode.")
        self.mode = mode

    def get_tile(self, y: int, x: int, h: int, w: int, identifier: str) -> Tile:
        """Get tile.

        Args:
            identifier: The ID (filename without extension) of the image to get the tile from.
            y: int y-coordinate of the upper left corner.
            x: int x-coordinate of the upper left corner.
            h: int height of the tile.
            w: int width of the tile.

        Returns:
            The requested Tile.
        """
        return self.slides[identifier].get_tile(y, x, h, w)

    def get_tile_using_dict(self, metadata: dict[str, int | str]) -> Tile:
        """Get tile using a metadata dictionary.

        Args:
            metadata: A dictionary with "identifier", "y", "x", "h", "w" entries for fully specifying the location of the tile to retrieve.

        Returns:
            The requested Tile.
        """
        y, x, h, w, identifier = metadata["y"], metadata["x"], metadata["h"], metadata["w"], metadata["identifier"]
        return self.get_tile(y, x, h, w, identifier)

    def set_custom_iter_locs(self, custom_iter_locs: list[dict[str, str | int]] | None):
        """Set custom tile locations for iteration.

        Each tile location is a dictionary with these key/value pairs:
            * "identifier": The ID (filename without extension) of the image to get the tile from.
            * "y": int y-coordinate of the upper left corner.
            * "x": int x-coordinate of the upper left corner.
            * "h": int height of the tile.
            * "w": int width of the tile.

        Args:
            custom_iter_locs: A list of tile locations (in their desired order), with each location being a dictionary.
        """
        self.custom_iter_locs = custom_iter_locs

    def set_tile_size(self, tile_size: int):
        """Set tile size.

        Args:
            tile_size: Tile size to use.
        """
        self.tile_size = tile_size

    def __len__(self) -> int:
        """Number of slides in this collection.

        Returns:
            The number of slides in this collection.
        """
        return len(self.slides)

    def __iter__(self):
        if self.mode == "random":
            return RandomSlideImageCollectionIterator(self.slides, self.tile_size)
        elif self.mode == "custom":
            if self.custom_iter_locs is not None:
                return CustomSlideImageCollectionIterator(self.slides, self.custom_iter_locs)
            else:
                raise Exception("custom_iter_locs must be set to perform custom iteration over this SlideImageCollection.")

class SlideImage:

    def __init__(
            self,
            path: str,
            identifier: str | None = None,
            tile_size: int = 512,
            mode: str = "sequential",
            custom_iter_locs: list[dict[str, int] | tuple[int, int, int, int]] | None = None,
            adjustment: Adjustment | None = None
            ):
        """A wrapper for a whole slide image. Manages reading different file types by choosing an appropriate reader based on the file extension.

        Args:
            path: A file path to the whole slide image.
            identifier: A str identifier for the image. If None, the identifier is taken from the filename.
            tile_size: Tile size to use.
            mode: Sampling mode ("sequential", "random", or "custom").
            custom_iter_locs: If "custom" mode is used, specify a list of tile locations to visit. It should be a list of tile locations to visit in their preferred order. Each item is a dictionary with keys "y", "x", "h", and "w" and values as ints specifying the tile location. Each item can also be a tuple of the ints in y, x, h, w order.
            adjustment: An Adjustment to apply to the image.
        """
        self.path = path
        self.slide, self.reader = self._guess_reader(path)
        self.tile_size = tile_size
        self.set_mode(mode)

        if identifier is None:
            self.identifier, _ = is_valid_extension(path, split=True, reader=self.reader)
        else:
            self.identifier = identifier

        self.set_adjustment(adjustment)

        if custom_iter_locs is not None:
            if mode != "custom":
                raise Exception("If custom_iter_locs is set, the mode should be set to \"custom\"")
            self.set_custom_iter_locs(custom_iter_locs)

        # Thumbnail cache
        self.thumbnail = None
        self.thumbnail_mask = None
        self.thumbnail_params = {}

    def pickleable(self):
        """Make this object pickleable by setting the slide reference to None.

        You can undo this operation by calling load_slide.
        """
        self.slide = None
    
    def load_slide(self):
        """Load the slide back again after calling pickleable.
        """
        self.slide, self.reader = self._guess_reader(self.path)

    def compute_random_sequence(self) -> list[tuple[int, int, int, int]]:
        """Compute a random tile location sequence.

        Calling this function automatically sets the sampling mode of this slide to "custom" and its custom_iter_locs to the computed random sequence.

        Returns:
            A list of tuples, each consisting of y, x, tile_h, tile_w ints for the tile location.
        """
        h, w = self.slide.shape
        coords = []
        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                tile_h = self.tile_size
                tile_w = self.tile_size
                if y + self.tile_size > h:
                    tile_h = h - y
                if x + self.tile_size > w:
                    tile_w = w - x
                coords.append((y, x, tile_h, tile_w))

        random.shuffle(coords)
        
        self.set_mode("custom")
        self.custom_iter_locs = coords
        return coords

    def __repr__(self):
        return f"SlideImage(identifier={self.identifier}, h={self.slide.shape[0]}, w={self.slide.shape[1]})"

    def purge(self):
        """Set thumbnail cache variables to None or default values to release memory.
        """
        self.thumbnail = None
        self.thumbnail_params = {}
        self.thumbnail_mask = None

    @property
    def shape(self):
        """Get the shape of the image.

        Returns:
            Provides the shape (in pixels) of the image as (H,W).
        """
        return self.slide.shape

    @property
    def resolution(self):
        """Get the resolution of the image.

        Returns:
            Provides the resolution (in microns per pixel) of the image as [x resolution, y resolution]. May be [None, None] if the resolution is not known.
        """
        return self.slide.resolution

    def set_mode(self, mode: str):
        """Set the sampling mode.

        Args:
            mode: "sequential" for row major order, "random" for random selection from a grid, "custom" for a custom order specified using set_custom_iter_locs.
        """
        if mode not in ["sequential", "random", "custom"]:
            raise Exception(f"{mode} is not a valid mode.")
        self.mode = mode

    def set_tile_size(self, tile_size: int):
        """Set the tile size.

        Args:
            tile_size: The tile size to set.
        """
        self.tile_size = tile_size

    def get_pyvips(self) -> pyvips.Image:
        """Get the pyvips version of this image.

        If an Adjustment has been set, the pyvips image will have the adjustment in its computational path.

        Returns:
            A pyvips.Image image.
        """
        image = pyvips.Image.new_from_file(self.path, access="sequential")
        if self.adjustment is not None:
            image = self.adjustment.apply(image)
            return image
        else:
            return image

    def get_thumbnail(self, height: int = 4096, width_max: int = 10_000, smooth: bool = True) -> np.ndarray:
        """Get the thumbnail of the whole slide image using Pyvips.

        Args:
            height: The maximum height of the thumbnail.
            width_max: The maximum width of the thumbnail.
            smooth: Allow the use of interpolation that smooths pixels.

        Returns:
            A np.ndarray of a thumbnail of the slide image.
        """
        if self.thumbnail is not None and self.thumbnail_params["height"] == height and self.thumbnail_params["width_max"] == width_max and self.thumbnail_params["smooth"] == smooth:
            return self.thumbnail

        self.thumbnail_params = {"height": height, "width_max": width_max, "smooth": smooth}
        self.thumbnail_mask = None

        if height > self.slide.shape[0]:
            height = self.slide.shape[0]

        # Estimate the output width
        scale = height / self.slide.shape[0]
        width = self.slide.shape[1] * scale
        if width > width_max:
            width = width_max
            scale = width / self.slide.shape[1]
            height = round(self.slide.shape[0] * scale)
        
        if smooth:
            self.thumbnail = pyvips.Image.thumbnail(self.path, height).numpy()
        else:
            self.thumbnail = self.get_pyvips().resize(scale, kernel="nearest").numpy()

        self.thumbnail = self.thumbnail[:,:,:3]
        if self.adjustment is not None:
            self.thumbnail = self.adjustment.apply(self.thumbnail)

        return self.thumbnail

    def get_thumbnail_mask(self, filter_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Get a foreground mask of the thumbnail.

        Args:
            filter_fn: A function for thresholding the thumbnail and outputs a mask.

        Returns:
            A np.ndarray for the foreground mask (True = foreground).
        """
        if self.thumbnail_mask is not None and self.thumbnail_params["filter_fn"] == filter_fn:
            return self.thumbnail_mask

        thumbnail = self.get_thumbnail()
        thumbnail = Tile(thumbnail)
        self.thumbnail_mask = filter_fn(thumbnail)

        self.thumbnail_params["filter_fn"] = filter_fn
        return self.thumbnail_mask

    def set_adjustment(self, adjustment: Adjustment | None):
        """Set a pre-adjustment on the image (or set back to None).

        Args:
            adjustment: The Adjustment to set.
        """
        self.adjustment = adjustment
        self.slide.set_adjustment(adjustment)

    def set_custom_iter_locs(self, custom_iter_locs: list[dict[str, int] | tuple[int]] | None):
        """Set the custom tile locations to visit during iteration.

        This automatically sets the mode to "custom".

        Args:
            custom_iter_locs: List of tile locations to visit. It should be a list of tile locations to visit in their preferred order. Each item is a dictionary with keys "y", "x", "h", and "w" and values as ints specifying the tile location. Each item can also be a tuple of the ints in y, x, h, w order.
        """
        self.set_mode("custom")
        self.custom_iter_locs = custom_iter_locs

    def __len__(self) -> int:
        """Number of tiles.

        Returns:
            The number of tiles.
        """
        return int(math.ceil(self.slide.shape[0] / self.tile_size) * math.ceil(self.slide.shape[1] / self.tile_size))

    def __iter__(self):
        if self.mode == "sequential":
            return SequentialSlideImageIterator(self.slide, self.tile_size)
        elif self.mode == "random":
            return RandomSlideImageIterator(self.slide, self.tile_size)
        elif self.mode == "custom":
            if self.custom_iter_locs is None:
                raise Exception("Mode set to custom. Call set_custom_iter_locs first to specify the custom tile order.")
            return CustomSlideImageIterator(self.slide, self.custom_iter_locs)

    def _guess_reader(self, path: str) -> tuple[Any, str]:
        """Lazy load the image by guessing which package will work.

        Args:
            path: The file path to the whole slide image.

        Returns:
            The whole slide image object and str for the reader used.
        """
        if not os.path.exists(path):
            raise Exception(f"{path} does not exist")

        try:
            if not is_valid_extension(path, reader="openslide"):
                raise Exception
            slide = OpenSlideImage(path)
            return slide, "openslide"
        except Exception:
            pass

        try:
            if not is_valid_extension(path, reader="tifffile"):
                raise Exception
            slide = TifffileZarrImage(path)
            return slide, "tifffile"
        except Exception:
            pass

        try:
            if not is_valid_extension(path, reader="bioformats"):
                raise Exception
            slide = BioformatsImage(path)
            return slide, "bioformats"
        except Exception:
            pass

        raise Exception(f"Tried all available slide readers on {path} but was unsuccessful. Convert your image into a format compatible with OpenSlide, tifffile, or bioio-bioformats.")

    def get_tile(self, y: int, x: int, h: int, w: int) -> Tile:
         """Get tile.

        Args:
            y: int y-coordinate of the upper left corner.
            x: int x-coordinate of the upper left corner.
            h: int height of the tile.
            w: int width of the tile.

        Returns:
            The requested Tile.
        """
        tile = self.slide.get_tile(y, x, h, w)
        if self.adjustment is not None:
            tile = self.adjustment.apply(tile)
        return tile

    def get_tile_using_dict(self, metadata: dict[str, int]) -> Tile:
        """Get tile using a metadata dictionary.

        Args:
            metadata: A dictionary with "y", "x", "h", "w" entries for fully specifying the location of the tile to retrieve.

        Returns:
            The requested Tile.
        """
        y, x, h, w = metadata["y"], metadata["x"], metadata["h"], metadata["w"]
        return self.get_tile(y, x, h, w)

    def write_thumbnail(self, output_path: str, height: int = 4096):
        """Write a thumbnail of this image to disk.

        Args:
            output_path: The path to output the thumbnail.
            height: The maximum height of the thumbnail.
        """
        if os.path.exists(output_path):
            raise Exception(f"{output_path} already exists.")
        thumbnail = self.get_thumbnail(height=height)
        thumbnail_pil = Image.fromarray(thumbnail)
        thumbnail_pil.save(output_path)

    def filter_coords_using_thumbnail(self, filter_fn: Callable[[np.ndarray], np.ndarray], coords: list[tuple[int, int, int, int]], tolerance: float = 0.05, purge_thumbnails_after: bool = True) -> list[tuple[int, int, int, int]]:
        """Detect the tissue in the thumbnail of the slide image. From this detection, accept only tile locations that have tissue in it and remove locations that don't have tissue in it.

        When this function is called, the filtered coordinates will be set as the custom_iter_locs and the sampling mode of the slide will change to "custom".

        Args:
            filter_fn: A function to threshold the thumbnail, outputting a foreground mask.
            coords: A list of tile coordinates to filter. Each tuple should be y, x, tile height, tile width.
            tolerance: Min percent of the tile that must have tissue to be accepted [0.0,1.0].
            purge_thumbnails_after: If True, thumbnail cache will be dereferenced to avoid using too much RAM.

        Returns:
            A list of tuples (each tuple consisting of y, x, tile height, tile width) of filtered tile locations.
        """
        print(f"Filtering given coordinates using thumbnail...")

        thumbnail_mask = self.get_thumbnail_mask(filter_fn)
        thumb_h, thumb_w = thumbnail_mask.shape
        real_h, real_w = self.shape
        scale = thumb_h / real_h
        if not math.isclose(scale, thumb_w / real_w, rel_tol=10, abs_tol=0.1):
            raise Exception(f"Thumbnail expected to be scaled equally in both dimensions. Y scale = {scale}, X scale = {thumb_w / real_w}")

        filtered_coords = []

        for y, x, tile_h, tile_w in coords:
            thumb_y = round(y * scale)
            thumb_x = round(x * scale)
            thumb_tile_h = round(tile_h * scale)
            thumb_tile_w = round(tile_w * scale)
            px_tolerance = min(round(tolerance * thumb_tile_h * thumb_tile_w), 1)

            tile_mask = thumbnail_mask[thumb_y:thumb_y+thumb_tile_h, thumb_x:thumb_x+thumb_tile_w]
            if np.sum(tile_mask) >= px_tolerance:
                filtered_coords.append((y, x, tile_h, tile_w))

        print(f"Tiles before filtering = {len(coords)}, Tiles after filtering = {len(filtered_coords)}")
        
        if purge_thumbnails_after:
            self.purge()
        self.set_mode("custom")
        self.set_custom_iter_locs(filtered_coords)
        return filtered_coords

class OpenSlideImage:
    
    def __init__(self, path: str):
        """Wrapper for an OpenSlide image.

        Args:
            path: File path of the image.
        """
        self.data = openslide.OpenSlide(path)
        self.shape = self.data.dimensions[::-1] # (width, height) -> (height, width)
        self.level = 0
        self.adjustment = None
        self.resolution = self._guess_resolution()

    def _guess_resolution(self) -> list[float]:
        """Guess the microns per pixel resolution of the image based on the metadata.

        Returns:
            A list with two floats [mpp_x, mpp_y] for the microns per pixel in the x and y directions.
        """
        mpp_x = None
        mpp_y = None
        if "openslide.mpp-x" in self.data.properties and "openslide.mpp-y" in self.data.properties:
            mpp_x = float(self.data.properties["openslide.mpp-x"])
            mpp_y = float(self.data.properties["openslide.mpp-y"])
        return [mpp_x, mpp_y]

    def get_tile(self, y: int, x: int, height: int, width: int) -> Tile:
        """Get tile.

        Args:
            y: int y-coordinate of the upper left corner.
            x: int x-coordinate of the upper left corner.
            height: int height of the tile.
            width: int width of the tile.

        Returns:
            The requested Tile.
        """
        tile = self.data.read_region((x, y), self.level, (width, height))
        tile = np.array(tile)[:,:,:3]
        tile = Tile(tile, adjustment=self.adjustment)
        return tile

    def set_adjustment(self, adjustment: Adjustment | None):
        """Set a pre-adjustment.

        Args:
            adjustment: The Adjustment to set.
        """
        self.adjustment = adjustment

class TifffileZarrImage:
    
    def __init__(self, path: str):
        """Wrapper for a tifffile/zarr image.

        Args:
            path: File path of the image. 
        """
        store = tifffile.imread(path, aszarr=True)
        self.data = zarr.open(store, mode='r')
        if self.data.shape[0] == 3:
            self.color_dim = 0
        elif self.data.shape[-1] == 3:
            self.color_dim = -1
        else:
            raise Exception("Could not find the color dimension.")
        self.shape = self.data.shape[1:] if self.color_dim == 0 else self.data.shape[:-1]
        self.adjustment = None
        self.resolution = [None, None]

    def get_tile(self, y: int, x: int, height: int, width: int) -> Tile:
        """Get tile.

        Args:
            y: int y-coordinate of the upper left corner.
            x: int x-coordinate of the upper left corner.
            height: int height of the tile.
            width: int width of the tile.

        Returns:
            The requested Tile.
        """
        if self.color_dim == 0:
            tile = self.data[:, y:y+height, x:x+width]
        else:
            tile = self.data[y:y+height, x:x+width, :]
        tile = Tile(tile, adjustment=self.adjustment)
        return tile

    def set_adjustment(self, adjustment: Adjustment | None):
        """Set a pre-adjustment.

        Args:
            adjustment: The Adjustment to set.
        """
        self.adjustment = adjustment

class BioformatsImage:
    
    def __init__(self, path: str):
        """Wrapper for a Bioformats image.

        Args:
            path: File path of the image. 
        """
        store = BioImage(path, reader=bioio_bioformats.Reader)

        if len(store.scenes) > 1:
            max_y = 0
            best_scene = 0
            for i in range(len(store.scenes)):
                store.set_scene(i)
                if store.dims.Y > max_y:
                    max_y = store.dims.Y
                    best_scene = i
            store.set_scene(best_scene)

        if store.dims.T > 1:
            raise Exception("Expected the T dimension of this image to be 1. Images with a T dimension are not supported yet.")
        if store.dims.Z > 1:
            raise Exception("Expected the Z dimension of this image to be 1. Images with a Z dimension are not supported yet.")
                
        if store.dims.C == 3:
            self.data = store.get_image_dask_data("YXC")
        elif store.dims.S == 3:
            self.data = store.get_image_dask_data("YXS")
        else:
            raise Exception("Cannot find the color dimension in this image.")

        self.shape = self.data.shape[:2]
        self.adjustment = None
        self.resolution = [None, None]

    def get_tile(self, y: int, x: int, height: int, width: int) -> Tile:
        """Get tile.

        Args:
            y: int y-coordinate of the upper left corner.
            x: int x-coordinate of the upper left corner.
            height: int height of the tile.
            width: int width of the tile.

        Returns:
            The requested Tile.
        """
        tile = np.array(self.data[y:y+height, x:x+width, :])
        tile = Tile(tile, adjustment=self.adjustment)
        return tile

    def set_adjustment(self, adjustment: Adjustment | None):
        """Set a pre-adjustment.

        Args:
            adjustment: The Adjustment to set.
        """
        self.adjustment = adjustment

openslide_valid_extensions = [
    ".svs",
    ".avs",
    ".dcm",
    ".vms", ".vmu", ".ndpi",
    ".tif", ".tiff",
    ".scn",
    ".mrxs",
    ".svslide",
    ".bif",
    ".czi",
]

tifffile_valid_extensions = [
    ".ome.tiff", ".ome.tif", ".tif", ".tiff",
]

bioformats_valid_extensions = [
    ".vsi",
]

two_part_extensions = [
    "ome", "nii",
]

def is_valid_extension(path: str, split: bool = False, reader: str | None = None) -> bool | tuple[str, str]:
    """Check if the path is a valid whole slide image based on its extension.

    Args:
        path: The file path to the whole slide image.
        split: If True, the filename without extension and the extension will be returned. If False, the function just returns a bool if the path is a valid whole slide image.
        reader: If None, the path is checked against all available readers. Otherwise, the path is checked against only the reader specified ("tifffile", "openslide", "bioformats").

    Returns:
        Either a bool (if split is False) indicating if the path is a valid whole slide image that can be read or two strings corresponding to the filename without extension and the extension if it is valid.
    """
    filename = os.path.basename(path)

    valid_extensions = []
    if reader == "tifffile" or reader is None:
        valid_extensions.extend(tifffile_valid_extensions)
    if reader == "openslide" or reader is None:
        valid_extensions.extend(openslide_valid_extensions)
    if reader == "bioformats" or reader is None:
        # A full listing of what bioformats can load is at bioio_bioformats.reader_metadata.ReaderMetadata.get_supported_extensions()
        # We restrict the valid formats using bioformats_valid_extensions since there are some formats (e.g., .txt) that don't make sense for our use.
        valid_extensions.extend(bioformats_valid_extensions)

    filename_wo_ext, filename_ext = split_filename(filename)

    for ext in valid_extensions:
        if filename_ext.lower() == ext:
            if split:
                return filename_wo_ext, filename_ext
            return True
    return False

def split_filename(filename: str) -> tuple[str, str]:
    """Helper function to split a filename into its name without extension and extension.

    Args:
        filename: The filename to split.

    Returns:
        The filename without extension and the extension.
    """

    parts = filename.split(".")

    if len(parts) >= 3:
        for ext in two_part_extensions:
            if parts[-2].lower() == ext:
                return ".".join(parts[:-2]), "." + ".".join(parts[-2:])
    
    return ".".join(parts[:-1]), "." + parts[-1]
