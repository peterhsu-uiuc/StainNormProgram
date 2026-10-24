import os
import random
import math
import time
import copy

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

    def __init__(self, rgb_array, adjustment=None):
        """Constructs a tile from an RGB array.

        Args:
            rgb_array : RGB (H,W,3) np.ndarray uint8 [0,255]
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
        self.gray_tile = None
        self.canny_tile = None
        self.lab_tile = None

    def rgb(self, as_uint8=False):
        return self.rgb_uint8 if as_uint8 else self.rgb_tile

    def hsv(self):
        if self.hsv_tile is None:
            self.hsv_tile = rgb2hsv(self.rgb_tile)
        return self.hsv_tile

    def luv(self):
        if self.luv_tile is None:
            self.luv_tile = rgb2luv(self.rgb_tile)
        return self.luv_tile

    def oklab(self):
        """
        sRGB to OKLAB conversion

        Expected output:
        * L 0.0 to 1.0
        * a -0.4 to 0.4
        * b -0.4 to 0.4
        """
        if self.oklab_tile is None:
            self.oklab_tile = colour.convert(self.rgb_tile, "sRGB", "Oklab")
        return self.oklab_tile

    def u_prime_v_prime(self):
        if self.u_prime is None or self.v_prime is None:
            xyz = rgb2xyz(self.rgb_tile)
            X = xyz[:,:,0]
            Y = xyz[:,:,1]
            Z = xyz[:,:,2]
            denominator = X + (15 * Y) + (3 * Z)
            self.u_prime = np.divide(4 * X, denominator, out=np.full_like(X, np.nan, dtype=np.float64), where=denominator!=0)
            self.v_prime = np.divide(9 * Y, denominator, out=np.full_like(Y, np.nan, dtype=np.float64), where=denominator!=0)
        return self.u_prime, self.v_prime

    def od(self, I0, clip_min=20):
        if self.od_tile is None:
            tile = np.clip(self.rgb_uint8, clip_min, 255)
            tile = np.log(tile / I0) * -1.0
            tile = np.clip(tile, 0, None)
            self.od_tile = tile
        return self.od_tile

    def gray(self):
        if self.gray_tile is None:
            self.gray_tile = rgb2gray(self.rgb_tile)
        return self.gray_tile

    def canny(self, sigma=1.0):
        if self.canny_tile is None:
            self.canny_tile = skimage.feature.canny(self.gray(), sigma=sigma)
        return self.canny_tile

    def lab(self):
        if self.lab_tile is None:
            self.lab_tile = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(self.rgb_tile))
        return self.lab_tile

        
class SequentialSlideImageIterator:

    def __init__(self, slide, tile_size=512):
        self.slide = slide
        self.tile_size = tile_size
        self.y = 0
        self.x = 0
        self.tile = None
        
    def __next__(self):

        
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

    def __init__(self, slide, tile_size=128):
        self.slide = slide
        self.tile_size = tile_size
        self.tile = None
        self.coords = self._compute_random_sequence()
        self.next_idx = 0

    def _compute_random_sequence(self):
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
        
    def __next__(self):

        if self.next_idx > len(self.coords) - 1:
            raise StopIteration

        y, x, tile_h, tile_w = self.coords[self.next_idx]
        self.tile = self.slide.get_tile(y, x, tile_h, tile_w)
        self.next_idx = self.next_idx + 1
        
        return self.tile, {"y": y, "x": x, "h": tile_h, "w": tile_w} 

class CustomSlideImageIterator:

    def __init__(self, slide, locs):
        self.slide = slide
        self.locs = locs
        self.tile = None
        self.next_idx = 0

    def __next__(self):

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

    def __init__(self, slides, locs):
        self.slides = slides
        self.locs = locs
        self.next_idx = 0
        
    def __next__(self):
        
        if self.next_idx > len(self.locs) - 1:
            raise StopIteration

        loc = self.locs[self.next_idx]
        identifier = loc["identifier"]
        slide = self.slides[identifier]

        if isinstance(loc, dict):
            y, x, tile_h, tile_w = loc["y"], loc["x"], loc["h"], loc["w"]
        else:
            y, x, tile_h, tile_w = loc
        tile = slide.get_tile(y, x, tile_h, tile_w)
        self.next_idx = self.next_idx + 1

        return tile, loc

class RandomSlideImageCollectionIterator:

    def __init__(self, slides, tile_size=512):
        self.slides = slides
        self.tile_size = tile_size

        for s in self.slides:
            s.set_mode("random")
            s.set_tile_size(tile_size)

        self.iterators = {}
        self.slide_keys = self.slides.keys()
        self.slide_keys_copy = []

    def __next__(self):
        
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

    def __init__(self, identifier, tile_size=512, mode="random", custom_iter_locs=None):
        self.identifier = identifier
        self.tile_size = tile_size

        self.slides = {}
        self.set_mode(mode)
        self.set_custom_iter_locs(custom_iter_locs)

    def append(self, path, identifier=None):
        slide = SlideImage(path, identifier=identifier, tile_size=self.tile_size)
        self.slides[identifier] = slide
    
    def __repr__(self):
        return f"SlideImageCollection(\n \
        identifier={self.identifier}\n \
        tile_size={self.tile_size}\n \
        slides={self.slides}\n)"

    def set_mode(self, mode):
        if mode not in ["random", "custom"]:
            raise Exception(f"{mode} is not a valid mode.")
        self.mode = mode

    def get_tile(self, y, x, h, w, identifier):
        return self.slides[identifier].get_tile(y, x, h, w)

    def get_tile_using_dict(self, metadata):
        y, x, h, w, identifier = metadata["y"], metadata["x"], metadata["h"], metadata["w"], metadata["identifier"]
        return self.get_tile(y, x, h, w, identifier)

    def set_custom_iter_locs(self, custom_iter_locs):
        self.custom_iter_locs = custom_iter_locs

    def set_tile_size(self, tile_size):
        self.tile_size = tile_size

    def __len__(self):
        return len(self.slides)

    def __iter__(self):
        if self.mode == "random":
            return RandomSlideImageCollectionIterator(self.slides, self.tile_size)
        elif self.mode == "custom":
            return CustomSlideImageCollectionIterator(self.slides, self.custom_iter_locs)

class SlideImage:

    def __init__(self, path, identifier=None, tile_size=512, mode="sequential", custom_iter_locs=None, adjustment=None):
        self.path = path
        self.slide, self.reader = self._guess_reader(path)
        self.tile_size = tile_size
        self.set_mode(mode)

        if identifier is None:
            self.identifier, _ = is_valid_extension(path, split=True, reader=self.reader)
        else:
            self.identifier = identifier

        self.set_adjustment(adjustment)
        self.set_custom_iter_locs(custom_iter_locs)

        self.thumbnail = None
        self.thumbnail_mask = None
        self.thumbnail_params = {}

    def pickleable(self):
        self.slide = None
    
    def load_slide(self):
        self.slide, self.reader = self._guess_reader(self.path)

    def compute_random_sequence(self):
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
        self.thumbnail = None
        self.thumbnail_params = {}
        self.thumbnail_mask = None

    @property
    def shape(self):
        return self.slide.shape

    @property
    def resolution(self):
        return self.slide.resolution

    def set_mode(self, mode):
        if mode not in ["sequential", "random", "custom"]:
            raise Exception(f"{mode} is not a valid mode.")
        self.mode = mode

    def set_tile_size(self, tile_size):
        self.tile_size = tile_size

    def get_pyvips(self):
        image = pyvips.Image.new_from_file(self.path, access="sequential")
        if self.adjustment is not None:
            image = self.adjustment.apply(image)
            return image
        else:
            return image

    def get_thumbnail(self, height=4096, width_max=10_000, smooth=True):
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

    def get_thumbnail_mask(self, filter_fn): # True = foreground
        if self.thumbnail_mask is not None and self.thumbnail_params["filter_fn"] == filter_fn:
            return self.thumbnail_mask

        thumbnail = self.get_thumbnail()
        thumbnail = Tile(thumbnail)
        self.thumbnail_mask = filter_fn(thumbnail)

        self.thumbnail_params["filter_fn"] = filter_fn
        return self.thumbnail_mask

    def set_adjustment(self, adjustment):
        self.adjustment = adjustment
        self.slide.set_adjustment(adjustment)

    def set_custom_iter_locs(self, custom_iter_locs):
        self.custom_iter_locs = custom_iter_locs

    def __len__(self):
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

    def _guess_reader(self, path):
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

    def get_tile(self, y, x, h, w):
        tile = self.slide.get_tile(y, x, h, w)
        if self.adjustment is not None:
            tile = self.adjustment.apply(tile)
        return tile

    def get_tile_using_dict(self, metadata):
        y, x, h, w = metadata["y"], metadata["x"], metadata["h"], metadata["w"]
        return self.get_tile(y, x, h, w)

    def write_thumbnail(self, output_path):
        if os.path.exists(output_path):
            raise Exception(f"{output_path} already exists.")
        thumbnail = self.get_thumbnail()
        thumbnail_pil = Image.fromarray(thumbnail)
        thumbnail_pil.save(output_path)

    def filter_coords_using_thumbnail(self, filter_fn, coords, tolerance=0.05, purge_thumbnails_after=True):

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

    def __init__(self, path):
        self.data = openslide.OpenSlide(path)
        self.shape = self.data.dimensions[::-1] # (width, height) -> (height, width)
        self.level = 0
        self.adjustment = None
        self.resolution = self._guess_resolution()

    def _guess_resolution(self):
        mpp_x = None
        mpp_y = None
        if "openslide.mpp-x" in self.data.properties and "openslide.mpp-y" in self.data.properties:
            mpp_x = self.data.properties["openslide.mpp-x"]
            mpp_y = self.data.properties["openslide.mpp-y"]
        return [mpp_x, mpp_y]

    def get_tile(self, y, x, height, width):
        tile = self.data.read_region((x, y), self.level, (width, height))
        tile = np.array(tile)[:,:,:3]
        tile = Tile(tile, adjustment=self.adjustment)
        return tile

    def set_adjustment(self, adjustment):
        self.adjustment = adjustment

class TifffileZarrImage:

    def __init__(self, path):
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

    def get_tile(self, y, x, height, width):
        if self.color_dim == 0:
            tile = self.data[:, y:y+height, x:x+width]
        else:
            tile = self.data[y:y+height, x:x+width, :]
        tile = Tile(tile, adjustment=self.adjustment)
        return tile

    def set_adjustment(self, adjustment):
        self.adjustment = adjustment

class BioformatsImage:

    def __init__(self, path):
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

    def get_tile(self, y, x, height, width):
        tile = np.array(self.data[y:y+height, x:x+width, :])
        tile = Tile(tile, adjustment=self.adjustment)
        return tile

    def set_adjustment(self, adjustment):
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

two_part_extensions = [
    "ome", "nii",
]

def is_valid_extension(path, split=False, reader=None):
    filename = os.path.basename(path)

    valid_extensions = []
    if reader == "tifffile" or reader is None:
        valid_extensions.extend(tifffile_valid_extensions)
    if reader == "openslide" or reader is None:
        valid_extensions.extend(openslide_valid_extensions)
    if reader == "bioformats" or reader is None:
        valid_extensions.extend(bioio_bioformats.reader_metadata.ReaderMetadata.get_supported_extensions())

    filename_wo_ext, filename_ext = split_filename(filename)

    for ext in valid_extensions:
        if filename_ext.lower() == ext:
            if split:
                return filename_wo_ext, filename_ext
            return True
    return False

def split_filename(filename):

    parts = filename.split(".")

    if len(parts) >= 3:
        for ext in two_part_extensions:
            if parts[-2].lower() == ext:
                return ".".join(parts[:-2]), "." + ".".join(parts[-2:])
    
    return ".".join(parts[:-1]), "." + parts[-1]


