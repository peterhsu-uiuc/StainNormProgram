import os
from collections.abc import Callable
from typing import Any

import numpy as np
import tifffile
import zarr
import pyvips
import imagecodecs

SNP_DESCRIPTION = "Stain normalized using StainNormProgram."

class TiledWriter:

    def __init__(self, output_filepath: str, shape: tuple[int], tile_size: int, mpp_x: float | None = None, mpp_y: float | None = None, maxworkers: int | None = None, compression: str | None = "zlib"):
        """Write a pyramidal whole slide image using a tile generator.

        Images outputted by this class can be opened using Aperio ImageScope or tifffile.
            * Aperio ImageScope: https://www.leicabiosystems.com/us/digital-pathology/manage/aperio-imagescope/
            * tifffile: https://github.com/cgohlke/tifffile

        Args:
            output_filepath: File path to save the output to.
            shape: (H,W,3) or (H,W) pixel dimensions of the image.
            tile_size: Pixel size of the tiles yielded by the generator.
            mpp_x: Microns per pixel of the x-dimension.
            mpp_y: Microns per pixel of the y-dimension.
            maxworkers: Max number of workers to use. Automatically calculated if None.
            compression: Compression to apply (None, "zlib", "jpeg").
        """

        if os.path.exists(output_filepath):
            raise Exception(f"Output filepath {output_filepath} already exists.")
            
        self.output_filepath = output_filepath
        self.shape = shape
        self.tile_size = tile_size
        self.compression = compression
        if mpp_x is not None and mpp_y is not None:
            self.mpp_x = mpp_x
            self.mpp_y = mpp_y
            self.mpp_is_known = True
        else:
            self.mpp_x = None
            self.mpp_y = None
            self.mpp_is_known = False
        self.maxworkers = maxworkers

        if self.maxworkers is None:
            self.maxworkers = max(1, (os.process_cpu_count() or 1) - 1)
        print(f"Max workers for writing = {self.maxworkers}")

        if len(self.shape) == 2:
            self.shape = (self.shape[0], self.shape[1], 3)

    def write(self, generator: Callable[[Any], np.ndarray]):
        """Write the output image slide.

        Args:
            generator: A generator that yields tiles.
        """

        print("Writing file...") 

        if not self.mpp_is_known:
            print("WARNING: Microns per pixel resolution is not known for this image.")
        
        metadata = {
            "PhysicalSizeX": self.mpp_x,
            "PhysicalSizeXUnit": "um",
            "PhysicalSizeY": self.mpp_y,
            "PhysicalSizeYUnit": "um",
            "Description": SNP_DESCRIPTION,
        }

        options = dict(
            photometric="rgb",
            resolutionunit="CENTIMETER" if self.mpp_is_known else None,
            tile=(self.tile_size, self.tile_size),
            compression=self.compression,
            maxworkers=self.maxworkers,
            dtype=np.uint8,
            extratags=[
                (34675, 7, None, imagecodecs.cms_profile('srgb'), True)
            ]
        )

        # Highest resolution (base)
        with tifffile.TiffWriter(self.output_filepath, bigtiff=True) as tif:
            tif.write(
                generator(),
                shape=self.shape,
                resolution=(1e4 / self.mpp_x, 1e4 / self.mpp_y) if self.mpp_is_known else None,
                metadata=metadata,
                **options,
            )
            
        # Pyramid resolutions
        base = tifffile.imread(self.output_filepath, aszarr=True)
        base_zarr = zarr.open(base, mode="r")

        def pyramid_generator(zr, ds, ts):
            for y in range(0, zr.shape[0], ts):
                for x in range(0, zr.shape[1], ts):
                    yield zr[y:y+ts:ds, x:x+ts:ds, :]

        with tifffile.TiffWriter(self.output_filepath, mode="r+") as tif:
            for i in range(6):
                ds_factor = 2 ** (i + 1)
                tif.write(
                    pyramid_generator(base_zarr, ds_factor, self.tile_size * ds_factor),
                    shape=(self.shape[0] // ds_factor, self.shape[1] // ds_factor, 3),
                    resolution=(1e4 / self.mpp_x / ds_factor, 1e4 / self.mpp_y / ds_factor) if self.mpp_is_known else None,
                    **options,
                )

        base.close()

        print(f"Finished writing file to {self.output_filepath}. Can be opened with Aperio ImageScope.")