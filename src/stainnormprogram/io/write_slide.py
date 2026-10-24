"""
Image writers

Note about viewers: Several WSI viewers were considered when designing the file writing system.
QuPath is open-source, but I have noticed that it can render colors incorrectly.
ASAP is open-source, but it doesn't seem to be actively maintained.
Aperio ImageScope seems to be best and is free, but it is owned by Leica.
"""

import os

import numpy as np
import tifffile
import zarr
import pyvips

SNP_DESCRIPTION = "Stain normalized using StainNormProgram."
SNP_MPP_UNKNOWN = "Stain normalized using StainNormProgram. MPP resolution is not known. 1 mpp used as a placeholder."

class TiledWriter:

    def __init__(self, output_filepath, shape, tile_size, mpp_x, mpp_y, maxworkers=None, compression="zlib"):
        """
        Args:
            compression (str, NoneType): compression to apply (None, "zlib", "jpeg")
        """

        if os.path.exists(output_filepath):
            raise Exception(f"Output filepath {output_filepath} already exists.")
            
        self.output_filepath = output_filepath
        self.shape = shape
        self.tile_size = tile_size
        self.compression = compression
        if isinstance(mpp_x, float) and isinstance(mpp_y, float):
            self.mpp_x = mpp_x
            self.mpp_y = mpp_y
            self.mpp_is_known = True
        else:
            self.mpp_x = 1
            self.mpp_y = 1
            self.mpp_is_known = False
        self.maxworkers = maxworkers

        if self.maxworkers is None:
            self.maxworkers = max(1, (os.cpu_count() or 1) - 1)
        print(f"Max workers for writing = {self.maxworkers}")

        if len(self.shape) == 2:
            self.shape = (self.shape[0], self.shape[1], 3)

    def write(self, generator):

        print("Writing file...") 
        
        if not self.mpp_is_known:
            print("WARNING: MPP resolution is not known. 1 mpp used as a placeholder.")

        metadata = {
            "PhysicalSizeX": self.mpp_x,
            "PhysicalSizeXUnit": "um",
            "PhysicalSizeY": self.mpp_y,
            "PhysicalSizeYUnit": "um",
            "Description": SNP_DESCRIPTION if self.mpp_is_known else SNP_MPP_UNKNOWN,
        }

        options = dict(
            photometric="rgb",
            resolutionunit="CENTIMETER",
            tile=(self.tile_size, self.tile_size),
            compression=self.compression,
            maxworkers=self.maxworkers,
            dtype=np.uint8,
        )

        # Highest resolution (base)
        with tifffile.TiffWriter(self.output_filepath, bigtiff=True) as tif:
            tif.write(
                generator(),
                shape=self.shape,
                resolution=(1e4 / self.mpp_x, 1e4 / self.mpp_y),
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
            for i in range(4):
                ds_factor = 2 ** (i + 1)
                tif.write(
                    pyramid_generator(base_zarr, ds_factor, self.tile_size * ds_factor),
                    shape=(self.shape[0] // ds_factor, self.shape[1] // ds_factor, 3),
                    resolution=(1e4 / self.mpp_x / ds_factor, 1e4 / self.mpp_y / ds_factor),
                    **options,
                )

        base.close()

        print(f"Finished writing file to {self.output_filepath}. Can be opened with Aperio ImageScope.")