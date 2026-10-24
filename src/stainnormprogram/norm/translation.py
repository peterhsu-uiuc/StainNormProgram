from stainnormprogram.norm.utils import nnls

import numpy as np
import math
import tifffile

class MacenkoTranslation:

    def __init__(self, slide, targetSM, targetMaxC, targetI0, sourceSM, sourceMaxC, sourceI0, post_adjustment=None, use_nnls=True):

        self.slide = slide
        self.targetSM = targetSM
        self.targetMaxC = targetMaxC
        self.targetI0 = targetI0
        self.sourceSM = sourceSM
        self.sourceMaxC = sourceMaxC
        self.sourceI0 = sourceI0
        self.post_adjustment = post_adjustment
        self.use_nnls = use_nnls

        self.scaling_factor = np.divide(targetMaxC, sourceMaxC)[:, np.newaxis]
        
        self.slide.set_mode("sequential")
        

    def translate(self, return_metadata=False):
        """
        Based on Slideflow
        """

        for tile, metadata in self.slide:

            tile_od = tile.od(self.sourceI0)
            h, w, _ = tile_od.shape

            tile_od = tile_od.reshape(-1,3).transpose() # (3,N)

            if self.use_nnls:
                C = nnls(self.sourceSM, tile_od)
            else:
                C = np.linalg.lstsq(self.sourceSM, tile_od, rcond=None)[0]

            C2 = C * self.scaling_factor

            Inorm = np.exp(-self.targetSM.dot(C2))
            if self.post_adjustment is not None:
                Inorm = self.post_adjustment.apply(Inorm)
            Inorm = Inorm * self.targetI0[:, np.newaxis]
            Inorm = np.clip(Inorm, 0, 255).astype(np.uint8)
            Inorm = np.reshape(Inorm.transpose(), (h, w, 3))

            if return_metadata:
                yield Inorm, metadata
            yield Inorm

            

            