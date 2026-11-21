from stainnormprogram.norm.utils import nnls
from stainnormprogram.norm.adjustments import Adjustment

import numpy as np
import math
import tifffile

class MacenkoTranslation:

    def __init__(
                self,
                slide: SlideImage,
                targetSM: np.ndarray,
                targetMaxC: np.ndarray,
                targetI0: np.ndarray,
                sourceSM: np.ndarray,
                sourceMaxC: np.ndarray,
                sourceI0: np.ndarray,
                pre_adjustment: Adjustment | None = None,
                post_adjustment: Adjustment | None = None,
                use_nnls: bool = True,
                ):
        """Perform a color transformation on an image using Macenko's method.

        Args:
            slide: SlideImage to perform the translation on.
            targetSM: Target stain matrix.
            targetMaxC: Target maxC (robust maximum concentrations).
            targetI0: Target initial intensity (background color).
            sourceSM: Source stain matrix.
            sourceMaxC: Source maxC (robust maximum concentrations).
            sourceI0: Source initial intensity (background color).
            pre_adjustment: Adjustment to apply before color transformation.
            post_adjustment: Adjustment to apply after color transformation.
            use_nnls: Use non-negative least squares for concentration estimates.
        """

        self.slide = slide
        self.targetSM = targetSM
        self.targetMaxC = targetMaxC
        self.targetI0 = targetI0
        self.sourceSM = sourceSM
        self.sourceMaxC = sourceMaxC
        self.sourceI0 = sourceI0
        self.pre_adjustment = pre_adjustment
        self.post_adjustment = post_adjustment
        self.use_nnls = use_nnls

        self.scaling_factor = np.divide(targetMaxC, sourceMaxC)[:, np.newaxis]
        
        self.slide.set_mode("sequential")
        self.slide.set_adjustment(self.pre_adjustment)
        

    def translate(self, return_metadata: bool = False):
        """Generator to perform translation.

        Some parts have been taken from https://github.com/slideflow/slideflow/blob/master/slideflow/norm/macenko.py.

        Args:
            return_metadata: Return metadata about the tile.

        Yields:
            A np.ndarray of the transformed image tile, along with a metadata dictionary if return_metadata is True.
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
