from stainnormprogram.norm.background import HSVBackgroundEstimation
from stainnormprogram.slide import SlideImage, SlideImageCollection

import numpy as np

class BackgroundEstimation:

    def __init__(self, slide, method="hsv_threshold"):
        self.slide = slide
        self.method = method
        self.identifier = self.slide.identifier

        self.I0 = None # np.ndarray uint8

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["slide"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.slide = None

    def compute(self):

        if self.method == "hsv_threshold":
            slide_be = HSVBackgroundEstimation(self.slide)
            self.I0 = slide_be.compute("tiled")["median"].astype(np.uint8)
        else:
            raise Exception

class BackgroundComparison:

    def __init__(self, reference, query):
        self.reference = reference
        self.query = query

        self.dist = None

    def get_val(self, val_type=None):
        return self.dist

    def compute(self):
        ref_I0 = self.reference.I0.astype(np.int32)
        q_I0 = self.query.I0.astype(np.int32)

        self.dist = np.linalg.norm(ref_I0 - q_I0)
    
class ReferenceBackground:

    def __init__(self, ref_backgrounds, identifier=None):
        self.ref_backgrounds = ref_backgrounds
        if identifier is not None:
            self.identifier = identifier
        elif isinstance(ref_backgrounds, BackgroundEstimation):
            self.identifier = ref_backgrounds.identifier
        elif isinstance(ref_backgrounds, list) and len(ref_backgrounds) == 1:
            self.identifier = ref_backgrounds[0].identifier
        else:
            raise Exception

        if isinstance(self.ref_backgrounds, BackgroundEstimation):
            self.ref_backgrounds = [self.ref_backgrounds]

        self.I0 = None

    def compute(self):

        I0s = []

        for b in self.ref_backgrounds:
            I0s.append(b.I0)

        self.I0 = np.median(I0s, axis=0)
        self.I0 = self.I0.astype(np.uint8)
