from stainnormprogram.slide import SlideImage, SlideImageCollection
from stainnormprogram.norm.background import HSVBackgroundEstimation

import os
import json

import numpy as np

class BackgroundEstimation:

    def __init__(self, slide: SlideImage | SlideImageCollection, method: str = "hsv_threshold", identifier: str | None = None):
        """Estimate the background color for the assessment.

        Options for ``method``:
            * "hsv_threshold": Threshold using saturation and value channels in HSV color space.

        Args:
            slide: Image to estimate the background color of.
            method: Background estimation method.
            identifier: Image ID. If None, taken from slide.
        """
        self.slide = slide
        self.method = method
        self.identifier = identifier if identifier is not None else self.slide.identifier

        self.I0 = None # np.ndarray uint8

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        json_output_path = os.path.join(output_dir, "background_estimation.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "identifier": self.identifier,
                "method": self.method,
                "I0": self.I0.tolist(),
            }, file)

    @classmethod
    def load(cls, input_dir: str) -> BackgroundEstimation:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A BackgroundEstimation.
        """

        json_input_path = os.path.join(input_dir, "background_estimation.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        be = cls(None, method=json_data["method"], identifier=json_data["identifier"])
        be.I0 = np.array(json_data["I0"])

        return be

    def compute(self):
        """Compute the background color estimation.
        """

        if self.method == "hsv_threshold":
            slide_be = HSVBackgroundEstimation(self.slide)
            self.I0 = slide_be.compute("tiled")["median"].astype(np.uint8)
        else:
            raise Exception(f"{self.method} is not a valid method.")

class BackgroundComparison:

    def __init__(self, reference: ReferenceBackground, query: BackgroundEstimation):
        """Compare a query BackgroundEstimation to a ReferenceBackground.

        Args:
            reference: The reference background.
            query: The query background.
        """
        self.reference = reference
        self.query = query

        self.dist = None

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        json_output_path = os.path.join(output_dir, "background_comparison.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "reference": self.reference.identifier,
                "query": self.query.identifier,
                "dist": self.dist,
            }, file)

    @classmethod
    def load(cls, input_dir: str) -> BackgroundComparison:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A BackgroundComparison.
        """

        json_input_path = os.path.join(input_dir, "background_comparison.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        bc = cls(json_data["reference"], json_data["query"])
        bc.dist = json_data["dist"]

        return bc

    def get_val(self) -> float:
        """Get the background comparison distance.

        Returns:
            The background comparison distance.
        """
        return self.dist

    def compute(self):
        """Compute the background comparison.
        """
        ref_I0 = self.reference.I0.astype(np.int32)
        q_I0 = self.query.I0.astype(np.int32)

        self.dist = np.linalg.norm(ref_I0 - q_I0)
    
class ReferenceBackground:

    def __init__(self, ref_backgrounds: list[BackgroundEstimation] | BackgroundEstimation, identifier: str | None = None):
        """ReferenceBackground is the representative background of one or more reference images.

        Args:
            ref_backgrounds: One or more BackgroundEstimation objects representing one or more reference images.
            identifier: An ID for this ReferenceBackground. If None, there should be only one BackgroundEstimation in ref_backgrounds, as the ID is taken from that.
        """
        self.ref_backgrounds = ref_backgrounds
        if identifier is not None:
            self.identifier = identifier
        elif isinstance(ref_backgrounds, BackgroundEstimation):
            self.identifier = ref_backgrounds.identifier
        elif isinstance(ref_backgrounds, list) and len(ref_backgrounds) == 1:
            self.identifier = ref_backgrounds[0].identifier
        else:
            raise Exception("Set an identifier.")

        if isinstance(self.ref_backgrounds, BackgroundEstimation):
            self.ref_backgrounds = [self.ref_backgrounds]

        self.I0 = None

    def save(self, output_dir: str):
        """Save metric to disk.

        Args:
            output_dir: Folder path to output the metric to.
        """

        ref_background_identifiers = []
        for b in self.ref_backgrounds:
            ref_background_identifiers.append(b.identifier)

        json_output_path = os.path.join(output_dir, "reference_background.json")
        with open(json_output_path, "w") as file:
            json.dump({
                "identifier": self.identifier,
                "ref_background_identifiers": ref_background_identifiers,
                "I0": self.I0.tolist(),
            }, file)

    @classmethod
    def load(cls, input_dir: str) -> ReferenceBackground:
        """Load metric from disk.

        Args:
            input_dir: Folder path to load the metric from.

        Returns:
            A ReferenceBackground.
        """

        json_input_path = os.path.join(input_dir, "reference_background.json")
        with open(json_input_path, "r") as file:
            json_data = json.load(file)

        rb = cls(None, identifier=json_data["identifier"])
        rb.I0 = np.array(json_data["I0"])

        return rb

    def compute(self):
        """Compute the reference background.
        """

        I0s = []

        for b in self.ref_backgrounds:
            I0s.append(b.I0)

        self.I0 = np.median(I0s, axis=0)
        self.I0 = self.I0.astype(np.uint8)
