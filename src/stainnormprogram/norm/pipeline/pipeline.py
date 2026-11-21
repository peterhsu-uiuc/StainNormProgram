"""
Abstract class for stain normalization pipelines.

General pipeline steps for normalization:
    1. Calculate a target style
    2. Calculate a source style
    3. Transform the source
"""
from stainnormprogram.slide import SlideImage
from stainnormprogram.norm.subset import Subset

from abc import ABC, abstractmethod

class Pipeline(ABC):

    def __init__(self):

        self.target = None
        self.source = None

        self.output_dir = None
        self.save_by_subset = True

    def set_target(self, target: dict[str, SlideImage]):
        """
        Args:
            target: Key is the image ID (filename without extension) and value is the SlideImage.
        """
        self.target = target

    @abstractmethod
    def set_target_from_json(self, target: str):
        """Set the target style from a JSON.

        Args:
            target: Path to a JSON file containing the target style.
        """
        pass

    def set_source(self, source: dict[str, SlideImage | Subset]):
        """
        Args:
            source: Key is the image ID (filename without extension) or subset ID. Value is the SlideImage or Subset.
        """
        self.source = source

    @abstractmethod
    def run(self):
        """Run the pipeline.
        """
        pass

    def set_output_dir(self, output_dir: str, save_by_subset: bool = True):
        """
        Args:
            output_dir: Output directory.
            save_by_subset: Within the output_dir, save each Subset into its own subfolder.
        """
        self.output_dir = output_dir
        self.save_by_subset = save_by_subset
        