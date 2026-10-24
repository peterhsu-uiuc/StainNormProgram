from stainnormprogram.norm.pipeline.pipeline import Pipeline
from stainnormprogram.norm.pipeline.macenko import MacenkoPipeline
from stainnormprogram.norm.utils import add_images_to_dict, validate_json
from stainnormprogram.norm.subset import Subset

import os

class Normalization:

    def __init__(self):

        self.target = {}
        self.source = {}
        self.target_json = None
        self.pipeline = None

    def __repr__(self):
        return f"Normalization=(\n \
        target={self.target}\n \
        target_json={self.target_json}\n \
        source={self.source}\n \
        pipeline={self.pipeline}\n)"

    def update_dict(self, items, mydict, allow_subsets=True):
        
        if isinstance(items, (str, Subset)):
            items = [items]

        for item in items:
            if isinstance(item, Subset):
                if not allow_subsets:
                    raise Exception("Subsets are not accepted.")
                subset_id = item.identifier
                if subset_id not in mydict:
                    mydict[subset_id] = item
    
            elif isinstance(item, str):
                add_images_to_dict(item, mydict)

    def set_target(self, target):
        """Add target images or a target style.

        If list, the target style is averaged across all target images.

        Args:
            target (str, list): a path to an image or directory of images, or a list of paths/directories, or a string to a target style JSON
        """
        if isinstance(target, str) and target.endswith(".json"):
            validate_json(target)
            self.target_json = target
        else:
            self.update_dict(target, self.target, allow_subsets=False)

    def set_source(self, source):
        """Add source images to normalize.

        If str or list, the source style is calculated from each image individually. If you want the source style to be computed across several images or from a separate reference, create a Subset. If you want to add pre- or post-adjustments, create a Subset. If you want to use a style JSON as the source style (e.g., you performed correction), create a Subset.
        
        Args:
            source (str, list, Subset): a path to an image or directory of images, or a Subset, or a list of paths/directories/Subsets
        """
        self.update_dict(source, self.source)

    def set_pipeline(self, pipeline):
        """Set a normalization pipeline.

        Args:
            pipeline (str, Pipeline): the name of a pipeline or a Pipeline object.
        """
        if isinstance(pipeline, str):
            if pipeline == "MacenkoPipeline":
                self.pipeline = MacenkoPipeline()
            else:
                raise Exception(f"{pipeline} is not a valid Pipeline.")
        elif isinstance(pipeline, Pipeline):
            self.pipeline = pipeline
        else:
            raise Exception("pipeline must be a str or Pipeline.")
        
    def run_norm(self, output_dir):
        """Run normalization.

        The target style will output to output_dir as target.json. For each source image that a style is calculated on, a source style will output to [ID].json. For Subset references, the reference style will output to [Subset ID]_reference.json.

        Args:
            output_dir (str): directory to output files to. Must not exist; will be created.
        """

        # Checks
        if os.path.exists(output_dir):
            raise Exception("output_dir should not exist. output_dir will be created by this function.")
        if self.pipeline is None:
            raise Exception("No pipeline was set. Use set_pipeline.")
        if (len(self.target) == 0 and self.target_json is None):
            raise Exception("No target was set. Use set_target.")
        if (len(self.target) > 0 and self.target_json is not None):
            raise Exception("You cannot have both target images and a target JSON.")
        if len(self.source) == 0:
            raise Exception("No source images were set. Use set_source.")

        # Set output_dir
        os.makedirs(output_dir, exist_ok=False)
        self.pipeline.set_output_dir(output_dir)

        # Set target
        if self.target_json is None:
            self.pipeline.set_target(self.target)
        else:
            self.pipeline.set_target_from_json(self.target_json)

        # Set source
        self.pipeline.set_source(self.source)

        # Run pipeline
        self.pipeline.run()