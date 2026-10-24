from stainnormprogram.norm.style_estimation.macenko import MacenkoStyleEstimation
from stainnormprogram.norm.background import HSVBackgroundEstimation, hsv_threshold_foreground
from stainnormprogram.norm.translation import MacenkoTranslation
from stainnormprogram.io.write_slide import TiledWriter
from stainnormprogram.norm.subset import Subset
from stainnormprogram.slide import SlideImage
from stainnormprogram.norm.pipeline.pipeline import Pipeline

import os
import json

import numpy as np

class MacenkoPipeline(Pipeline):

    def __init__(self,
                 stain="HE",
                 bkgd_method="tiled",
                 bkgd_options={},
                 style_est_mode="tiled_parallel",
                 style_est_options={"detect_fg": True},
                 alpha=1.0,
                 beta=0.15,
                 use_nnls=True,
                 maxworkers=None,
                 compression="zlib",
                ):
        """Create a Macenko pipeline.

        Args:
            stain (str): stain type (either "HE" or "DAB")
            bkgd_method (str): background estimation method ("thumbnail", "tiled", or "pyvips")
            bkgd_options (dict): kwargs for background estimation
            style_est_mode (str): style estimation mode ("tiled_parallel", "thumbnail", or "tiled")
            style_est_options (dict): kwargs for style estimation
            alpha (float): Macenko's alpha
            beta (float): Macenko's beta
            use_nnls (bool): use non-negative least squares for concentration estimations
            maxworkers (int): max workers for file saving
            compression (str): compression method for file saving (None, "zlib", or "jpeg")
        """
        super().__init__()

        self.stain = stain

        self.bkgd_method = bkgd_method
        self.bkgd_options = bkgd_options
        
        self.style_est_mode = style_est_mode
        self.style_est_options = style_est_options
        self.alpha = alpha
        self.beta = beta
        self.use_nnls = use_nnls

        self.maxworkers = maxworkers
        self.compression = compression

        self.target_style = None

        if not (self.stain == "HE" or self.stain == "DAB"):
            raise Exception(f"{self.stain} is not a valid stain.")
        
    def __repr__(self):
        return "MacenkoPipeline"

    def calculate_style(self, slide, pre_adjustment=None):

        slide.set_adjustment(pre_adjustment)
            
        bgd_est = HSVBackgroundEstimation(slide)
        background_result = bgd_est.compute(method=self.bkgd_method, options=self.bkgd_options)
        I0 = background_result["median"]
        print(f"Estimated I0: {I0}")

        mse = MacenkoStyleEstimation(slide,
                                     I0,
                                     mode=self.style_est_mode,
                                     stain_balancing=True if self.stain == "DAB" else False,
                                     alpha=self.alpha,
                                     beta=self.beta,
                                     use_nnls=self.use_nnls,
                                     options=self.style_est_options,
                                    )
        mse.compute_matrix()
        mse.compute_maxC()

        style = {
            "id": slide.identifier,
            "matrix": mse.matrix_avg,
            "maxC": mse.maxC_avg,
            "I0": I0
        }

        return style

    def calculate_style_average(self, mydict, pre_adjustments=None):
        styles = []

        # Calculate the style of each image
        for key, value in mydict.items():

            style = self.calculate_style(value)
            styles.append(style)

        # Average
        matrices = []
        maxCs = []
        I0s = []
        for style in styles:
            matrices.append(style["matrix"])
            maxCs.append(style["maxC"])
            I0s.append(style["I0"])
        avg_style = {}
        avg_style["matrix"] = np.mean(matrices, axis=0)
        avg_style["maxC"] = np.mean(maxCs, axis=0)
        avg_style["I0"] = np.mean(I0s, axis=0)

        return avg_style

    def save_style(self, style, filename, subfolder=""):
        output_path = os.path.join(self.output_dir, subfolder, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            raise Exception(f"Output style path {output_path} already exists.")

        output_dict = {}
        for key, value in style.items():
            if isinstance(value, np.ndarray):
                output_dict[key] = value.tolist()
            else:
                output_dict[key] = value
            
        with open(output_path, "w") as file:
            json.dump(output_dict, file, indent=4)

        print(f"Saved stain style to {output_path}")

    def load_style(self, path, separate=False):
        with open(path, "r") as file:
            style = json.load(file)
        style["matrix"] = np.array(style["matrix"])
        style["maxC"] = np.array(style["maxC"])
        style["I0"] = np.array(style["I0"])
        if not separate:
            return style
        return style["matrix"], style["maxC"], style["I0"]

    def set_target_from_json(self, target):
        self.target_style = self.load_style(target)

    def transform_source(self, slide, source_style, pre_adjustment=None, post_adjustment=None, subfolder=""):

        slide.set_adjustment(pre_adjustment)
        
        mt = MacenkoTranslation(slide,
                                self.target_style["matrix"],
                                self.target_style["maxC"],
                                self.target_style["I0"], 
                                source_style["matrix"],
                                source_style["maxC"],
                                source_style["I0"],
                                post_adjustment=post_adjustment,
                                use_nnls=self.use_nnls,
                               )

        output_path = os.path.join(self.output_dir, subfolder, slide.identifier + ".tif")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tw = TiledWriter(output_path, slide.shape, slide.tile_size, slide.resolution[0], slide.resolution[1], maxworkers=self.maxworkers, compression=self.compression)
        tw.write(mt.translate)

    def run(self):

        # Calculate target style if needed
        if self.target_style is None:
            print("Calculating target style...")
            self.target_style = self.calculate_style_average(self.target)
            self.save_style(self.target_style, "target.json")

        # Process each source item
        for key, value in self.source.items():
            # value could be a SlideImage or a Subset

            if isinstance(value, Subset):

                print(f"Processing {value.identifier}...")
                
                # Get reference
                reference_style = None
                if value.reference_json is not None:
                    reference_style = self.load_style(value.reference_json)
                elif len(value.reference) > 0:
                    print("Calculating reference style...")
                    pre_adjustments = None
                    if value.global_pre_adjustment is not None:
                        pre_adjustments = value.global_pre_adjustment
                    else:
                        pre_adjustments = value.reference_pre_adjustments
                    reference_style = self.calculate_style_average(value.reference, pre_adjustments=pre_adjustments)
                    self.save_style(reference_style, value.identifier + "_reference.json", subfolder=key if self.save_by_subset else "")

                # Norm
                for subset_k, subset_v in value.images.items():
                    # Get pre-adjustment
                    pre_adjustment = None
                    if subset_k in value.pre_adjustments:
                        pre_adjustment = value.pre_adjustments[subset_k]
                    elif value.global_pre_adjustment is not None:
                        pre_adjustment = value.global_pre_adjustment
                        
                    # Calculate source style
                    source_style = None
                    if reference_style is None:
                        print(f"Calculating source style of {subset_k}...")
                        source_style = self.calculate_style(subset_v, pre_adjustment=pre_adjustment)
                        self.save_style(source_style, subset_k + ".json", subfolder=key if self.save_by_subset else "")
                    else:
                        source_style = reference_style

                    # Get post-adjustment
                    post_adjustment = None
                    if subset_k in value.post_adjustments:
                        post_adjustment = value.post_adjustments[subset_k]
                    elif value.global_post_adjustment is not None:
                        post_adjustment = value.global_post_adjustment

                    # Transform image
                    print(f"Transforming {subset_k}...")
                    self.transform_source(subset_v, source_style, pre_adjustment=pre_adjustment, post_adjustment=post_adjustment, subfolder=key if self.save_by_subset else "")

            elif isinstance(value, SlideImage):

                print(f"Processing {value.identifier}...")

                source_style = self.calculate_style(value)
                self.save_style(source_style, key + ".json")
                self.transform_source(value, source_style)

