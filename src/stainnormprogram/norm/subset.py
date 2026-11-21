from stainnormprogram.norm.utils import add_images_to_dict
from stainnormprogram.slide import SlideImage, is_valid_extension
from stainnormprogram.norm.adjustments import Adjustment

class Subset:
    
    def __init__(self, identifier: str):
        """Define a Subset of source images. Subsets enable greater control over the stain normalization process.
        
        Args:
            identifier: A unique name for this Subset.
        """
        
        self.identifier = identifier
        self.images = {}
        self.reference = {}
        self.reference_json = None
        self.reference_pre_adjustments = {}
        self.pre_adjustments = {}
        self.post_adjustments = {}
        self.global_pre_adjustment = None
        self.global_post_adjustment = None
        self.global_pre_adjustment_includes_reference = None

    def __repr__(self):
        return f"Subset=(\n \
        identifier={self.identifier}\n \
        images={self.images}\n \
        reference={self.reference}\n \
        reference_json={self.reference_json}\n)"

    def add_images(self, images: str | list[str]):
        """Add images to the Subset.

        You can call this function multiple times, which will accumulate images in the Subset.

        Args:
            images: a full path to an image or directory of images, or a list of paths/directories
        """
        add_images_to_dict(images, self.images)

    def set_global_pre_adjustment(self, adjustment: Adjustment | None, include_reference: bool = True):
        """Set a pre-adjustment for all images in this Subset.

        Args:
            adjustment: The Adjustment to apply to all images in this Subset.
            include_reference: Apply the global pre-adjustment to the Subset's reference as well.
        """
        if len(self.pre_adjustments) != 0:
            raise Exception("You already have individual image pre-adjustments set. You cannot have a global pre-adjustment set at the same time.")
        self.global_pre_adjustment = adjustment
        self.global_pre_adjustment_includes_reference = include_reference

    def set_global_post_adjustment(self, adjustment: Adjustment | None):
        """Set a post-adjustment for all images in this Subset.

        Args:
            adjustment: The Adjustment to apply to all images in this Subset.
        """
        if len(self.post_adjustments) != 0:
            raise Exception("You already have individual image post-adjustments set. You cannot have a global post-adjustment set at the same time.")
        self.global_post_adjustment = adjustment
            
    def add_pre_adjustments(self, adjustments: Adjustment | dict[str, Adjustment]):
        """Add individual image pre-adjustments to this Subset.

        Args:
            adjustments: If dict, key is the ID of the image (filename without extension) and value is Adjustment. If just Adjustment (not a dict), only one image can be in this Subset.
        """
        if self.global_pre_adjustment is not None:
            raise Exception("You already have a global pre-adjustment set. You cannot have individual image pre-adjustments set at the same time.")
        self.add_adjustments_to_dict(adjustments, self.pre_adjustments, self.images)

    def add_post_adjustments(self, adjustments: Adjustment | dict[str, Adjustment]):
        """Add individual image post-adjustments to this Subset.

        Args:
            adjustments: If dict, key is the ID of the image (filename without extension) and value is Adjustment. If just Adjustment (not a dict), only one image can be in this Subset.
        """
        if self.global_post_adjustment is not None:
            raise Exception("You already have a global post-adjustment set. You cannot have individual image post-adjustments set at the same time.")
        self.add_adjustments_to_dict(adjustments, self.post_adjustments, self.images)

    def add_adjustments_to_dict(self, adjustment: Adjustment | dict[str, Adjustment], mydict: dict[str, Adjustment], images_dict: dict[str, SlideImage]):
        """Helper function to add Adjustments to a dictionary.

        Args:
            adjustments: If dict, key is the ID of the image (filename without extension) and value is Adjustment. If just Adjustment (not a dict), only one image can be in this Subset.
            mydict: Dictionary of Adjustments.
            images_dict: Dictionary of SlideImages.
        """
        if isinstance(adjustment, Adjustment):
            if len(images_dict) > 1:
                raise Exception("You have multiple images. Please provide adjustments as a dict with the key being the ID of the image (filename without extension) and value being the Adjustment.")
            if len(images_dict) == 0:
                raise Exception("Add images first before adding adjustments.")
            adjustment = {images_dict.keys()[0]: adjustment}
            
        mydict.update(adjustment)

    def add_reference(self, reference: str | list[str], pre_adjustments: Adjustment | dict[str, Adjustment] | None = None):
        """Add reference image(s) for this source Subset. This is not the same as a target image.

        When a reference is added, the source stain style for this Subset is estimated from the reference. The color transformation is then performed on the images added to the Subset.
        This allows you to have fine-grained control over which images are used for style estimation and which images are color transformed.

        The images added to the Subset can be the same as the images added as reference (i.e., when using multiple images to calculate an average source style).
        
        Args:
            reference: Path to an image or directory of images, or a list of paths/directories, or a JSON to a reference style.
            pre_adjustments: If dict, key is the ID of the image (filename without extension) and value is Adjustment. If just Adjustment (not a dict), only one image can be the reference.
        """
        if isinstance(reference, str) and reference.endswith(".json"):
            if len(self.reference) != 0:
                raise Exception("A reference was already set.")
            self.reference_json = reference
            return

        if self.reference_json is not None:
            raise Exception("A reference JSON was already set.")
            
        add_images_to_dict(reference, self.reference)

        if pre_adjustments is not None:
            if self.global_pre_adjustment is not None and self.global_pre_adjustment_includes_reference:
                raise Exception("A global pre-adjustment that includes the reference was already set. Set includes_reference=False in the set_global_pre_adjustment function to specify a different adjustment.")
            self.add_adjustments_to_dict(pre_adjustments, self.reference_pre_adjustments, self.reference)
