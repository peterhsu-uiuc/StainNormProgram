from stainnormprogram.norm.utils import add_images_to_dict
from stainnormprogram.slide import SlideImage, is_valid_extension
from stainnormprogram.norm.adjustments import Adjustment

class Subset:
    """Define a subset of source images. The ID for each image in the subset is the file name, which must be unique.
    """

    def __init__(self, identifier):
        """Create a subset.

        Args:
            identifier (str): A unique name for this Subset.
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

    def __repr__(self):
        output = f"""Subset=(
            identifier={self.identifier}
            images={self.images}
            reference={self.reference}
            reference_json={self.reference_json}
        )
        """
        return output

    def add_images(self, images):
        """
        Args:
            images (str, list): a full path to an image or directory of images, or a list of paths/directories
        """
        add_images_to_dict(images, self.images)

    def set_global_pre_adjustment(self, adjustment):
        if len(self.pre_adjustments) != 0:
            raise Exception
        self.global_pre_adjustment = adjustment

    def set_global_post_adjustment(self, adjustment):
        if len(self.post_adjustments) != 0:
            raise Exception
        self.global_post_adjustment = adjustment
            
    def add_pre_adjustments(self, adjustments):
        """
        Args:
            adjustments (Adjustment, dict): If dict, key is ID and value is Adjustment.
        """
        if self.global_pre_adjustment is not None:
            raise Exception
        self.add_adjustments_to_dict(adjustments, self.pre_adjustments)

    def add_post_adjustments(self, adjustments):
        """
        Args:
            adjustments (Adjustment, dict): If dict, key is ID and value is Adjustment
        """
        if self.global_post_adjustment is not None:
            raise Exception
        self.add_adjustments_to_dict(adjustments, self.post_adjustments)

    def add_adjustments_to_dict(self, adjustment, mydict):
        if isinstance(adjustment, Adjustment):
            if len(mydict) != 1:
                raise Exception
            adjustment = {mydict.keys()[0]: adjustment}
            
        mydict.update(adjustment)

    def add_reference(self, reference, pre_adjustments=None):
        """
        If list, the source style is calculated from all references.
        
        Args:
            reference (str, list): a full path to an image or directory of images, or a list of paths/directories, or a JSON to a reference style
            pre_adjustments (Adjustment, dict): If dict, key is ID and value is Adjustment
        """
        if isinstance(reference, str) and reference.endswith(".json"):
            if len(self.reference) != 0:
                raise Exception
            self.reference_json = reference
            return

        if self.reference_json is not None:
            raise Exception
            
        add_images_to_dict(reference, self.reference)

        if pre_adjustments is not None:
            if self.global_pre_adjustment is not None:
                raise Exception
            self.add_adjustments_to_dict(pre_adjustments, self.reference_pre_adjustments)