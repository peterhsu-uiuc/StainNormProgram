import pyvips
import numpy as np

class Adjustment:
    """Adjustment operations.

    Sometimes, it may be desirable to perform pre- or post-processing operations on the images. Such operations may be required by multiple parts of the normalization pipeline. For example, if you want to pre-brighten an image but do not want to spend I/O time and disk memory writing a brightened version of the image, you can create an Adjustment, which applies pre-brightening whenever the pipeline retrieves pixels from the image.
    
    """

    def __init__(self, shift=None, gamma=None):
        """Create an adjustment.

        Adjustment only allows for one type of adjustment.

        Args:
            shift (int): Add/subtract a value from the RGB values (which results in brightening/darkening).
            gamma (float): Perform a gamma adjustment.
        """

        self.shift = None
        self.gamma = None

        if shift is not None:
            self.shift = shift
        elif gamma is not None:
            self.gamma = gamma
        else:
            raise Exception
        
    def apply_shift(self, image):

        output = None

        if isinstance(image, pyvips.Image):
            output = image.cast("float")
            output = output + self.shift
            output = output.clamp(min=0, max=255)
            output = output.cast("uchar")
        elif isinstance(image, np.ndarray):
            output = image.astype(np.float32)
            output = output + self.shift
            output = np.clip(output, 0, 255)
            output = output.astype(np.uint8)
        else:
            raise Exception

        return output

    def apply_gamma(self, image):

        output = None

        if isinstance(image, np.ndarray):
            output = np.pow(image, self.gamma)
        else:
            raise Exception

        return output

    def apply(self, image):

        output = None

        if self.shift is not None:
            output = self.apply_shift(image)

        elif self.gamma is not None:
            output = self.apply_gamma(image)

        return output

            

            
            

