import pyvips
import numpy as np

class Adjustment:
 
    def __init__(self, shift: int | None = None, gamma: float | None = None):
        """
        ``Adjustment`` allows you to apply pre- and post-processing operations on images.
        For example, changing the brightness of an image before stain normalization.

        **Note:** Only one type of adjustment is allowed (e.g., do not specify both shift and gamma).

        Args:
            shift: Add/subtract a value from the RGB values (which results in brightening/darkening).
            gamma: Perform a gamma adjustment.
        """

        self.shift = None
        self.gamma = None

        if shift is not None:
            self.shift = shift
        elif gamma is not None:
            self.gamma = gamma

    def apply_shift(self, image: pyvips.Image | np.ndarray) -> pyvips.Image | np.ndarray:
        """Apply a brightness shift.

        Args:
            image: Image to perform the brightness shift on.

        Returns:
            The image (same type as the input) with a brightness shift applied.
        """
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
            raise Exception("image was not a Pyvips image nor a NumPy array.")

        return output

    def apply_gamma(self, image: np.ndarray) -> np.ndarray:
        """Apply a gamma correction.

        Args:
            image: Image to perform the gamma correction on.

        Returns:
            A np.ndarray of the gamma corrected image.
        """
        output = None

        if isinstance(image, np.ndarray):
            output = np.pow(image, self.gamma)
        else:
            raise Exception("image must be a np.ndarray.")

        return output

    def apply(self, image: Any) -> Any:
        """Apply the adjustment to the image.

        Args:
            image: Image to perform the adjustment on.

        Returns:
            The adjusted image.
        """

        output = None

        if self.shift is not None:
            output = self.apply_shift(image)

        elif self.gamma is not None:
            output = self.apply_gamma(image)

        return output
