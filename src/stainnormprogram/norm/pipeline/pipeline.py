"""

Overall pipeline for normalization:
1. Calculate a target style
2. Calculate a source style
3. Transform the source

"""

class Pipeline:

    def __init__(self):

        self.target = None
        self.source = None

        self.output_dir = None
        self.save_by_subset = None

    def set_target(self, target):
        """
        Args:
            target (dict): Key is ID and value is a path to an image.
        """
        self.target = target

    def set_target_from_json(self, target):
        """Set the target style from a JSON.

        Args:
            target (str): Full path to a JSON.
        """
        raise NotImplementedError

    def set_source(self, source):
        """
        Args:
            source (dict): Key is ID and value is a SlideImage or a Subset.
        """
        self.source = source

    def run(self):
        """Run pipeline
        """
        raise NotImplementedError

    def set_output_dir(self, output_dir, save_by_subset=True):
        """
        Args:
            output_dir (str): Output directory.
            save_by_subset (bool): Within the output_dir, save each Subset into its own subfolder.
        """
        self.output_dir = output_dir
        self.save_by_subset = save_by_subset