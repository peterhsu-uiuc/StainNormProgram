from stainnormprogram.slide import SlideImage, SlideImageCollection, is_valid_extension

from stainnormprogram.assessment.color import ColorSample, ColorDistribution, ColorComparison, ColorHistogramEdges
from stainnormprogram.assessment.palette import Palette, ReferencePalette, PaletteEvaluation, PaletteComparison
from stainnormprogram.assessment.structure import StructurePreservation
from stainnormprogram.assessment.contrast import Contrast, ContrastChange
from stainnormprogram.assessment.gamut import Gamut, ReferenceGamut, GamutComparison
from stainnormprogram.assessment.background import BackgroundEstimation, ReferenceBackground, BackgroundComparison
from stainnormprogram.assessment.plot import plot_labels, ScatterPlot, HistPlot

import pickle
import os
import csv
from typing import Any
import json

import plotly

class ReferenceSet:
    def __init__(self, assessment_dir: str):
        """ReferenceSet keeps track of the assessment statistics computed on the reference.

        The reference is what we compare each image to in the Assessment.

        Args:
            assessment_dir: Path to the assessment directory.
        """
        self.identifier = "reference"

        self.assessment_dir = assessment_dir
        self.reference_dir = os.path.join(self.assessment_dir, self.identifier)
        os.makedirs(self.reference_dir, exist_ok=True)

    def get_stat(self, stat_type: str, **kwargs) -> Any:
        """Get a particular statistic.

        Options for ``stat_type``:
            * ColorDistribution
            * ColorComparison
            * ReferenceBackground
            * Contrast
            * ReferenceGamut
            * ReferencePalette
            * PaletteEvaluation

        Args:
            stat_type: The name of the statistic to get.
            kwargs: Additional keyword arguments required by the statistic.

        Returns:
            The requested statistic, or None if the statistic is not available.
        """
        try:
            if stat_type == "ColorDistribution":
                edges = ColorHistogramEdges.load(self.assessment_dir)
                return ColorDistribution.load(self.reference_dir, edges=edges)
            elif stat_type == "ColorComparison":
                return ColorComparison.load(self.reference_dir)
            elif stat_type == "ReferenceBackground":
                return ReferenceBackground.load(self.reference_dir)
            elif stat_type == "Contrast":
                return Contrast.load(self.reference_dir)
            elif stat_type == "ReferenceGamut":
                return ReferenceGamut.load(self.reference_dir)
            elif stat_type == "ReferencePalette":
                return ReferencePalette.load(self.reference_dir)
            elif stat_type == "PaletteEvaluation":
                return PaletteEvaluation.load(self.reference_dir, **kwargs)
            else:
                return Exception(f"{stat_type} is not a valid statistic that can be retrieved.")
        except:
            return None

    def save_stat(self, stat: Any):
        """Save the statistic to disk.

        Args:
            stat: Any statistic object.
        """
        if isinstance(stat, ColorDistribution):
            stat.save(self.reference_dir, include_edges=False)
        else:
            stat.save(self.reference_dir)

    def __repr__(self):
        return f"ReferenceSet(identifier={self.identifier}, reference_dir={self.reference_dir})"

class Subset:

    def __init__(self, assessment_dir: str, group: str | None = None, version: str | None = None, original_version: bool = False):
        """Subset of images for the assessment.

        Assessment Subsets and Normalization Subsets are different.

        Each Subset has three parameters that distinguish it from other Subsets.
            * ``group`` is a group name. This allows you to group images in whatever way you want. One way is to group images by where they came from. For example, one group could be "Hospital A" and another group could be "Hospital B".
            * ``version`` is a version name. This allows you to compare different versions of an image within the Assessment. For example, one Subset could be group="Hospital A" version="Original" and another Subset could be group="Hospital A" version="Normalized".
            * ``original_version`` indicates whether this Subset represents the "Original" version of the images. "Original" means before normalization. Providing this information establishes which version is the baseline to check if morphological structures or color contrast were changed by stain normalization in the other versions.

        An Assessment is thus composed of a collection of Subsets representing different groupings (image sources) and versions (original and different normalization methods to compare).
        An Assessment also has a reference, which acts as the reference to compare all images to when computing assessment metrics/statistics. Any images in any Subset can be marked as reference.

        **Note:** An image ID (filename without extension) cannot appear in more than one group. An image ID must be unique within a group. An image ID can appear in multiple versions.

        Args:
            assessment_dir: Path to the assessment directory.
            group: Group name.
            version: Version name.
            original_version: If True, this Subset is the original version of the images (i.e., before normalization).
        """

        self.assessment_dir = assessment_dir
        if group is None and version is None:
            self.subset_folder = self.assessment_dir
        elif group is None:
            self.subset_folder = os.path.join(self.assessment_dir, f"version_{version}")
        elif version is None:
            self.subset_folder = os.path.join(self.assessment_dir, f"group_{group}")
        else:
            self.subset_folder = os.path.join(self.assessment_dir, f"group_{group}", f"version_{version}")
        os.makedirs(self.subset_folder, exist_ok=True)

        subset_json_path = os.path.join(self.subset_folder, "subset.json")
        if not os.path.exists(subset_json_path):
            with open(subset_json_path, "w") as file:
                json.dump({
                    "group": group,
                    "version": version,
                    "original_version": original_version,
                }, file)

        self.group = group
        self.version = version
        self.original_version = original_version
        
        self.images = {}
        self.reference_ids = []

        for f in os.listdir(self.subset_folder):
            if f.startswith("id_"):
                self.images[f[3:]] = None

    @classmethod
    def load(cls, assessment_dir: str, subset_folder: str) -> Subset:
        """Load a Subset from disk.

        Args:
            assessment_dir: Path to the assessment directory.
            subset_folder: Path to the Subset folder.

        Returns:
            A Subset.
        """

        subset_json_path = os.path.join(subset_folder, "subset.json")
        with open(subset_json_path, "r") as file:
            json_data = json.load(file)
        
        return cls(assessment_dir, group=json_data["group"], version=json_data["version"], original_version=["original_version"])

    def __repr__(self):
        return f"Subset(\n \
        group={self.group}\n \
        version={self.version}\n \
        images={self.images}\n \
        reference_ids={self.reference_ids}\n)"

    def set_all_as_reference(self):
        """Set all images currently in the Subset as reference.
        """
        self.reference_ids = self.images.keys()

    def get_paths_in_folder(self, folder_path: str) -> list[str]:
        """Helper function for getting valid image paths in a folder.

        Args:
            folder_path: Folder path to get image paths from.

        Returns:
            A list of str paths.
        """
        paths = []
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path) and is_valid_extension(full_path):
                paths.append(full_path)
        return paths

    def add_path_to_subset(self, path: str, slide: str | None = None, reference: list[str] | bool = False):
        """Add an image to the Subset.

        To add images in bulk, use ``add_images`` instead.

        Images can be part of a "slide." This is the case when you have multiple images associated with a single microscope slide.
        For example, you have image tiles from a single slide, or one image per core in a tissue microarray (TMA) slide.
        Add each path with the same slide name, and Assessment will associate those images together.

        Args:
            path: File path to the image.
            slide: Slide name if the image is part of a slide that consists of multiple images.
            reference: If True, this image becomes part of the reference. If list of IDs, checks if this image is in the list and becomes a reference only if in the list.
        """

        image_id = is_valid_extension(path, split=True)[0]
        if slide is None:
            self.images[image_id] = SlideImage(path, identifier=image_id)
            if (isinstance(reference, list) and image_id in reference) or (isinstance(reference, bool) and reference):
                self.reference_ids.append(image_id)
        else:
            if slide in self.images:
                self.images[slide].append(path, identifier=image_id)
            else:
                self.images[slide] = SlideImageCollection(slide)
                self.images[slide].append(path, identifier=image_id)
            if (isinstance(reference, list) and slide in reference) or (isinstance(reference, bool) and reference):
                self.reference_ids.append(slide)

    def add_images(self, images: list[str] | str, slide: str | None = None, reference: list[str] | bool = None):
        """Add images to the Subset.

        Images can be part of a "slide." This is the case when you have multiple images associated with a single microscope slide.
        For example, you have image tiles from a single slide, or one image per core in a tissue microarray (TMA) slide.
        Add each path with the same slide name, and Assessment will associate those images together.

        Args:
            images: Image paths.
            slide: Slide name to assign the paths to.
            reference: If True, all images added become part of the reference. If list of IDs, only those images will become part of the reference.
        """

        if isinstance(images, str):
            images = [images]
    
        if isinstance(images, list):
            
            for path in images:
                
                if not isinstance(path, str) or (not os.path.isfile(path) and not os.path.isdir(path)):
                    raise Exception(f"{path} is not a valid path.")
                    
                if os.path.isfile(path):
                    self.add_path_to_subset(path, slide=slide, reference=reference)

                elif os.path.isfile(path) and is_valid_extension(path):
                    self.add_path_to_subset(path, slide=slide, reference=reference)

                elif os.path.isdir(path):
                    for p in self.get_paths_in_folder(path):
                        self.add_path_to_subset(p, slide=slide, reference=reference)

    def get_stat(self, identifier: str, stat_type: str, **kwargs) -> Any:
        """Get a statistic/metric.

        Options for ``stat_type``:
            * ColorSample
            * ColorDistribution
            * ColorComparison
            * BackgroundEstimation
            * Contrast
            * ContrastChange
            * Gamut
            * GamutComparison
            * StructurePreservation
            * Palette
            * PaletteEvaluation
            * PaletteComparison

        Args:
            identifier: Image ID (filename without extension) or slide name.
            stat_type: Name of the statistic to get.
            kwargs: Additional keyword arguments for the statistic.

        Returns:
            The requested statistic, or None if the statistic is not available.
        """

        if identifier not in self.images:
            raise Exception(f"{identifier} is not a valid image or slide in this subset.")

        image_stat_dir = os.path.join(self.subset_folder, f"id_{identifier}")

        try:
            if stat_type == "ColorSample":
                return ColorSample.load(image_stat_dir, edges=ColorHistogramEdges.load(self.assessment_dir))
            elif stat_type == "ColorDistribution":
                return ColorDistriution.load(image_stat_dir, edges=ColorHistogramEdges.load(self.assessment_dir))
            elif stat_type == "ColorComparison":
                return ColorComparison.load(image_stat_dir)
            elif stat_type == "BackgroundEstimation":
                return BackgroundEstimation.load(image_stat_dir)
            elif stat_type == "Contrast":
                return Contrast.load(image_stat_dir)
            elif stat_type == "ContrastChange":
                return ContrastChange.load(image_stat_dir)
            elif stat_type == "Gamut":
                return Gamut.load(image_stat_dir)
            elif stat_type == "GamutComparison":
                return GamutComparison.load(image_stat_dir)
            elif stat_type == "StructurePreservation":
                return StructurePreservation.load(image_stat_dir)
            elif stat_type == "Palette":
                return Palette.load(image_stat_dir)
            elif stat_type == "PaletteEvaluation":
                return PaletteEvaluation.load(image_stat_dir, **kwargs)
            elif stat_type == "PaletteComparison":
                return PaletteComparison.load(image_stat_dir)
            else:
                return ValueError(f"{stat_type} is not a valid statistic.")
        except ValueError as ve:
            raise
        except Exception:
            return None

    def save_stat(self, stat: Any, identifier: str):
        """Save the statistic to disk.

        Args:
            stat: The statistic object.
            identifier: The ID of the image or slide name.
        """

        if identifier not in self.images:
            raise Exception(f"{identifier} is not a valid image or slide in this subset.")

        image_stat_dir = os.path.join(self.subset_folder, f"id_{identifier}")
        os.makedirs(image_stat_dir, exist_ok=True)

        if isinstance(stat, (ColorSample, ColorDistribution)):
            stat.save(image_stat_dir, include_edges=False)
        else:
            stat.save(image_stat_dir)

class Assessment:

    def __init__(self, assessment_dir: str):
        """Stain normalization assessment.

        An Assessment consists of reference and query images. A reference distribution is calculated from the reference images, and this reference distribution is used to compare all images (query and reference images) against.
        Once you compute the assessment metrics on an image, they will be saved to the assessment_dir, which can be reloaded in a later session.

        Args:
            assessment_dir: Assessment directory.
        """

        # Make a directory for saving assessment calculations if it doesn't exist already.
        os.makedirs(assessment_dir, exist_ok=True)
        self.assessment_dir = assessment_dir

        # If color histogram edges exist, load that. Otherwise, make default edges.
        self.edges = ColorHistogramEdges.load(self.assessment_dir)
        if self.edges is None:
            self.edges = ColorHistogramEdges()
            self.edges.save(self.assessment_dir)

        self.reference = None # ReferenceSet
        self.reference_is_set = False
        self.image_sets = []

        # If assessment_dir has files already, load in what Subsets are available.
        if os.path.exists(os.path.join(self.assessment_dir, "subset.json")):
            self.image_sets.append(Subset.load(self.assessment_dir, self.assessment_dir))
        for f in os.listdir(self.assessment_dir):
            path = os.path.join(self.assessment_dir, f)
            if not os.path.isdir(path):
                continue
            if f.startswith("group_"):
                if os.path.exists(os.path.join(path, "subset.json")):
                    self.image_sets.append(Subset.load(self.assessment_dir, path))
                for f2 in os.listdir(path):
                    path2 = os.path.join(path, f2)
                    if os.path.isdir(path2) and f2.startswith("version_"):
                        self.image_sets.append(Subset.load(self.assessment_dir, path2))
            elif f == "reference":
                self.reference = ReferenceSet(self.assessment_dir)
                self.reference_is_set = True
                    
        # Keep track of what metrics have been computed already
        self.computed = set()

        # Plots
        self.plots = {}

    def __repr__(self):
        return f"Assessment(\n \
        assessment_dir={self.assessment_dir}\n \
        reference={self.reference}\n \
        image_sets={self.image_sets}\n \
        computed={self.computed}\n \
        reference_is_set={self.reference_is_set}\n)"

    def add_images(
            self,
            images: list[str] | str,
            slide: str | None = None,
            group: str | None = None,
            version: str | None = None,
            original_version: bool = False,
            reference: str | bool | list[str] | None = None
            ):
        """Add images to perform the assessment on.

        How to set the parameters:
            * If you have images which can be **grouped** in some way (e.g., time period, organ, project), then call ``add_images`` for each group and specify a group name.
            * If you have images where each image might have multiple **versions** (e.g., original, normalized, corrected), then call ``add_images`` for each version and specify a version name.
            * If you have separate images that you want to consider as one **slide** (e.g., tiles from a slide), then specify a slide name. Use a for loop to add images if needed.
            * For the version that is the **original (before normalization) version**, set ``original_version=True``.
            * To perform the assessment, we need a reference to compare all images against. Set ``reference`` to the images you want to be marked as the reference. Good references could be the target or a group of images from the past that were normalized well that you want to get your new normalized images as close to.

        **Note:** If you are using the ``slide`` parameter and want this slide to be a reference, ``reference=True`` or using the slide name str will add all images in the slide as reference.
        To set particular images within the slide as reference, do not use ``slide``; keep each image as separate entities. Using individual image ID names for images in a ``slide`` for setting the reference will not work.

        Args:
            images: Path to an image, directory of images, or list of images/directories.
            slide: If images are crops from a larger slide image, specify a slide name.
            group: Group name. For example, if images are from the same source, group could be the name of the source.
            version: Version name. For example, "original" or "normalized." This allows you to have multiple versions of the same slide.
            original_version: If True, makes this version the original. Only one version within a group can be the original.
            reference: If True, all images being added are set as reference. Or can provide list of IDs (file names without extension) of images or slide names to use as reference. Or can provide a single ID or slide name.
        """
        for image_set in self.image_sets:

            if image_set.group == group and image_set.version == version:
                image_set.add_images(images, slide=slide, reference=reference)
                return

        if original_version:
            for image_set in self.image_sets:
                if image_set.group == group and image_set.version != version and image_set.original_version:
                    raise Exception("Only one version in a group can be the original version.")

        # No matching image sets, make a new set
        image_set = Subset(self.assessment_dir, group=group, version=version, original_version=original_version)
        image_set.add_images(images, slide=slide, reference=reference)
        self.image_sets.append(image_set)

        if reference:
            self.reference_is_set = True

    def get_image_set(self, group: str | None = None, version: str | None = None) -> Subset:
        """Get an image Subset.

        Args:
            group: Group name.
            version: Version name.

        Returns:
            A Subset.
        """
        for image_set in self.image_sets:
            if image_set.group == group and image_set.version == version:
                return image_set
        raise Exception(f"Subset with group {group} and version {version} not found.")

    def write_to_csv(self):
        """Write assessment metrics to a CSV file.
        """
        csv_path = os.path.join(self.output_dir, "results.csv")
        with open(csv_path, mode="w", newline='', encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow(["identifier", "reference", "group", "version", "h_dist", "s_dist", "v_dist", "rgb_dist", "oklab_dist", "palette_compare", "palette_Hdiv", "palette_Ediv", "palette_HEsep", "background", "oog_count", "ig_count", "structure_mean", "structure_std", "contrast_mean", "contrast_std"])

            for image_set in self.image_sets:
                for identifier, img in image_set.images.items():
                    version = image_set.version
                    group = image_set.group
                    is_reference = identifier in image_set.reference_ids

                    row = [identifier, is_reference, group, version]

                    cc = image_set.get_stat("ColorComparison")
                    if cc is not None:
                        row.extend([cc.get_val("h"), cc.get_val("s"), cc.get_val("v"), cc.get_val("rgb"), cc.get_val("oklab")])
                    else:
                        row.extend([None] * 5)

                    pc = image_set.get_stat("PaletteComparison")
                    if pc is not None:
                        row.extend([pc.get_val("compare"), pc.get_val("Hdiv"), pc.get_val("Ediv"), pc.get_val("HEsep")])
                    else:
                        row.extend([None] * 4)

                    bc = image_set.get_stat("BackgroundComparison")
                    if bc is not None:
                        row.extend([bc.get_val()])
                    else:
                        row.extend([None])

                    gc = image_set.get_stat("GamutComparison")
                    if gc is not None:
                        row.extend([gc.get_val("oog_count"), gc.get_val("ig_count")])
                    else:
                        row.extend([None] * 2)

                    s = image_set.get_stat("StructurePreservation")
                    if s is not None:
                        row.extend([s.get_val()[0], s.get_val()[1]])
                    else:
                        row.extend([None] * 2)

                    ct = image_set.get_stat("ContrastChange")
                    if ct is not None:
                        row.extend([ct.get_val()[0], ct.get_val()[1]])
                    else:
                        row.extend([None] * 2)

                    writer.writerow(row)

    def color_distance_compute(self):
        """Compute the color distance assessment.
        """
        
        # Get the reference color distribution if it exists
        cd = self.reference.get_stat("ColorDistribution")
        
        # If the reference color distribution does not exist, we will need to compute it
        ref_color_samples = []

        for image_set in self.image_sets:
            for key, img in image_set.images.items():
                cs = image_set.get_stat(key, "ColorSample")
                if cs is None:
                    cs = ColorSample(img, self.edges)
                    cs.compute()
                    image_set.save_stat(cs, key)
                if cd is None and key in image_set.reference_ids:
                    ref_color_samples.append(cs)

        # Compute the reference color distribution
        if cd is None:
            cd = ColorDistribution(ref_color_samples, identifier=self.reference.identifier)
            cd.compute()
            self.reference.save_stat(cd)
    
        # Compute the comparison between the colors samples and the reference color distribution
        for image_set in self.image_sets:
            for key in image_set.images.keys():
                if image_set.get_stat(key, "ColorComparison") is None:
                    cc = ColorComparison(cd, image_set.get_stat(key, "ColorSample"))
                    cc.compute()
                    image_set.save_stat(cc, key)

    def palette_compute(self):
        """Compute the palette assessment.
        """

        rp = self.reference.get_stat("ReferencePalette")
        rpe = self.reference.get_stat("PaletteEvaluation")

        ref_palettes = []
        
        for image_set in self.image_sets:
            for key, img in image_set.images.items():
                p = image_set.get_stat(key, "Palette")
                if p is None:
                    p = Palette(img)
                    p.compute()
                    image_set.save_stat(p, key)
                pe = image_set.get_stat(key, "PaletteEvaluation", palette=p)
                if pe is None:
                    pe = PaletteEvaluation(p)
                    pe.compute()
                    image_set.save_stat(pe, key)
                if rp is None and key in image_set.reference_ids:
                    ref_palettes.append(p)

        if rp is None:
            rp = ReferencePalette(ref_palettes, identifier=self.reference.identifier)
            rp.compute()
            self.reference.save_stat(rp)
            rpe = PaletteEvaluation(rp)
            rpe.compute()
            self.reference.save_stat(rpe)
        
        for image_set in self.image_sets:
            for key in image_set.images.keys():
                if image_set.get_stat(key, "PaletteComparison") is None:
                    p = image_set.get_stat(key, "Palette")
                    pc = PaletteComparison(rpe, image_set.get_stat(key, "PaletteEvaluation", palette=p))
                    pc.compute()
                    image_set.save_stat(pc, key)
                
    def background_compute(self):
        """Compute the background assessment.
        """

        rb = self.reference.get_stat("ReferenceBackground")

        ref_backgrounds = []

        for image_set in self.image_sets:
            for key, img in image_set.images.items():
                be = image_set.get_stat(key, "BackgroundEstimation")
                if be is None:
                    be = BackgroundEstimation(img)
                    be.compute()
                    image_set.save_stat(be, key)
                if rb is None and key in image_set.reference_ids:
                    ref_backgrounds.append(be)

        if rb is None:
            rb = ReferenceBackground(ref_backgrounds, identifier=self.reference.identifier)
            rb.compute()
            self.reference.save_stat(rb)
        
        for image_set in self.image_sets:
            for key in image_set.images.keys():
                if image_set.get_stat(key, "BackgroundComparison") is None:
                    bc = BackgroundComparison(rb, image_set.get_stat(key, "BackgroundEstimation"))
                    bc.compute()
                    image_set.set_save(bc, key)


    def gamut_compute(self):
        """Compute the gamut assessment.
        """

        rg = self.reference.get_stat("ReferenceGamut")

        ref_gamuts = []

        for image_set in self.image_sets:
            for key, img in image_set.images.items():
                g = image_set.get_stat(key, "Gamut")
                if g is None:
                    g = Gamut(img)
                    g.compute()
                    image_set.save_stat(g, key)
                if rg is None and key in image_set.reference_ids:
                    ref_gamuts.append(g)

        if rg is None:
            rg = ReferenceGamut(ref_gamuts, identifier=self.reference.identifier)
            rg.compute()
            self.reference.save_stat(rg)

        for image_set in self.image_sets:
            for key in image_set.images.keys():
                if image_set.get_stat(key, "GamutComparison") is None:
                    gc = GamutComparison(rg, image_set.get_stat(key, "Gamut"))
                    gc.compute()
                    image_set.set_save(gc, key)
    
    def structure_compute(self, image_sets_grouped: dict[str, tuple[Subset, list[Subset]]]):
        """Compute the structure preservation assessment.

        Args:
            image_sets_grouped: Image Subsets organized as a dictionary. The keys are the group names. Each value is a tuple of the original_version Subset and a list of all other Subsets to compare with.
        """

        for group, (original_version, other_versions) in image_sets_grouped.items():

            if original_version is None or len(other_versions) == 0:
                continue

            for other_version in other_versions:
                for key, img in other_version.images.items():
                    sp = other_version.get_stat(key, "StructurePreservation")
                    if sp is None and key in original_version.images:
                        sp = StructurePreservation(original_version.images[key], img)
                        sp.compute()
                        other_version.save_stat(sp, key)

    def contrast_compute(self, image_sets_grouped):
        """Compute the contrast assessment.

        Args:
            image_sets_grouped: Image Subsets organized as a dictionary. The keys are the group names. Each value is a tuple of the original_version Subset and a list of all other Subsets to compare with.
        """

        for group, (original_version, other_versions) in image_sets_grouped.items():

            if original_version is None or len(other_versions) == 0:
                continue

            for other_version in other_versions:
                for key, img in other_version.images.items():
                    original_contrast = original_version.get_stat(key, "Contrast")
                    if original_contrast is None and key in original_version.images:
                        original_contrast = Contrast(original_version.images[key])
                        original_contrast.compute()
                        original_version.save_stat(original_contrast, key)
                    other_contrast = other_version.get_stat(key, "Contrast")
                    if other_contrast is None:
                        other_contrast = Contrast(img, tile_locs=original_contrast.tile_locs)
                        other_contrast.compute()
                        other_version.save_stat(other_contrast, key)
                    cc = other_version.get_stat(key, "ContrastChange")
                    if cc is None and other_contrast is not None and original_contrast is not None:
                        cc = ContrastChange(original_contrast, other_contrast)
                        cc.compute()
                        other_version.save_stat(cc, key)

    def create_traces(self, stat_type: str, plot_type: str) -> dict[str | None, list[tuple[str, str, float]]]:
        """Create traces for graphing.

        Args:
            stat_type: Statistic name.
            plot_type: Plot name (within the statistic).

        Returns:
            Traces as a dictionary with each key being a version (one trace) and value being a list of data points. Each data point is a tuple consisting of (identifier, group, value).
        """

        traces = {}
        for image_set in self.image_sets:
            for identifier in image_set.images.keys():
                version = image_set.version
                group = image_set.group
                stat = image_set.get_stat(identifier, stat_type)
                if stat is None:
                    continue
                value = stat.get_val(plot_type)
                entry = (identifier, group, value)
                if version not in traces:
                    traces[version] = [entry]
                else:
                    traces[version].append(entry)

        return traces

    def color_distance(
            self,
            plot_type: str = "h",
            threshold: float | None = None,
            sort: bool = True,
            sort_by: str | None = None,
            clip_ids: bool = False,
            hide_ids: bool = False,
            save_path: str | None = None,
            ) -> plotly.graph_objects.Figure:
        """Create a color distance plot.

        Args:
            plot_type: Color distance plot type ("h", "s", "v", "rgb", "oklab").
            threshold: Value for a horizontal threshold line drawn on the graph.
            sort: Sort points. If True, set sort_by.
            sort_by: Version name to sort by. If None and sort=True, the None version is used.
            clip_ids: Clip IDs if they are too long.
            hide_ids: Hide IDs.
            save_path: Path to save the figure.

        Returns:
            The requested Plotly figure.
        """

        if "color_dist" not in self.computed:
            raise Exception("Call compute(criteria=\"color_dist\") first.")

        plot_name = "color_dist_" + plot_type

        if plot_name not in self.plots:
            traces = self.create_traces("ColorComparison", plot_type)
            scatter = ScatterPlot(traces, yaxis_title=plot_labels[plot_name][0], title=plot_labels[plot_name][1])
            self.plots[plot_name] = scatter
        
        if save_path is not None:
            self.plots[plot_name].save_plot(save_path=save_path, threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)

        return self.plots[plot_name].make_plot(threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)

    def color_hist(
            self,
            ids: str | list [str],
            plot_type: str = "h",
            groups: str | list[str] | None = None,
            versions: list[str] | str | None = None,
            reference: bool = False,
            density: bool = True,
            color: list[str] | str | None = None,
            hide_axes: bool = False,
            ) -> plotly.graph_objects.Figure :
        """Create a color histogram for a single slide or multiple slides.

        If you have used group and version names in the Assessment, you should specify the ``groups`` and ``versions`` parameters so it is unambiguous which images you are referring to.
        If you want to refer to an image that has no group name or no version name, its group name or version name is ``None`` (NoneType).

        Args:
            ids: One or more IDs (or slide names) of the images to be represented in the color histogram.
            plot_type: Histogram plot type to show ("h", "s", "v", "rgb", "oklab", "uv").
            groups: A group name (if ids is set to only one image) or a list of group names corresponding to the list of ids provided.
            versions: A version name (if ids is set to only one image) or a list of version names corresponding to the list of ids provided.
            reference: If True, includes the reference distribution in the plot.
            density: If True, plot as a density. Otherwise, plots as counts.
            color: A color (if ids is set to only one image) or a list of colors corresponding to the list of ids provided. These colors will be used for the plot. Only for ``plot_type="uv"``.
            hide_axes: Hide axes. Only for ``plot_type="uv"``.

        Returns:
            The requested Plotly figure.
        """

        cs_cd = []

        if not isinstance(ids, list):
            ids = [ids]
        if not isinstance(groups, list):
            groups = [groups]
        if not isinstance(versions, list):
            versions = [versions]

        if groups is None:
            groups = [None] * len(ids)
        if versions is None:
            versions = [None] * len(ids)

        for i, img_id in enumerate(ids):
            cs = self.get_image_set(group=groups[i], version=versions[i]).get_stat(img_id, "ColorSample", edges=self.edges)
            cs_cd.append(cs)

        if reference:
            cd = self.reference.get_stat("ColorDistribution", edges=self.edges)
            cs_cd.append(cd)

        if len(cs_cd) == 1:
            color_hist_plot = HistPlot(cs_cd[0])

            if plot_type == "h":
                return color_hist_plot.h_hist_plot(density=density)
            elif plot_type == "s":
                return color_hist_plot.s_hist_plot(density=density)
            elif plot_type == "v":
                return color_hist_plot.v_hist_plot(density=density)
            elif plot_type == "rgb":
                return color_hist_plot.rgb_hist_plot(density=density)
            elif plot_type == "oklab":
                return color_hist_plot.oklab_hist_plot(density=density)
            elif plot_type == "uv":
                return color_hist_plot.uv_hist(color=color, density=density, hide_axes=hide_axes)
            else:
                raise Exception(f"{plot_type} is not a valid plot_type.")

        else:
            trace_labels = []
            for i, img_id in enumerate(ids):
                trace_label = img_id
                if versions[i] is not None:
                    trace_label += f" {versions[i]}"
                trace_labels.append(trace_label)
            if reference:
                trace_labels.append(self.reference.identifier)
                
            color_hist_plot = HistPlot(cs_cd, trace_labels=trace_labels)

            if plot_type == "h":
                return color_hist_plot.h_hist_compare(density=density)
            elif plot_type == "s":
                return color_hist_plot.s_hist_compare(density=density)
            elif plot_type == "v":
                return color_hist_plot.v_hist_compare(density=density)
            elif plot_type == "uv":
                return color_hist_plot.uv_hist(color=color, density=density, hide_axes=hide_axes)
            else:
                raise Exception(f"{plot_type} is not a valid plot_type.")

    def palette(
        self,
        plot_type: str = "compare",
        threshold: float | None = None,
        sort: bool = True,
        sort_by: str | None = None, 
        clip_ids: bool = False, 
        hide_ids: bool = False, 
        save_path: str | None = None
        ) -> plotly.graph_objects.Figure:
        """Create palette plot.

        Options for ``plot_type``:
            * "compare": Comparison between each query palette with the reference distribution's palette.
            * "Hdiv": Hematoxylin diversity within each query palette.
            * "Ediv": Eosin diversity within each query palette.
            * "HEsep": Separability (difference) between the hematoxylin and eosin colors within each query palette.

        Args:
            plot_type: Palette plot type ("compare", "Hdiv", "Ediv", or "HEsep").
            threshold: Value for a horizontal threshold line drawn on the graph.
            sort: Sort points. If True, set sort_by.
            sort_by: Version name to sort by. If None and sort=True, the None version is used.
            clip_ids: Clip IDs if they are too long.
            hide_ids: Hide IDs.
            save_path: Path to save the figure.

        Returns:
            The requested Plotly figure.
        """

        if "palette" not in self.computed:
            raise Exception("Call compute(criteria=\"palette\") first.")

        plot_name = "palette_" + plot_type

        if plot_name not in self.plots:
            traces = self.create_traces("PaletteComparison", plot_type)
            scatter = ScatterPlot(traces, yaxis_title=plot_labels[plot_name][0], title=plot_labels[plot_name][1])
            self.plots[plot_name] = scatter

        if save_path is not None:
            self.plots[plot_name].save_plot(save_path=save_path, threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)
            
        return self.plots[plot_name].make_plot(threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)


    def display_palette(self, ids: str | list[str], groups: str | list[str] | None = None, versions: str | list[str] | None = None, reference: bool = False):
        """Display palettes.

        Args:
            ids: One or more IDs (or slide names) of the images to show the palette of.
            groups: A group name (if ids is set to only one image) or a list of group names corresponding to the list of ids provided.
            versions: A version name (if ids is set to only one image) or a list of version names corresponding to the list of ids provided.
            reference: If True, includes the reference distribution palette.
        """
        
        palettes = []

        if not isinstance(ids, list):
            ids = [ids]
        
        if groups is None:
            groups = [None] * len(ids)
        if versions is None:
            versions = [None] * len(ids)

        if not isinstance(groups, list):
            groups = [groups]
        if not isinstance(versions, list):
            versions = [versions]

        for i, img_id in enumerate(ids):
            p = self.get_image_set(group=groups[i], version=versions[i]).get_stat(img_id, "Palette")
            palettes.append(p)

        if reference:
            p = self.reference.get_stat("Palette")
            palettes.append(p)

        for p in palettes:
            print(p.identifier)
            p.display()

    def gamut(
        self,
        plot_type: str = "oog_count",
        threshold: float | None = None,
        sort: bool = True,
        sort_by: str | None = None,
        clip_ids: bool = False,
        hide_ids: bool = False,
        save_path: str = None
        ) -> plotly.graph_objects.Figure:
        """Create gamut plot.

        Options for ``plot_type``:
            * "oog_count": Out-of-gamut color count.
            * "ig_count": In-gamut color count.

        Args:
            plot_type: Gamut plot type ("oog_count", "ig_count").
            threshold: Value for a horizontal threshold line drawn on the graph.
            sort: Sort points. If True, set sort_by.
            sort_by: Version name to sort by. If None and sort=True, the None version is used.
            clip_ids: Clip IDs if they are too long.
            hide_ids: Hide IDs.
            save_path: Path to save the figure.

        Returns:
            The requested Plotly figure.
        """

        if "gamut" not in self.computed:
            raise Exception("Call compute(criteria=\"gamut\") first.")

        plot_name = "gamut_" + plot_type

        if plot_name not in self.plots:
            traces = self.create_traces("GamutComparison", plot_type)
            scatter = ScatterPlot(traces, yaxis_title=plot_labels[plot_name][0], title=plot_labels[plot_name][1])
            self.plots[plot_name] = scatter

        if save_path is not None:
            self.plots[plot_name].save_plot(save_path=save_path, threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)

        return self.plots[plot_name].make_plot(threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)

    def background(
        self,
        threshold: float | None = None,
        sort: bool = True,
        sort_by: str | None = None,
        clip_ids: bool = False,
        hide_ids: bool = False,
        save_path: str = None
        ) -> plotly.graph_objects.Figure:
        """Create background plot.

        Args:
            threshold: Value for a horizontal threshold line drawn on the graph.
            sort: Sort points. If True, set sort_by.
            sort_by: Version name to sort by. If None and sort=True, the None version is used.
            clip_ids: Clip IDs if they are too long.
            hide_ids: Hide IDs.
            save_path: Path to save the figure.

        Returns:
            The requested Plotly figure.
        """

        if "background" not in self.computed:
            raise Exception("Call compute(criteria=\"background\") first.")

        plot_name = "background"

        if plot_name not in self.plots:
            traces = self.create_traces("BackgroundComparison", "dist")
            scatter = ScatterPlot(traces, yaxis_title=plot_labels[plot_name][0], title=plot_labels[plot_name][1])
            self.plots[plot_name] = scatter

        if save_path is not None:
            self.plots[plot_name].save_plot(save_path=save_path, threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)

        return self.plots[plot_name].make_plot(threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)

    def contrast(
        self,
        threshold: float | None = None,
        sort: bool = True,
        sort_by: str | None = None,
        clip_ids: bool = False,
        hide_ids: bool = False,
        save_path: str = None
        ) -> plotly.graph_objects.Figure:
        """Create contrast change plot.

        Args:
            threshold: Value for a horizontal threshold line drawn on the graph.
            sort: Sort points. If True, set sort_by.
            sort_by: Version name to sort by. If None and sort=True, the None version is used.
            clip_ids: Clip IDs if they are too long.
            hide_ids: Hide IDs.
            save_path: Path to save the figure.

        Returns:
            The requested Plotly figure.
        """

        if "contrast" not in self.computed:
            raise Exception("Call compute(criteria=\"contrast\") first.")

        plot_name = "contrast"

        if plot_name not in self.plots:
            traces = self.create_traces("ContrastChange", "dist")
            scatter = ScatterPlot(traces, yaxis_title=plot_labels[plot_name][0], title=plot_labels[plot_name][1])
            self.plots[plot_name] = scatter

        if save_path is not None:
            self.plots[plot_name].save_plot(save_path=save_path, threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)

        return self.plots[plot_name].make_plot(threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)

    def structure(
        self,
        threshold: float | None = None,
        sort: bool = True,
        sort_by: str | None = None,
        clip_ids: bool = False,
        hide_ids: bool = False,
        save_path: str = None
        ) -> plotly.graph_objects.Figure:
        """Create structure preservation plot.

        Args:
            threshold: Value for a horizontal threshold line drawn on the graph.
            sort: Sort points. If True, set sort_by.
            sort_by: Version name to sort by. If None and sort=True, the None version is used.
            clip_ids: Clip IDs if they are too long.
            hide_ids: Hide IDs.
            save_path: Path to save the figure.

        Returns:
            The requested Plotly figure.
        """

        if "structure" not in self.computed:
            raise Exception("Call compute(criteria=\"structure\") first.")

        plot_name = "structure"

        if plot_name not in self.plots:
            traces = self.create_traces("StructurePreservation", "dist")
            scatter = ScatterPlot(traces, yaxis_title=plot_labels[plot_name][0], title=plot_labels[plot_name][1])
            self.plots[plot_name] = scatter

        if save_path is not None:
            self.plots[plot_name].save_plot(save_path=save_path, threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)

        return self.plots[plot_name].make_plot(threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)

    def original_version_check(self) -> dict[str, tuple[Subset, list[Subset]]]:
        """Group image Subsets by version and check if each group has an original_version.

        This check is needed for the contrast change and structure preservation assessments since both compare the stain normalized versions with the original_version.
        In the case of contrast change, we are interested in whether we had a contrast gain/loss compared to the original_version. In the case of structure preservation,
        we are interested in whether the stain normalized versions changed the morphology of the tissue compared to the original_version.

        The check does not raise an Exception. It only prints a warning message.

        Returns:
            Image Subsets organized as a dictionary. The keys are the group names. Each value is a tuple of the original_version Subset and a list of all other Subsets to compare with.
        """

        image_sets_grouped = {}

        for image_set in self.image_sets:
            group = image_set.group
            original_version = image_set.original_version
            if group not in image_sets_grouped:
                image_sets_grouped[group] = (None, [])
            if original_version:
                image_sets_grouped[group][0] = image_set
            else:
                image_sets_grouped[group][1].append(image_set)

        warning = []
        for group, (original_version, other_versions) in image_sets_grouped.items():
            if original_version is None and len(other_versions) > 1:
                warning.append(group)
        if len(warning) > 0:
            print("WARNING: You have one or more groups where a version hasn't been marked as original. When using add_images, one version should have original_version=True.")
            print(f"The groups without a version marked as original are: {warning}")
            print("Criteria that depend on an original version being marked will be skipped for the above groups.")

        return image_sets_grouped

    
    def compute(self, criteria: list[str] | str = ["color_dist", "palette", "gamut", "background", "structure", "contrast"]):
        """Compute the assessment on one or more criteria.

        ``criteria`` that can be computed:
            * "color_dist": Distance between color distributions.
            * "palette": Metrics on the color palettes.
            * "gamut": Counts of out-of-gamut and in-gamut colors.
            * "background": Distance between background colors.
            * "structure": Analysis of structural changes (e.g., morphology changes due to normalization).
            * "constrast": Analysis of color contrast changes.

        Args:
            criteria: Criterion/critiera to compute. Can be a single str for one criterion or a list of str for multiple.
        """

        if isinstance(criteria, str):
            criteria = [criteria]

        if self.reference is None:
            self.reference = ReferenceSet(self.assessment_dir)

        if not self.reference_is_set:
            for image_set in self.image_sets:
                image_set.set_all_as_reference()
            self.reference_is_set = True

        if "color_dist" in criteria:
            self.color_distance_compute()
            self.computed.add("color_dist")

        if "palette" in criteria:
            self.palette_compute()
            self.computed.add("palette")

        if "gamut" in criteria:
            self.gamut_compute()
            self.computed.add("gamut")
        
        if "background" in criteria:
            self.background_compute()
            self.computed.add("background")

        if "structure" in criteria:
            image_sets_grouped = self.original_version_check()
            self.structure_compute(image_sets_grouped)
            self.computed.add("structure")

        if "contrast" in criteria:
            image_sets_grouped = self.original_version_check()
            self.contrast_compute(image_sets_grouped)
            self.computed.add("contrast")

        print("Finished Assessment computation.")
