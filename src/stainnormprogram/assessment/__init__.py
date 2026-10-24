from stainnormprogram.slide import SlideImage, SlideImageCollection, is_valid_extension
from stainnormprogram.assessment.color import ColorSample, ColorDistribution, ColorComparison
from stainnormprogram.assessment.palette import Palette, ReferencePalette, PaletteEvaluation, PaletteComparison
from stainnormprogram.assessment.structure import StructurePreservation
from stainnormprogram.assessment.contrast import Contrast, ContrastChange
from stainnormprogram.assessment.gamut import Gamut, ReferenceGamut, GamutComparison
from stainnormprogram.assessment.background import BackgroundEstimation, ReferenceBackground, BackgroundComparison
from stainnormprogram.assessment.plot import plot_labels, ScatterPlot, HistPlot

import pickle
import os
import csv
from typing import Union, List, Optional

class ReferenceSet:
    def __init__(self, identifier, from_snp_ref=False):
        self.identifier = identifier
        self.from_snp_ref = from_snp_ref

        self.stats = {
            "color_distribution": None,
            "palette": None,
            "palette_evaluation": None,
            "gamut": None,
            "background": None,
        }

    def set_stat(self, name, stat, compute=True):
        self.stats[name] = stat
        if compute:
            self.stats[name].compute()
        self.from_snp_ref = False

    def get_stat(self, name):
        if name in self.stats:
            return self.stats[name]
        return None

    def __repr__(self):
        output = f"ReferenceSet(identifier={self.identifier}, from_snp_ref={self.from_snp_ref})"
        return output

class AssessmentSet:

    def __init__(self, identifier, from_snp=False):
        self.identifier = identifier
        self.from_snp = from_snp

        self.slide_level_stats = {
            "color_sample": None,
            "palette": None,
            "palette_evaluation": None,
            "gamut": None,
            "background": None,
            "structure": None,
            "contrast": None,
            "contrast_change": None,
        }

        self.comparative_stats = {
            "color_comparison": None,
            "palette_comparison": None,
            "gamut_comparison": None,
            "background_comparison": None,
        }

    def set_stat(self, name, stat, compute=True):
        if name in self.slide_level_stats:
            self.slide_level_stats[name] = stat
            self.from_snp = False
        elif name in self.comparative_stats:
            self.comparative_stats[name] = stat
        else:
            raise Exception(f"{name} is not a valid AssessmentSet statistic.")
        if compute:
            stat.compute()
        
    def get_stat(self, name):
        if name in self.slide_level_stats:
            return self.slide_level_stats[name]
        elif name in self.comparative_stats:
            return self.comparative_stats[name]
        raise Exception(f"{name} is not a valid AssessmentSet statistic.")

    def __repr__(self):
        output = f"AssessmentSet(identifier={self.identifier}, from_snp={self.from_snp})"
        return output

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["comparative_stats"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.comparative_stats = {
            "color_comparison": None,
            "palette_comparison": None,
            "gamut_comparison": None,
            "background_comparison": None,
        }

        
class Subset:
    """Subset of images for the assessment.
    """

    def __init__(self, group=None, version=None, original_version=False):

        self.group = group
        self.version = version
        self.original_version = original_version

        self.set_images = {}
        self.assessment_sets = {}
        self.reference_ids = []

    def __repr__(self):
        output = f"""Subset(
            group={self.group}
            version={self.version}
            set_images={self.set_images}
            assessment_sets={self.assessment_sets}
            reference_ids={self.reference_ids}
        )
        """
        return output

    def set_all_as_reference(self):
        self.reference_ids = self.set_images.keys()

    def load_snp_file(self, path):
        contents = None
        with open(path, "rb") as file:
            contents = pickle.load(file)
        return contents

    def get_paths_in_folder(self, folder_path):
        paths = []
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path) and (is_valid_extension(full_path) or full_path.endswith(".snp")):
                paths.append(full_path)
        return paths

    def add_path_to_subset(self, path, slide=None, reference=None):

        if path.endswith(".snp"):
            snp_contents = self.load_snp_file(path)
            snp_contents.from_snp = True
            identifier = snp_contents.identifier
            if identifier in self.set_images:
                return
            if slide is not None:
                raise Exception
            self.assessment_sets[identifier] = snp_contents
            if (isinstance(reference, list) and identifier in reference) or (isinstance(reference, bool) and reference):
                self.reference_ids.append(identifier)
            return
        
        image_id = is_valid_extension(path, split=True)[0]
        if slide is None:
            self.set_images[image_id] = SlideImage(path, identifier=image_id)
            if image_id not in self.assessment_sets:
                self.assessment_sets[image_id] = AssessmentSet(image_id)
                if (isinstance(reference, list) and image_id in reference) or (isinstance(reference, bool) and reference):
                    self.reference_ids.append(image_id)
        else:
            if slide in self.set_images:
                self.set_images[slide].append(path, identifier=image_id)
            else:
                self.set_images[slide] = SlideImageCollection(slide)
                self.set_images[slide].append(path, identifier=image_id)
            if slide not in self.assessment_sets:
                self.assessment_sets[slide] = AssessmentSet(slide)
                if (isinstance(reference, list) and slide in reference) or (isinstance(reference, bool) and reference):
                    self.reference_ids.append(slide)

    def add_images(self, images, slide=None, reference=None):

        if isinstance(images, str):
            images = [images]
    
        if isinstance(images, list):
            
            for path in images:
                
                if not isinstance(path, str) or (not os.path.isfile(path) and not os.path.isdir(path)):
                    raise Exception
                    
                if os.path.isfile(path) and path.endswith(".snp"):
                    self.add_path_to_subset(path, slide=slide, reference=reference)

                elif os.path.isfile(path) and is_valid_extension(path):
                    self.add_path_to_subset(path, slide=slide, reference=reference)

                elif os.path.isdir(path):
                    for p in self.get_paths_in_folder(path):
                        self.add_path_to_subset(p, slide=slide, reference=reference)

    

class Assessment:
    """Stain normalization assessment.

    An Assessment consists of reference and query images. Once you compute the assessment metrics on an image, you can save them as .snp files for faster loading later.
    """

    def __init__(self, output_dir: str):
        """Creates an Assessment.

        Args:
            output_dir: Output directory. Must not exist; will be created.
        """

        if os.path.exists(output_dir):
            raise Exception(f"{output_dir} already exists. Specify a new output directory that does not exist.")
        os.makedirs(output_dir, exist_ok=False)
        self.output_dir = output_dir

        self.reference = None # ReferenceSet
        self.reference_is_set = False
        self.image_sets = []

        self.computed = set()

        # Plots
        self.plots = {}

    def __repr__(self):
        output = f"""Assessment(
            output_dir={self.output_dir}
            reference={self.reference}
            image_sets={self.image_sets}
            computed={self.computed}
        )
        """
        return output

    def set_reference_from_file(self, reference: str):
        """Set reference (optional).

        Args:
            reference: .snp_ref file path
        """
        # Check if .snp_ref
        if isinstance(reference, str) and reference.endswith(".snp_ref"):
            with open(reference, "rb") as file:
                self.reference = pickle.load(file)
                self.reference.from_snp_ref = True
                self.reference_is_set = True
                return
        else:
            raise Exception

    def add_images(self, images: Union[List[str], str], slide: str = None, group: str = None, version: str = None, original_version: bool = False, reference: Union[str, bool, List[str]] = None):
        """Add images to perform the assessment on.

        Add images to perform the assessment on. If you have images which can be grouped in some way (e.g., time period, organ, project), then call add_images for each group and specify group. If you have images where each image might have multiple versions (e.g., original, normalized, corrected), then call add_images for each version and specify version. If you have separate images that you want to consider as one slide (e.g., tiles from a slide), then specify a slide name. Use a for loop to add images if needed.

        If group is not specified, added images will be assigned to the appropriate group if their ID has been added already. An exception is thrown if you attempt to assign an ID to multiple groups.

        Args:
            images: Path to an image, directory of images, or list of images/directories (an image could be an image file or a .snp file).
            slide: If images are crops from a larger slide image, specify a slide name.
            group: Group name. For example, if images are from the same source, group could be the name of the source.
            version: Version name. For example, "original" or "normalized." This allows you to have multiple versions of the same slide.
            original_version: If True, makes this version the original. Only one version within a group can be the original.
            reference: If True, all images being added are set as reference. Or can provide list of IDs (file names without extension) of images to use as reference or a single ID.
        """
        for image_set in self.image_sets:

            if image_set.group == group and image_set.version == version:
                image_set.add_images(images, slide=slide, reference=reference)
                return

        if original_version:
            for image_set in self.image_sets:
                if image_set.group == group and image_set.version != version and original_version:
                    raise Exception("Only one version in a group can be the original version.")

        # No matching image sets, make a new set
        image_set = Subset(group=group, version=version, original_version=original_version)
        image_set.add_images(images, slide=slide, reference=reference)
        self.image_sets.append(image_set)

        if reference:
            self.reference_is_set = True

    def get_assessment_set(self, group=None, version=None, identifier=None):
        for image_set in self.image_sets:
            if image_set.group == group and image_set.version == version:
                return image_set.assessment_sets[identifier]

    def save_cache(self, include_all: bool = False):
        """Save assessment calculations as .snp files so you don't have to re-calculate or have the images available when performing the assessment again.

        Args:
            include_all: When True, an .snp file will be outputted for each image. When False, .snp files are not created for images that were loaded in as .snp files and no changes were made.
        """
        for image_set in self.image_sets:

            if image_set.group is None and image_set.version is None:
                folder = self.output_dir
            elif image_set.group is None:
                folder = os.path.join(self.output_dir, f"version_{image_set.version}")
            elif image_set.version is None:
                folder = os.path.join(self.output_dir, f"group_{image_set.group}")
            else:
                folder = os.path.join(self.output_dir, f"group_{image_set.group}_version_{image_set.version}")
            
            for key, value in image_set.assessment_sets.items():
                if not include_all and value.from_snp:
                    continue
                path = os.path.join(folder, f"{value.identifier}.snp")
                os.makedirs(folder, exist_ok=True)
                with open(path, "wb") as file:
                    pickle.dump(value, file)

        if include_all or (not self.reference.from_snp_ref):
            path = os.path.join(self.output_dir, f"{self.reference.identifier}.snp_ref")
            with open(path, "wb") as file:
                pickle.dump(self.reference, file)

    def write_to_csv(self):
        """Write assessment metrics to CSV.
        """
        csv_path = os.path.join(self.output_dir, "results.csv")
        if os.path.exists(csv_path):
            raise Exception
        with open(csv_path, mode="w", newline='', encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow(["identifier", "reference", "group", "version", "h_dist", "s_dist", "v_dist", "rgb_dist", "oklab_dist", "palette_compare", "palette_Hdiv", "palette_Ediv", "palette_HEsep", "background", "oog_count", "ig_count", "structure_mean", "structure_std", "contrast_mean", "contrast_std"])

            for image_set in self.image_sets:
                for identifier, v in image_set.assessment_sets.items():
                    version = image_set.version
                    group = image_set.group
                    is_reference = identifier in image_set.reference_ids

                    row = [identifier, is_reference, group, version]

                    cc = v.get_stat("color_comparison")
                    if cc is not None:
                        row.extend([cc.get_val("h"), cc.get_val("s"), cc.get_val("v"), cc.get_val("rgb"), cc.get_val("oklab")])
                    else:
                        row.extend([None] * 5)

                    pc = v.get_stat("palette_comparison")
                    if pc is not None:
                        row.extend([pc.get_val("compare"), pc.get_val("Hdiv"), pc.get_val("Ediv"), pc.get_val("HEsep")])
                    else:
                        row.extend([None] * 4)

                    bc = v.get_stat("background_comparison")
                    if bc is not None:
                        row.extend([bc.get_val()])
                    else:
                        row.extend([None])

                    gc = v.get_stat("gamut_comparison")
                    if gc is not None:
                        row.extend([gc.get_val("oog_count"), gc.get_val("ig_count")])
                    else:
                        row.extend([None] * 2)

                    s = v.get_stat("structure")
                    if s is not None:
                        row.extend([s.get_val()[0], s.get_val()[1]])
                    else:
                        row.extend([None] * 2)

                    ct = v.get_stat("contrast_change")
                    if ct is not None:
                        row.extend([ct.get_val()[0], ct.get_val()[1]])
                    else:
                        row.extend([None] * 2)

                    writer.writerow(row)

    def color_distance_compute(self):
        ref_color_samples = []
        
        for image_set in self.image_sets:
            for key, value in image_set.assessment_sets.items():
                if value.get_stat("color_sample") is None:
                    img = image_set.set_images[key]
                    value.set_stat("color_sample", ColorSample(img))
                if key in image_set.reference_ids:
                    ref_color_samples.append(value.get_stat("color_sample"))

        if self.reference.get_stat("color_distribution") is None:
            self.reference.set_stat("color_distribution", ColorDistribution(ref_color_samples, identifier=self.reference.identifier))
        
        for image_set in self.image_sets:
            for key, value in image_set.assessment_sets.items():
                if value.get_stat("color_comparison") is None:
                    value.set_stat("color_comparison", ColorComparison(self.reference.get_stat("color_distribution"), value.get_stat("color_sample")))

    def palette_compute(self):
        ref_palettes = []
        
        for image_set in self.image_sets:
            for key, value in image_set.assessment_sets.items():
                if value.get_stat("palette") is None:
                    img = image_set.set_images[key]
                    value.set_stat("palette", Palette(img))
                    value.set_stat("palette_evaluation", PaletteEvaluation(value.get_stat("palette")))
                if key in image_set.reference_ids:
                    ref_palettes.append(value.get_stat("palette"))

        if self.reference.get_stat("palette") is None:
            self.reference.set_stat("palette", ReferencePalette(ref_palettes, identifier=self.reference.identifier))
            self.reference.set_stat("palette_evaluation", PaletteEvaluation(self.reference.get_stat("palette")))
        
        for image_set in self.image_sets:
            for key, value in image_set.assessment_sets.items():
                if value.get_stat("palette_comparison") is None:
                    value.set_stat("palette_comparison", PaletteComparison(self.reference.get_stat("palette_evaluation"), value.get_stat("palette_evaluation")))
                
    def background_compute(self):
        ref_backgrounds = []

        for image_set in self.image_sets:
            for key, value in image_set.assessment_sets.items():
                if value.get_stat("background") is None:
                    img = image_set.set_images[key]
                    value.set_stat("background", BackgroundEstimation(img))
                if key in image_set.reference_ids:
                    ref_backgrounds.append(value.get_stat("background"))

        if self.reference.get_stat("background") is None:
            self.reference.set_stat("background", ReferenceBackground(ref_backgrounds, identifier=self.reference.identifier))
        
        for image_set in self.image_sets:
            for key, value in image_set.assessment_sets.items():
                if value.get_stat("background_comparison") is None:
                    value.set_stat("background_comparison", BackgroundComparison(self.reference.get_stat("background"), value.get_stat("background")))

    def gamut_compute(self):
        ref_gamuts = []

        for image_set in self.image_sets:
            for key, value in image_set.assessment_sets.items():
                if value.get_stat("gamut") is None:
                    img = image_set.set_images[key]
                    value.set_stat("gamut", Gamut(img))
                if key in image_set.reference_ids:
                    ref_gamuts.append(value.get_stat("gamut"))

        if self.reference.get_stat("gamut") is None:
            self.reference.set_stat("gamut", ReferenceGamut(ref_gamuts, identifier=self.reference.identifier))
        
        for image_set in self.image_sets:
            for key, value in image_set.assessment_sets.items():
                if value.get_stat("gamut_comparison") is None:
                    value.set_stat("gamut_comparison", GamutComparison(self.reference.get_stat("gamut"), value.get_stat("gamut")))
    
    def structure_compute(self, image_sets_grouped):

        for group, (original_version, other_versions) in image_sets_grouped.items():

            if original_version is None or len(other_versions) == 0:
                continue

            for other_version in other_versions:
                for key, value in other_version.assessment_sets.items():
                    if value.get_stat("structure_preservation") is None:
                        query = other_version.set_images[key]
                        if key in original_version.set_images:
                            reference = original_version.set_images[key]
                            value.set_stat("structure_preservation", StructurePreservation(reference, query))

    def contrast_compute(self, image_sets_grouped):

        for group, (original_version, other_versions) in image_sets_grouped.items():

            if original_version is None or len(other_versions) == 0:
                continue

            for other_version in other_versions:
                for key, value in other_version.assessment_sets.items():
                    if value.get_stat("contrast") is None:
                        query = other_version.set_images[key]
                        value.set_stat("contrast", Contrast(query))
                    if key in original_version.set_images:
                        ref_as = original_version.assessment_sets[key]
                        if ref_as.get_stat("contrast") is None:
                            reference = original_version.set_images[key]
                            ref_as.set_stat("contrast", Contrast(reference))
                        value.set_stat("contrast_change", ContrastChange(ref_as.get_stat("contrast"), value.get_stat("contrast")))

    def overall_compute(self):
        pass

    def create_traces(self, stat_type, plot_type):

        traces = {}
        for image_set in self.image_sets:
            for identifier, assessment_set in image_set.assessment_sets.items():
                version = image_set.version
                group = image_set.group
                stat = assessment_set.get_stat(stat_type)
                if stat is None:
                    continue
                value = stat.get_val(plot_type)
                entry = (identifier, group, value)
                if version not in traces:
                    traces[version] = [entry]
                else:
                    traces[version].append(entry)

        return traces

    def color_distance(self, plot_type: str = "h", threshold: Union[float, None] = None, sort: bool = True, sort_by: Union[str, None] = None, clip_ids: bool = False, hide_ids: bool = False):
        """Create color distance plot.

        Args:
            plot_type: Color distance plot type (h, s, v, rgb, oklab)
        """

        if "color_dist" not in self.computed:
            raise Exception

        plot_name = "color_dist_" + plot_type
        if plot_name not in self.plots:
            traces = self.create_traces("color_comparison", plot_type)
            scatter = ScatterPlot(traces, yaxis_title=plot_labels[plot_name][0], title=plot_labels[plot_name][1])
            scatter.make_plot(threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)
            self.plots[plot_name] = scatter

        return self.plots[plot_name].get_fig()

    def color_hist(self, ids: Union[str, List[str]], plot_type: str = "h", groups: Optional[Union[List[str], str]] = None, versions: Optional[Union[List[str], str]] = None, reference: bool = False, density: bool = True, color: Optional[Union[List[str], str]] = None):
        """Creates a color histogram for a single slide or multiple slides.
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
            cs = self.get_assessment_set(group=groups[i], version=versions[i], identifier=img_id).get_stat("color_sample")
            cs_cd.append(cs)

        if reference:
            cd = self.reference.get_stat("color_distribution")
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
                return color_hist_plot.uv_hist(color=color, density=density)
            else:
                raise Exception

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
                return color_hist_plot.uv_hist(color=color, density=density)
            else:
                raise Exception

    def palette(self, plot_type: str = "compare", threshold: Union[float, None] = None, sort: bool = True, sort_by: Union[str, None] = None, clip_ids: bool = False, hide_ids: bool = False):
        """Create palette plot.

        Args:
            plot_type: Palette plot type (compare, Hdiv, Ediv, or HEsep)
        """

        if "palette" not in self.computed:
            raise Exception

        plot_name = "palette_" + plot_type
        if plot_name not in self.plots:
            traces = self.create_traces("palette_comparison", plot_type)
            scatter = ScatterPlot(traces, yaxis_title=plot_labels[plot_name][0], title=plot_labels[plot_name][1])
            scatter.make_plot(threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)
            self.plots[plot_name] = scatter

        return self.plots[plot_name].get_fig()


    def display_palette(self, ids, groups=None, versions=None, reference=False):
        
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
            p = self.get_assessment_set(group=groups[i], version=versions[i], identifier=img_id).get_stat("palette")
            palettes.append(p)

        if reference:
            p = self.reference.get_stat("palette")
            palettes.append(p)

        for p in palettes:
            print(p.identifier)
            p.display()

    def gamut(self, plot_type: str = "oog_count", threshold: Union[float, None] = None, sort: bool = True, sort_by: Union[str, None] = None, clip_ids: bool = False, hide_ids: bool = False):
        """Create gamut plot.

        Args:
            plot_type: Gamut plot type (oog_count, ig_count)
        """

        if "gamut" not in self.computed:
            raise Exception

        plot_name = "gamut_" + plot_type
        if plot_name not in self.plots:
            traces = self.create_traces("gamut_comparison", plot_type)
            scatter = ScatterPlot(traces, yaxis_title=plot_labels[plot_name][0], title=plot_labels[plot_name][1])
            scatter.make_plot(threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)
            self.plots[plot_name] = scatter

        return self.plots[plot_name].get_fig()

    def background(self, threshold: Union[float, None] = None, sort: bool = True, sort_by: Union[str, None] = None, clip_ids: bool = False, hide_ids: bool = False):
        """Create background plot.
        """

        if "background" not in self.computed:
            raise Exception

        plot_name = "background"
        if plot_name not in self.plots:
            traces = self.create_traces("background_comparison", "dist")
            scatter = ScatterPlot(traces, yaxis_title=plot_labels[plot_name][0], title=plot_labels[plot_name][1])
            scatter.make_plot(threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)
            self.plots[plot_name] = scatter

        return self.plots[plot_name].get_fig()

    def contrast(self, threshold: Union[float, None] = None, sort: bool = True, sort_by: Union[str, None] = None, clip_ids: bool = False, hide_ids: bool = False):
        """Create contrast plot.
        """

        if "contrast" not in self.computed:
            raise Exception

        plot_name = "contrast"
        if plot_name not in self.plots:
            traces = self.create_traces("contrast_change", "dist")
            scatter = ScatterPlot(traces, yaxis_title=plot_labels[plot_name][0], title=plot_labels[plot_name][1])
            scatter.make_plot(threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)
            self.plots[plot_name] = scatter

        return self.plots[plot_name].get_fig()

    def structure(self, threshold: Union[float, None] = None, sort: bool = True, sort_by: Union[str, None] = None, clip_ids: bool = False, hide_ids: bool = False):
        """Create structure plot.
        """

        if "structure" not in self.computed:
            raise Exception

        plot_name = "structure"
        if plot_name not in self.plots:
            traces = self.create_traces("structure", "dist")
            scatter = ScatterPlot(traces, yaxis_title=plot_labels[plot_name][0], title=plot_labels[plot_name][1])
            scatter.make_plot(threshold=threshold, sort=sort, sort_by=sort_by, clip_ids=clip_ids, hide_ids=hide_ids)
            self.plots[plot_name] = scatter

        return self.plots[plot_name].get_fig()

    def overall(self):
        pass

    def original_version_check(self):

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

    
    def compute(self, criteria: Union[List[str], str] = ["color_dist", "palette", "gamut", "background", "structure", "contrast", "overall"]):
        """Compute the assessment.

        Args:
            criteria: Criterion/critiera to compute.
        """

        if isinstance(criteria, str):
            criteria = [criteria]

        if self.reference is None:
            self.reference = ReferenceSet("Reference")

        if not self.reference_is_set:
            for image_set in self.image_sets:
                image_set.set_all_as_reference()

        # image_sets_grouped = self.original_version_check()

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
            self.structure_compute(image_sets_grouped)
            self.computed.add("structure")

        if "contrast" in criteria:
            self.contrast_compute(image_sets_grouped)
            self.computed.add("contrast")

        if "overall" in criteria:
            pass
                