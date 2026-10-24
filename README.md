# StainNormProgram

**StainNormProgram** is a program that performs stain normalization, assessment, and correction.

## Background

Stain normalization transforms the color distribution of stained tissue images. The process of staining biological tissues with dyes reveals important structures that can be used by clinicians to diagnose patients. However, variability between different hospitals and laboratories creates noise in the appearance of these stains, which may fool machine learning algorithms. Stain normalization aims to reduce such variabilities by converting *source* images with different staining styles into a single staining style, which is computed from a *target* reference. 

StainNormProgram provides the utilities needed to perform stain normalization on whole slide images, tissue microarrays, and tiles. The program supports hematoxylin and eosin (H&E) and immunohistochemistry (IHC) with diaminobenzidine (DAB) chromogen. StainNormProgram provides also provides features for assessing and correcting processed images.

## Getting Started

```
pip install stainnormprogram
```

To learn how to use StainNormProgram, check out the `tutorials/` folder.

## Citation

```
@software{stainnormprogram,
    author = {Hsu, Peter and Falahkheirkhah, Kianoush and Bhargava, Rohit},
    title = {StainNormProgram},
    year = {2026},
    publisher = {GitHub},
    version = {1.0.0},
    url = {https://github.com/peterhsu-uiuc/StainNormProgram"}
}
```

Built at the University of Illinois Urbana-Champaign. Support for this project was provided by the National Science Foundation Graduate Research Fellowship Program.
