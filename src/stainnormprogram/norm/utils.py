from stainnormprogram.slide import SlideImage, is_valid_extension

import numpy as np
import os

def nnls(HE, Y): # HE is (3,2) and Y is (3,N)
    """
    Modified form of Lawson and Hanson NNLS
    """

    C = np.linalg.lstsq(HE, Y, rcond=None)[0] # (2,N)

    negIdx = np.where(np.any(C < 0, axis=0))[0]

    Y_neg = Y[:, negIdx]
    C_neg = C[:, negIdx]

    if Y_neg.size == 0:
        return C

    H = HE[:,0:1]
    E = HE[:,1:2]

    Eidx = C_neg[0,:] < 0
    Hidx = ~Eidx

    YE = Y_neg[:,Eidx]
    YH = Y_neg[:,Hidx]

    negNNLS = np.zeros((2, Y_neg.shape[1]))

    if YH.size != 0:
        CYH = np.linalg.lstsq(H, YH, rcond=None)[0]
        negNNLS[0,Hidx] = CYH
    if YE.size != 0:
        CYE = np.linalg.lstsq(E, YE, rcond=None)[0]
        negNNLS[1,Eidx] = CYE

    C[:,negIdx] = negNNLS

    return C

def get_paths_in_folder(folder_path):
    paths = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path) and is_valid_extension(full_path):
            paths.append(full_path)
    return paths

def add_images_to_dict(images, mydict):
    
    if isinstance(images, str):
        images = [images]
    
    if isinstance(images, list):
        for path in images:
            if not isinstance(path, str) or (not os.path.isfile(path) and not os.path.isdir(path)):
                raise Exception(f"{path} is not a valid image or directory path.")
            elif os.path.isfile(path) and is_valid_extension(path):
                image_id = is_valid_extension(path, split=True)[0]
                mydict[image_id] = SlideImage(path, identifier=image_id)
            elif os.path.isdir(path):
                for p in get_paths_in_folder(path):
                    image_id = is_valid_extension(p, split=True)[0]
                    mydict[image_id] = SlideImage(p, identifier=image_id)

def validate_json(json_path):

    if not os.path.exists(json_path):
        raise Exception(f"{json_path} does not exist.")