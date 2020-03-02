from config import DATAPATH
import os
import imgaug as ia
import cv2
import tqdm
import shutil


def data_augmentation_directory(name_directory):
    path = os.path.join(DATAPATH, "images")
    path_render = os.path.join(path, name_directory)
    path_data_augmentation = os.path.join(path, "data_augmentation")
    path_data_augmentation_render = os.path.join(path_data_augmentation, name_directory)

    if os.path.isdir(path_data_augmentation_render):
        if len(os.listdir(path_data_augmentation_render)) == len(os.listdir(path_render)) * 3:
            print(f"Data augmentation already done on {name_directory}")
        return

    if not os.path.isdir(path_data_augmentation):
        os.mkdir(path_data_augmentation)

    shutil.rmtree(path_data_augmentation_render, ignore_errors=True)
    os.mkdir(path_data_augmentation_render)

    print("creation of new data ...")
    for f in tqdm.tqdm(os.listdir(path_render)):
        image = cv2.imread(os.path.join(path_render, f))

        # rotation de 3 degrés dans le sens horaire et anti-horaire
        rotate3 = ia.augmenters.Affine(rotate=3)
        rotated_image = rotate3.augment_image(image)
        cv2.imwrite(os.path.join(path_data_augmentation_render, f[:-4] + "_rot.png"), rotated_image)

        rotateinv = ia.augmenters.Affine(rotate=-3)
        rotated_image_inv = rotateinv.augment_image(image)
        cv2.imwrite(os.path.join(path_data_augmentation_render, f[:-4] + "_rotinv.png"), rotated_image_inv)

        # flip horizontal
        flip_hr = ia.augmenters.Fliplr(p=1.0)
        flip_hr_image = flip_hr.augment_image(image)
        cv2.imwrite(os.path.join(path_data_augmentation_render, f[:-4] + "_flip.png"), flip_hr_image)

    if len(os.listdir(path_data_augmentation_render)) != len(os.listdir(path_render) * 3):
        shutil.rmtree(path_data_augmentation_render)
        raise AssertionError(f"{name_directory} data augmentation failed")


def data_augmentation():
    data_augmentation_directory("render")
    data_augmentation_directory("ground")
