from config import DATAPATH
import os
import imgaug as ia
import cv2
import tqdm



def data_augmentation():
    path = os.path.join(DATAPATH, "images")
    path_render = os.path.join(path, "render")
    path_ground = os.path.join(path, "ground")
    path_data_augmentation = os.path.join(path, "data_augmentation")
    path_data_augmentation_render = os.path.join(path_data_augmentation, "render")
    path_data_augmentation_ground = os.path.join(path_data_augmentation, "ground")

    files_render = [f for f in os.listdir(path_render) if os.path.isfile(os.path.join(path_render,f))]
    files_ground = [f for f in os.listdir(path_ground) if os.path.isfile(os.path.join(path_ground,f))]

    if not os.path.isdir(path_data_augmentation)
        os.mkdir(path_data_augmentation)
        os.mkdir(path_data_augmentation_ground)
        os.mkdir(path_data_augmentation_render)

        print("creation of new data ...")
        for f in tqdm.tqdm(files_render):
            image = cv2.imread(os.path.join(path_render, f))

            # rotation de 3 degrés dans le sens horaire et anti-horaire
            rotate3 = ia.augmenters.Affine(rotate=3)
            rotated_image = rotate3.augment_image(image)
            cv2.imwrite(os.path.join(path_data_augmentation_ground, f[:-4] + "_rot.png"), rotated_image)

            rotateinv = ia.augmenters.Affine(rotate=-3)
            rotated_image_inv = rotateinv.augment_image(image)
            cv2.imwrite(os.path.join(path_data_augmentation_ground, f[:-4] + "_rotinv.png"), rotated_image_inv)

            # flip horizontal
            flip_hr = ia.augmenters.Fliplr(p=1.0)
            flip_hr_image = flip_hr.augment_image(image)
            cv2.imwrite(os.path.join(path_data_augmentation_ground, f[:-4] + "_flip.png"), flip_hr_image)

        for f in tqdm.tqdm(files_ground):
            image = cv2.imread(os.path.join(path_ground,f))

            # rotation de 3 degrés dans le sens horaire et anti-horaire
            rotate3 = ia.augmenters.Affine(rotate= 3)
            rotated_image = rotate3.augment_image(image)
            cv2.imwrite(os.path.join(path_data_augmentation_ground, f[:-4] + "_rot.png"), rotated_image)

            rotateinv = ia.augmenters.Affine(rotate=-3)
            rotated_image_inv = rotateinv.augment_image(image)
            cv2.imwrite(os.path.join(path_data_augmentation_ground, f[:-4] + "_rotinv.png"), rotated_image_inv)

            # flip horizontal
            flip_hr = ia.augmenters.Fliplr(p=1.0)
            flip_hr_image = flip_hr.augment_image(image)
            cv2.imwrite(os.path.join(path_data_augmentation_ground, f[-4] + "_flip.png"), flip_hr_image)

    else:
        print("data augmentation already done")


if __name__ == "__main__":
    data_augmentation()










