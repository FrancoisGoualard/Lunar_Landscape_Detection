import os
from data_augmentation import data_augmentation
from config import DATAPATH
import tqdm
import shutil

def create_dataset():

    data_augmentation()

    path_data = os.path.join(DATAPATH, "images")
    path_data_render = os.path.join(path_data, "render")
    path_data_ground = os.path.join(path_data, "ground")
    path_dataset = os.path.join(path_data, "dataset")
    path_dataset_render = os.path.join(path_dataset, "render")
    path_dataset_ground = os.path.join(path_dataset, "ground")
    path_data_augmentation = os.path.join(path_data, "data_augmentation")
    path_data_augmentation_render = os.path.join(path_data_augmentation, "render")
    path_data_augmentation_ground = os.path.join(path_data_augmentation, "ground")

    if not os.path.dir(path_dataset):
        os.mkdir(path_dataset)
        os.mkdir(path_dataset_render)
        os.mkdir(path_dataset_ground)

        print("creating dataset ...")
        for f in tqdm.tqdm([f for f in os.listdir(path_data_render) if os.path.isfile(os.path.join(path_data_render, f))]):
            path_origin = os.path.join(path_data_render, f)
            path_target = os.path.join(path_dataset_render, f)
            shutil.copyfile(path_origin, path_target)

        for f in tqdm.tqdm([f for f in os.listdir(path_data_ground) if os.path.isfile(os.path.join(path_data_ground, f))]):
            path_origin = os.path.join(path_data_ground, f)
            path_target = os.path.join(path_dataset_ground, f)
            shutil.copyfile(path_origin, path_target)

        for f in tqdm.tqdm([f for f in os.listdir(path_data_augmentation_ground) if os.path.isfile(os.path.join(path_data_augmentation_ground, f))]):
            path_origin = os.path.join(path_data_augmentation_ground, f)
            path_target = os.path.join(path_dataset_ground, f)
            os.rename(path_origin, path_target)

        for f in tqdm.tqdm([f for f in os.listdir(path_data_augmentation_render) if os.path.isfile(os.path.join(path_data_augmentation_render, f))]):
            path_origin = os.path.join(path_data_augmentation_render, f)
            path_target = os.path.join(path_dataset_render, f)
            os.rename(path_origin, path_target)
    else:
        print("dataset already created")


if __name__ == "__main__":
    create_dataset()










