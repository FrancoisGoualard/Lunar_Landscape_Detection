import os
from content.data_augmentation import data_augmentation
from config import DATAPATH
import tqdm
import shutil


def create_dataset():
    path_data = os.path.join(DATAPATH, "images")
    path_data_render = os.path.join(path_data, "render")
    path_data_ground = os.path.join(path_data, "ground")
    path_dataset = os.path.join(path_data, "dataset")
    path_dataset_render = os.path.join(path_dataset, "render")
    path_dataset_ground = os.path.join(path_dataset, "ground")
    path_data_augmentation = os.path.join(path_data, "data_augmentation")
    path_data_augmentation_render = os.path.join(path_data_augmentation, "render")
    path_data_augmentation_ground = os.path.join(path_data_augmentation, "ground")

    #  check that the dataset is not created or does not correspond
    nb_data = len(os.listdir(path_data_render))
    if os.path.isdir(path_dataset):
        if os.path.isdir(path_dataset_render) & os.path.isdir(path_dataset_ground):
            if len(os.listdir(path_dataset_render)) == 4 * nb_data &\
                    len(os.listdir(path_dataset_ground)) == 4 * nb_data:
                return 0

    # else we run the function
    data_augmentation()
    shutil.rmtree(path_dataset, ignore_errors=True)
    os.mkdir(path_dataset)
    os.mkdir(path_dataset_render)
    os.mkdir(path_dataset_ground)

    print("creating dataset ...")
    for f in tqdm.tqdm(os.listdir(path_data_render)):
        path_origin = os.path.join(path_data_render, f)
        path_target = os.path.join(path_dataset_render, f)
        shutil.copyfile(path_origin, path_target)

    for f in tqdm.tqdm(os.listdir(path_data_ground)):
        path_origin = os.path.join(path_data_ground, f)
        path_target = os.path.join(path_dataset_ground, f)
        shutil.copyfile(path_origin, path_target)

    for f in tqdm.tqdm(os.listdir(path_data_augmentation_ground)):
        path_origin = os.path.join(path_data_augmentation_ground, f)
        path_target = os.path.join(path_dataset_ground, f)
        os.rename(path_origin, path_target)

    for f in tqdm.tqdm(os.listdir(path_data_augmentation_render)):
        path_origin = os.path.join(path_data_augmentation_render, f)
        path_target = os.path.join(path_dataset_render, f)
        os.rename(path_origin, path_target)
    else:
        print("Dataset already created")


if __name__ == "__main__":
    create_dataset()










