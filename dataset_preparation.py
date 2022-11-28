import shutil
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
import os
from PIL import Image


def dataset_2_np(dataset_dir='dataset'):
    """
        This function takes the directory of the dataset as input and returns the X and Y numpy arrays.
        The X array contains the images and the Y array contains the labels.
        The images are resized to 96x96.
        The labels are one-hot encoded.
    """
    IMG_SIZE = 96

    dataset = ImageDataGenerator()

    dataset = dataset.flow_from_directory(directory=dataset_dir,
                                          target_size=(IMG_SIZE, IMG_SIZE),
                                          color_mode='rgb',
                                          classes=None,  # can be set to labels
                                          class_mode='categorical',
                                          batch_size=1,
                                          seed=1)

    X = []
    Y = []
    for i in range(dataset.__len__()):
        x, y = dataset.__getitem__(i)
        X.append(x)
        Y.append(y)

    X = np.concatenate(X)
    Y = np.concatenate(Y)

    return X, Y


def extract_dataset(zipped_dataset, output_directory, data_aug=False):
    """
        Extracts the dataset from the zipped file and moves the folders to the root directory.
        If data_aug is True, the function will augment the dataset to have 500 images per class.
        :param zipped_dataset: The path to the zipped dataset.
        :param output_directory: The path to the output directory.
        :param data_aug: Whether to augment the dataset or not.
        :return: None
    """
    os.makedirs(output_directory, exist_ok=True)
    with zipfile.ZipFile(zipped_dataset, 'r') as zip_ref:
        zip_ref.extractall(output_directory)

    path_to_data = output_directory + 'training_data_final'
    for folder in os.listdir(path_to_data):
        shutil.rmtree(os.path.join(output_directory, folder), ignore_errors=True)
        shutil.move(os.path.join(path_to_data, folder), output_directory)
    os.rmdir(path_to_data)

    if data_aug is True:
        for species in range(8):
            dir_path = r'dataset/Species{}'.format(species + 1)
            image_directory = dir_path + '/'
            n_images = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
            print("\n", n_images, end=' ')
            if n_images < 500:
                dataset = []
                datagen = ImageDataGenerator(
                    rotation_range=40,
                    fill_mode='reflect',
                    horizontal_flip=True,
                    vertical_flip=True)
                my_images = os.listdir(image_directory)

                for i, image_name in enumerate(my_images):
                    if (image_name.split('.')[1] == 'jpg'):
                        image = io.imread(image_directory + image_name)
                        image = Image.fromarray(image, 'RGB')
                        dataset.append(np.array(image))
                x = np.array(dataset)

                for batch in datagen.flow(x, batch_size=10,
                                          save_to_dir=dir_path,
                                          save_prefix='dr',
                                          save_format='jpg',
                                          ):

                    if len([entry for entry in os.listdir(dir_path) if
                            os.path.isfile(os.path.join(dir_path, entry))]) >= 500:
                        print('->', len([entry for entry in os.listdir(dir_path) if
                                         os.path.isfile(os.path.join(dir_path, entry))]), end=' ')
                        break


if __name__ == '__main__':
    extract_dataset('training_dataset_homework1.zip', 'dataset/', data_aug=False)
