from os.path import exists
import numpy as np
import cv2 as cv
import torchvision.transforms.v2.functional as F
from scipy import ndimage 
import random

from torch.utils.data import Dataset

class DatasetValue(Dataset):
    def __init__(self, data):
        self.X = data["arr_0"]
        print(self.X[0], self.X[0].shape)
        self.Y = data["arr_1"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


def normalize(image):
    image = cv.resize(image, (50, 50))
    normalized = (image - image.mean())
    normalized = normalized / (image.std() + 1e-8)
    return normalized
    # return image


def create_npy_file():
    file_data = []
    file_types = []
    test_data = []
    test_types = []
    for i in range(1, 6):
        for j in range(1, 1000+1):
            file_name = "images/%d/opencv_frame_%d.png" % (i, j);
            #print(file_name)
            #if exists(file_name):
            data = cv.imread(file_name, cv.IMREAD_GRAYSCALE)
            data = np.asarray(data)
            inverted = np.invert(data)

            data = normalize(data)
            inverted = normalize(inverted)

            rot_direction = (random.random() % 2) * (-1)

            augmented_versions = []
            augmented_versions.append(data)
            augmented_versions.append(np.fliplr(data))
            augmented_versions.append(ndimage.rotate(data, rot_direction * (random.random() % 10) + 5 * rot_direction, reshape=False))
            augmented_versions.append(np.fliplr(ndimage.rotate(data, rot_direction * (random.random() % 10) + 5 * rot_direction, reshape=False)))
            augmented_versions.append(inverted)
            augmented_versions.append(np.fliplr(inverted))

            for augmented in augmented_versions:
                if j <= 900+1:
                    file_data.append(augmented)
                    file_types.append(np.eye(5)[i-1])
                else:
                    test_data.append(augmented)
                    test_types.append(np.eye(5)[i-1])

    print(len(file_data), len(file_types), len(test_types), len(test_data))
    np.savez("images/data.npz", file_data, file_types)
    np.savez("images/test.npz", test_data, test_types)


if __name__ == "__main__":
    create_npy_file()



        

