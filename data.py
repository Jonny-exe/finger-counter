from os.path import exists
import numpy as np
import cv2

def create_npy_file():
    file_data = []
    file_types = []
    test_data = []
    test_types = []
    for i in range(1, 6):
        for j in range(1, 400):
            file_name = "images/%d/opencv_frame_%d.png" % (i, j);
            #print(file_name)
            #if exists(file_name):
            data = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            data = np.asarray(data)
            if j < 300:
                file_data.append(data)
                file_types.append(np.eye(5)[i-1])
            #else:
            else:
                test_data.append(data)
                test_types.append(np.eye(5)[i-1])

            j += 1
    print(len(file_data), len(file_types), len(test_types), len(test_data))
    np.savez("images/data.npz", file_data, file_types)
    np.savez("images/test.npz", test_data, test_types)


if __name__ == "__main__":
    create_npy_file()



        

