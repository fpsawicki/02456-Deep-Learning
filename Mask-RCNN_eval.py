from metrics.binary_metrics import BinaryImageMetrics
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.image as image


path_testA = "/home/karl/DEEPlearning/testa"
path_testB = "/home/karl/DEEPlearning/testb"
path_mrcnn = "/home/karl/DEEPlearning/mrcnn_output"
path_mrcnn2 = "/home/karl/DEEPlearning/mrcnn_output2"

files_mrcnn = os.listdir(path_mrcnn)
files_mrcnn = [file for file in files_mrcnn if file.count("_") < 2]
files_A = sorted([file[:-4] for file in files_mrcnn if "A" in file])
files_B = sorted([file[:-4] for file in files_mrcnn if "B" in file])


##_anno.bmp

def get_acc(files, path_true, path_proposed):
    acc = [0,0,0,0]
    length = len(files)
    print(length)
    c= 0
    for file in files:
        true_file = os.path.join(path_true, file+ "_anno.bmp")
        proposed_file = os.path.join(path_proposed, file+ ".bmp")
        true_img = np.asarray(Image.open(true_file)).astype("int64")
        propsoed_img = np.asarray(Image.open(proposed_file)).astype("int64")

        true_img[true_img > 0.5] = 1
        true_img[true_img <= 0.5] = 0

        propsoed_img[propsoed_img > 0.5] = 1
        propsoed_img[propsoed_img <= 0.5] = 0

        #image.imsave('name.png', propsoed_img)

        metrics = BinaryImageMetrics(true_img, propsoed_img)
        acc[0] += metrics.get_count()
        acc[1] += metrics.get_f1()
        acc[2] += metrics.get_f1_obj()
        acc[3] += metrics.get_hausdorff_obj_distance()
        c+=1


    acc = [val / length for val in acc]
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(true_img)
    ax2.imshow(propsoed_img)
    fig.show()

    return acc

print("testb: ", get_acc(files_B, path_testB, path_mrcnn2))
print("testa: ", get_acc(files_A, path_testA, path_mrcnn2))
