import numpy as np
import os
import pickle
from config import path_classifiers, path_predictions


def get_conflict_voxels(atlases, all_voxels=True, save=True):
  labels = np.array([atlas.label_vector() for atlas in atlases.values()], dtype=int)
  print(labels.shape)
  conflict_voxels = []
  for i in range(labels.shape[1]):
    labels_i = labels[:,i]
    if len(np.unique(labels_i)) != 1:

      conflict_voxels.append(i)
    if i%1000000 == 0:
        print(i)
        if not all_voxels:
            if len(conflict_voxels) > 10000:
                return conflict_voxels


  del labels
  with open("conflict_voxels", 'wb') as file:
        pickle.dump(conflict_voxels, file)
  return conflict_voxels


def load_conflict_voxels():
    with open("conflict_voxels", 'rb') as file:
        conflict_voxels = pickle.load(file)
    return conflict_voxels


def save_model(classifiers, file):
    # Save to file in the current working directory
    pkl_filename = path_classifiers + file
    print("saving", pkl_filename)
    with open(pkl_filename, 'wb') as file:

        pickle.dump(classifiers, file)



def save_predictions(predictions):
    pkl_filename = path_predictions + "predictions"
    print("saving", pkl_filename)
    with open(pkl_filename, 'wb') as file:

        pickle.dump(predictions, file)

def load_model(file):
    pkl_filename = path_classifiers + file
    classifiers = []
    with open(pkl_filename, 'rb') as file:
        classifiers = pickle.load(file)
    return classifiers

def save_naive_coeffs(coeffs):
    pkl_filename = path_classifiers + "naive_coeffs"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(coeffs, file)


def load_naive_coeffs():
    pkl_filename = path_classifiers + "naive_coeffs"
    with open(pkl_filename, 'rb') as file:
        coeffs = pickle.load(file)
    return coeffs


def dice(img_1, img_2):

    n = np.sum(img_1 + img_2 == 2)
    n1 = np.sum(img_1)
    n2 = np.sum(img_2)
    return 2 * n / (n1 + n2)



def create_directories(atlases):
    for index in atlases.keys():
        directory = path_classifiers + str(index) + "/"
        if not os.path.exists(directory):
            print("Create directory %s" % directory)
            os.makedirs(directory)
