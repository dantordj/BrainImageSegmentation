import os
from utils import get_conflict_voxels, save_model, create_directories, load_conflict_voxels, save_naive_coeffs
from load_data import load_data
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import dice
from config import path_train, model, patch, model
import pickle

load_voxels = True

def train_naive_classifiers(save=False, all_images=True):
    atlases = load_data(path_train, all_images)
    num_atlasses = len(atlases)
    if load_voxels:
        conflict_voxels = load_conflict_voxels()
    else:
        conflict_voxels = get_conflict_voxels(atlases)
    num_conflict_voxels = len(conflict_voxels)
    print("num conflict voxel %s"%num_conflict_voxels)
    labels = np.zeros((num_conflict_voxels, num_atlasses), dtype=int)
    # reindex the atlasses from 0 to nul_atlasses
    atlases_index = {index: i for i,index in enumerate(atlases.keys())}
    coeffs = np.zeros((num_conflict_voxels, num_atlasses))
    for index, atlas in atlases.items():
        labels[:,atlases_index[index]] = atlas.label_vector()[conflict_voxels]

    for index, atlas in atlases.items():
        for index_vox, vox in enumerate(conflict_voxels):
            cij = np.sum(labels[index_vox, :] == labels[index_vox, atlases_index[index]])
            cij = cij * 1. / num_atlasses
            coeffs[index_vox, atlases_index[index]] = cij
    if save:
        save_naive_coeffs(coeffs)




def train_log_reg_classifiers(save=False, all_images=True, model="logReg", patch=1):
    atlases = load_data(path_train, all_images)
    naive = (model == "naive")
    num_atlasses = len(atlases)
    print("Number of atlases %s" %num_atlasses)
    if load_voxels:
        conflict_voxels = load_conflict_voxels()
    else:
        conflict_voxels = get_conflict_voxels(atlases)
    num_conflict_voxels = len(conflict_voxels)
    print("num conflict voxel %s"%num_conflict_voxels)
    # reindex the conflictual voxels from 0 to num_conflictual_voxels
    labels = np.zeros((num_conflict_voxels, num_atlasses), dtype=int)
    # reindex the atlasses from 0 to nul_atlasses
    atlases_index = {index: i for i,index in enumerate(atlases.keys())}

    for index, atlas in atlases.items():
      labels[:,atlases_index[index]] = atlas.label_vector()[conflict_voxels]

    if patch == 1:
        features = np.array([atlas.features(conflict_voxels) for atlas in atlases.values()])
    else:
        features = np.array([atlas.image_vector()[conflict_voxels] for atlas in atlases.values()])

    count = 0

    for index, atlas in atlases.items():

        print("processing index %s" %index)
        index_atlas = atlases_index[index]

        classifiers = []
        for i in range(len(conflict_voxels)):

            if patch == 1:
                train_x = np.array(features[:,i] - features[index_atlas, i])
            else:
                train_x = np.array(features[:,i] - features[index_atlas, i])
                train_x = train_x.reshape(len(atlases), 1)

            train_y = [int(labels[i,j] == labels[i,index_atlas]) for j in range(num_atlasses)]

            clf = LogisticRegression().fit(train_x, train_y)
            classifiers += [clf]

            if i % 10000 == 0:
                print(i)
        count += 1
        if save:
            save_model(classifiers, "%s/classifier_%s"%(index, patch))
        print("Processed %s / %s" %(count,num_atlasses))

    return classifiers


def train_classifiers(save=False, all_images=True, model="logReg", patch=1):
    if model == "naive":
        train_naive_classifiers(save=save, all_images=all_images)
    else:
        train_log_reg_classifiers(save=save, all_images=all_images, model=model, patch=patch)



classifier = train_classifiers(save=True, all_images=True, model=model, patch=patch)
