import os
from utils import get_conflict_voxels, save_model, create_directories
from load_data import load_data
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


path_train = "mni/"
def train_classifiers(save=False, all_images=True):
    atlases = load_data(path_train, all_images)
    create_directories(atlases)
    conflict_voxels = get_conflict_voxels(atlases, save)
    mni_size = 7109137
    num_atlasses = len(atlases)
    num_conflict_voxels = len(conflict_voxels)
    # reindex the conflictual voxels from 0 to num_conflictual_voxels
    labels = np.zeros((num_conflict_voxels, num_atlasses), dtype=int)
    print((num_conflict_voxels, num_atlasses))

    #classifiers = [[None for i in range(num_atlasses)] for i in range(num_conflict_voxels)]

    # reindex the atlasses from 0 to nul_atlasses
    atlases_index = {index: i for i,index in enumerate(atlases.keys())}

    for index, atlas in atlases.items():
      labels[:,atlases_index[index]] = atlas.label_vector()[conflict_voxels]

    features = np.array([atlas.features(conflict_voxels) for atlas in atlases.values()])
    print("Voxels")
    print(features.shape)
    count = 0

    for index, atlas in atlases.items():
        if index == 1009:
            print("1009 processed")
            count += 1
            continue
        print("processing index %s" %index)
        index_atlas = atlases_index[index]

        classifiers = []
        for i in range(len(conflict_voxels)):

            train_x = np.array(features[:,i] - features[index_atlas, i])
            #print(train_x)
            train_y = [int(labels[i,j] == labels[i,index_atlas]) for j in range(num_atlasses)]
            # passer à des pathchs
            clf = LogisticRegression().fit(train_x, train_y)
            classifiers += [clf]

            if i % 10000 == 0:
                print(i)
        count += 1
        if save:
            save_model(classifiers, "%s/classifier"%(index))
        print("Processed %s / %s" %(count,num_atlasses))

    return classifiers

classifier = train_classifiers(save=True, all_images=True)
