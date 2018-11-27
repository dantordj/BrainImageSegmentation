from utils import load_model, load_conflict_voxels, save_predictions, load_naive_coeffs, dice
from load_data import load_data
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from config import path_train, model, patch, model, path_classifiers, path_test, path_predictions
import nibabel as nib
import nilearn


def prior_distribution(atlases, images, labels, conflict_voxels):

    num_conflict_voxels = len(conflict_voxels)
    num_atlasses = len(atlases)
    distrib = np.count_nonzero(labels, axis=1) / len(atlases)
    return distrib

def compute_coefficients(atlases, images, conflict_voxels, model="logReg"):
    coeffs = np.zeros((len(conflict_voxels), len(images), len(atlases)))
    if model == "naive":
        coeffs = load_naive_coeffs()
        return coeffs
    if patch == 1:
        train_features = np.array([atlas.features(conflict_voxels) for atlas in atlases.values()])
        test_features = np.array([image.features(conflict_voxels) for image in images.values()])
    else:
        train_features = np.array([atlas.image_vector()[conflict_voxels] for atlas in atlases.values()])
        test_features = np.array([image.image_vector()[conflict_voxels] for image in images.values()])

    size_window = 1
    num_images = len(images)
    labels_test = [image.label_vector()[conflict_voxels] for image in images.values()]
    if patch == 1:
        size_window = 27

    for i, atlas in enumerate(atlases.values()):

        index_atlas = atlas.id
        file = "%s/classifier_%s"%(index_atlas, patch)
        print("atlas %s" %index_atlas)
        classifiers = load_model(file)

        for index_voxel, voxel in enumerate(conflict_voxels):

            test_x = test_features[:, index_voxel] - train_features[i, index_voxel]
            test_x = test_x.reshape(num_images, size_window)
            val_test = [labels_test[i][index_voxel] == labels_test[j][index_voxel] for j in range(num_images)]

            val_test = np.array(val_test, dtype=int)
            c = classifiers[index_voxel].predict_proba(test_x)[:,1]

            coeffs[index_voxel, :, i] = c
            if index_voxel % 20000 == 0:
                print(index_voxel)

    return coeffs

def predict(model=model):
    naive = (model == "naive")
    mv = (model == "mv")

    images = load_data(path_test, labels=True)

    atlases = load_data(path_train, labels=True)
    atlases_index = {index: i for i,index in enumerate(atlases.keys())}
    num_atlasses = len(atlases)
    conflict_voxels = load_conflict_voxels()
    num_conflict_voxels = len(conflict_voxels)
    train_labels = np.zeros((num_conflict_voxels, num_atlasses), dtype=int)
    for index, atlas in atlases.items():
        train_labels[:,atlases_index[index]] = atlas.label_vector()[conflict_voxels]
    if not mv and not naive:
        coeffs = compute_coefficients(atlases, images, conflict_voxels, model)
    if naive:
        coeffs = load_naive_coeffs()

    distrib = prior_distribution(atlases, images, train_labels, conflict_voxels)

    pred_labels = np.zeros((len(images), num_conflict_voxels))
    for index_image, image in enumerate(images):
        print(index_image)
        for index_vox, vox in enumerate(conflict_voxels):
            prior = distrib[index_vox]
            prob1 = prior
            prob0 = (1 - prior)
            if mv:
                pred_labels[index_image, index_vox] = int((prob1 >= prob0))
                continue

            for index_atlas in range(num_atlasses):
                Fi = train_labels[index_vox, index_atlas]
                if naive:
                    C = coeffs[index_vox, index_atlas]
                else:
                    C = coeffs[index_vox, index_image, index_atlas]
                if Fi == 1:
                    prob1 *= C
                    prob0 *= (1 - C)
                else:
                    prob1 *= (1 - C)
                    prob0 *= C

                if prob1 >= prob0:
                    pred_labels[index_image, index_vox] = 1


    ref_image = list(atlases.values())[0].image
    ref_labels = list(atlases.values())[0].label_vector()
    shape_mni = ref_image.shape
    dices = []
    for index_image, image in enumerate(images.values()):
        labels = ref_labels.copy()
        labels[conflict_voxels] = pred_labels[index_image, :]
        dices += [dice(labels, image.label_vector())]
        labels = labels.reshape(shape_mni)
        pred_image = nilearn.image.new_img_like(ref_image, labels)
        nib.save(pred_image, path_predictions + str(image.id) + ".nii")
    print(dices)
    print(np.mean(dices))

predict()
