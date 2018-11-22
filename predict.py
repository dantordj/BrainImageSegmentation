from utils import load_model, load_conflict_voxels, save_predictions
from load_data import load_data
import numpy as np
import pickle

processed = [1038, 1008, 1009]


path_test = "mni/"
path_train = "mni/"
images = load_data(path_test, labels=False)
atlases = load_data(path_train, labels=True)
conflict_voxels = load_conflict_voxels()

processed = [1104, 1116, 1039, 1006, 1038, 1009]
train_features = np.array([atlases[i].features(conflict_voxels) for i in processed])

features = np.array([atlas.features(conflict_voxels) for atlas in images.values()])

preds = np.zeros((len(conflict_voxels), len(images), len(atlases)))
for i,atlas in enumerate(atlases.values()):
    index_atlas = atlas.id
    file = "%s/classifier"%(index_atlas)
    print("atlas %s" %index_atlas)
    classifiers = load_model(file)

    for index_voxel, voxel in enumerate(conflict_voxels):

        test_x = features[:,index_voxel] - train_features[i, index_voxel]

        pred = classifiers[index_voxel].predict_proba(test_x)[:,1]

        preds[index_voxel, :, i] = pred
        if index_voxel % 20000 == 0:
            print(index_voxel)

save_predictions(predictions)
