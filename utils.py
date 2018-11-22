import numpy as np
import os
import pickle
path = ""
save_path = path + "classifiers/"

def get_conflict_voxels(atlases, all_voxels=True, save=True):
  # crée une liste avec les voxels conflictuel. Représentation de l'image sous forme d'un vecteur. Voir fonction flatten
  conflict_voxels = []
  labels = np.array([atlas.label_vector() for atlas in atlases.values()], dtype=int)
  print(labels.shape)
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
    pkl_filename = save_path + file
    print("saving", pkl_filename)
    with open(pkl_filename, 'wb') as file:

        pickle.dump(classifiers, file)



def save_predictions(predictions):
    pkl_filename = save_path + "predictions"
    print("saving", pkl_filename)
    with open(pkl_filename, 'wb') as file:

        pickle.dump(predictions, file)

def load_model(file):
    pkl_filename = save_path + file
    classifiers = []
    with open(pkl_filename, 'rb') as file:
        classifiers = pickle.load(file)
    return classifiers



def create_directories(atlases):
    for index in atlases.keys():
        directory = save_path + str(index) + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
