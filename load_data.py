# Pour tester sur un nombre réduit d'atlas, mettre all indices à True pour utiliser tous les atlas
from tqdm import tqdm
import nibabel as nib
import os
from atlas import Atlas

test_indices = [1107, 1110, 1116]

def load_images(path,atlases, all_images=True):
  " Load toutes les images, à checker quand toutes les images seront uploader"

  for image in tqdm(os.listdir(path)):
    if 'glm' not in image:

      try:
        index = int(image[:-13])
      except ValueError:
        continue

      if index not in test_indices and not all_images:
        continue
      atlas = Atlas(index)
      image = nib.load(path + image)

      atlas.image = image

      atlases[index] = atlas
      #plotting.plot_img(image)
      #print(atlas.image)
  return atlases


def load_labels(path, atlases, all_images=True):
  # load tous les labels. certains labels sont en double donc pb à regler
  for image in tqdm(os.listdir(path)):
    if 'glm' in image:

      try:
         index = int(image[:-17])
      except ValueError:
        continue

      if index not in test_indices and not all_images:
        continue


      image_label = nib.load(path + image)

      try:
        atlas = atlases[index]
      except KeyError:
        print("missing image %s" %index)
        continue
      atlas.label = image_label

      atlases[index] = atlas


  return atlases

def load_data(path, all_images=True, labels=True):
  # load toutes les données, format : dictionnary {id:atlas}

  atlases = {}
  load_images(path,atlases, all_images)
  if labels:
      load_labels(path, atlases, all_images)
  return atlases
