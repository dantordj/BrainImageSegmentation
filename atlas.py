import numpy as np

class Atlas():


    def __init__(self, id):
        self.id = id
        self.image = None
        self.label = None

    def image_vector(self):
        return self.image.get_fdata().flatten()

    def label_vector(self):
        return self.label.get_fdata().flatten()

    def features(self, conflict_voxels):
        """ à implémente, return un array de size (nbr_voxels, size voisinage)
        avec les différences d'intensité en faisant le matching voxels par voxel
        dans le voisinage """
        print("get features %s" %self.id)
        voxels = self.image.get_fdata()
        init_shape = voxels.shape
        n_conflict = len(conflict_voxels)
        features = np.zeros((n_conflict, 27))
        for ind_vox, vox in enumerate(conflict_voxels):
            i, j, k = np.unravel_index(vox, init_shape)
            r = 0
            for a in range(-1,2):
                for b in range(-1,2):
                    for c in range(-1,2):
                        try:
                            val = voxels[a+i, b+j, c+k]
                        except IndexError:
                            print(a,b,c)
                            val = 0
                        features[ind_vox, r] = val
                        r += 1

        return features
