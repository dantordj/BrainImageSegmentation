from nilearn import image
from tqdm import tqdm
from nipype.interfaces.ants import RegistrationSynQuick
import nibabel as nib
import os
from nipype.interfaces.ants import ApplyTransforms

folder_data = "data/MICCAI/diencephalon/"
folder_images = folder_data + "testing-images/"
folder_transformations = "transformations_test/"
folder_mni = "mni_test/"
path_template = "mni1.nii"
processed = os.listdir(folder_mni)
transform = True

for image in tqdm(os.listdir(folder_images)):
    if transform:
        if "nii.gz" not in image:
            print(image + "problem")
            continue
        image = image[:-7]
        print(image)
        if image + "_mni.nii.gz" in processed:
            print("image already processed")
            continue



        reg = RegistrationSynQuick()
        reg.inputs.fixed_image = path_template
        reg.inputs.moving_image = folder_images + image + ".nii.gz"
        reg.inputs.num_threads = 2
        reg.verbose = False
        #reg.output_prefix = image + "_"
        reg.use_histogram_matching = True

        reg.run()
        os.rename('transform0GenericAffine.mat', folder_transformations + image + '_transform0GenericAffine.mat')
        os.rename('transform1Warp.nii.gz', folder_transformations + image + '_transform1Warp.nii.gz')
        os.rename('transform1InverseWarp.nii.gz', folder_transformations + image + '_transform1InverseWarp.nii.gz')
        os.rename('transformInverseWarped.nii.gz', folder_transformations + image + '_transformInverseWarped.nii.gz')
        os.rename('transformWarped.nii.gz', folder_mni + image + '_mni.nii.gz')
        
