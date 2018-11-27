from nilearn import image
from tqdm import tqdm
from nipype.interfaces.ants import RegistrationSynQuick
import nibabel as nib
import os
from nipype.interfaces.ants import ApplyTransforms

folder_data = "data/MICCAI/diencephalon/"
folder_images = folder_data + "training-images/"
folder_labels = folder_data + "training-labels/"
folder_transformations = "transformations/"
folder_mni = "mni/"

path_template = "mni1.nii"
processed = []
transform = True
for image in tqdm(os.listdir(folder_images)):
    if "nii.gz" not in image:
        print(image + "problem")
        continue
    image = image[:-7]
    print(image)
    if image + "_mni.nii.gz" in processed:
        print("image already processed")
        continue
    if transform:
        reg = RegistrationSynQuick()
        reg.inputs.fixed_image = path_template
        reg.inputs.moving_image = folder_images + image + ".nii.gz"
        reg.inputs.num_threads = 2
        reg.verbose = False
        #reg.output_prefix = image + "_"
        reg.use_histogram_matching = True
        print(reg.cmdline)
        reg.run()

    at = ApplyTransforms()
    at.inputs.input_image = folder_labels + image + "_glm.nii.gz"
    at.inputs.reference_image = path_template
    at.inputs.interpolation = "MultiLabel"
    at.inputs.transforms = [folder_transformations + image + '_transform0GenericAffine.mat',
                            folder_transformations + image + '_transform1Warp.nii.gz']

    if transform:
        at.inputs.transforms = ["transform0GenericAffine.mat", "transform1Warp.nii.gz"]
    print(at.cmdline)
    at.run()
    if transform:
        os.rename('transform0GenericAffine.mat', folder_transformations + image + '_transform0GenericAffine.mat')
        os.rename('transform1Warp.nii.gz', folder_transformations + image + '_transform1Warp.nii.gz')
        os.rename('transform1InverseWarp.nii.gz', folder_transformations + image + '_transform1InverseWarp.nii.gz')
        os.rename('transformInverseWarped.nii.gz', folder_transformations + image + '_transformInverseWarped.nii.gz')
        os.rename('transformWarped.nii.gz', folder_mni + image + '_mni.nii.gz')

    os.rename(image + '_glm_trans.nii.gz', folder_mni + image + '_glm_mni.nii.gz')
