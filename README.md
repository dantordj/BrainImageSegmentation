# BrainImageSegmentation

In this repos we implement the method describre in the article *Discriminative confidence estimation for probabilistic multi-atlas label fusion,Benkarim OM1, Piella G2, González Ballester MA3, Sanroma G 
*

## how to run the code 
To file ants.py allows to register the images to the mni template. 

The file train.py allows to train the classifiers to compute the coefficients Cij defined in the article. 

The file predict.py allows to perform the prediction on the validation images and to compute the Dice scores. 


Change the config file to perform the segmentation with : 

- majority voting 
- the naive approach 
- the local approach with logistic regressions

## requirements:
- nilearn
- numpy 
- nibabel
- sklearn
- tqdm 


