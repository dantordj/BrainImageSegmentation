path_train = "mni/"
# can be naive, mb or logReg
model = "logReg"

# can be 0 or 1
patch = 1
path_test = "mni_val/"
path_predictions = "pred_labels_%s/"%model
path_classifiers = "classifiers_%s/"%model

if model == "logReg":
    path_predictions = "pred_labels_%s_%s/"%(model, patch)
