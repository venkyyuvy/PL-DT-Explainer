def weights_computer(self, path, present_features, X):
    """compute the w_i(x, S)

    Parameters
    ----------
    path_feature : iterator of ints
        list features used in the nodes along a path to
        specific leaf
    present_features : iterator of ints
        list features that are present for the model to
        make prediction. It can preferrably in set
        for performance.
    X : iterator of ints
        input datapoint for which the feature contribution
        needs to be evaluated

    Returns
    -------
    weight: float
        weightage share of the present features for the prediction
        of leaf value 
    """
    out_weight = 1
    parent = 0 # node 0 is the tree's root
    for dir in path:
        feature = self.tree.tree_.feature[parent]
        threshold = self.tree.tree_.threshold[parent]
        if feature in present_features:
            out_weight *= 1 if (X[feature] <= threshold)\
                    == (dir == 'l') else 0
        else:
            out_weight *= self.branching_prob[parent]
        if dir == 'l':
            parent = self.tree.tree_.children_left[parent]
        else:
            parent = self.tree.tree_.children_right[parent]

# path => list of directions from the root node
# (left or right branching)
# does the shap surrogate model needs to return zero
# when the path is applicable
# for the datapoint of interest
#  => shap values would be based on A single path for a datapoint?