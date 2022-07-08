from functools import lru_cache
from collections import defaultdict
from math import factorial as fac
import numpy as np
from regex import W
from sklearn.tree import BaseDecisionTree
from utils import cached

#  first make the shap explainer for constant leaf predictor
#  how to we prune the sklearn's DT to have only one path in it?
#  or can we use the existing sklearn DT itself, but play within 
# the path scoped for us?

class Shap:
    def __init__(self,):
        pass
    
    def fit(self, tree: BaseDecisionTree):
        self.tree = tree
        return self

    def explain(self, X, feature):
        # find path of the leaf, that would contain X
        nodes =   self.tree.decision_path(X).nonzero()[1]
        cur_node, path = 0, []
        for node in nodes:
            if cur_node == self.tree.tree_.children_left[cur_node]:
                path.append('l')
            else:
                path.append('r')
            cur_node = node
        self._weights(path, X.ravel())
        # feature_order = tuple(range(self.tree.n_features_))
        # feature_order = set(range(self.tree.n_features_in_))\
        #     - {feature}
        print(self.weights)
        feature_order = (feature,) + tuple(f for f in self.weights.keys()\
            if f != feature)
        # Q
        # if a feature is not present in the path, 
        # then its contribution is zero for that prediction?
        F = len(feature_order)
        # does it need to contain only the feature
        # in the path?
        # yes, need to have only the features in the path
        contrib = 0
        for i in reversed(range(len(self.weights))):
            contrib += (fac(i)* fac(F - i -1)) / fac(F)\
            * self.dp(k_order=feature_order,
                K=len(feature_order)-1, T=i) 
        contrib *= self.tree.predict(X)[0]
        return contrib

    @property
    def branching_prob_(self):
        """
        returns the probability of taking the left branch
        for a particular node based on a ratio of
        training samples that take the left branch.

        Parameters
        ----------
        c_tree : sklearn.tree._tree.Tree

        """
        c_tree = self.tree.tree_
        branching_prob = np.full_like(c_tree.threshold, -2)
        samples = c_tree.n_node_samples
        for node_id, node_feature in enumerate(c_tree.feature):
            if node_feature != -2: # non- leaf node
                left_node_id = c_tree.children_left[node_id]
                branching_prob[node_id] = samples[left_node_id]\
                     / samples[node_id]
        return branching_prob
    # each node would have a prob for taking
    # the left and right children.
    # p(l) = 1- p(r)
    # above will be valid if any one of the children is a leaf?

    def _weights(self, path, x):
        self.weights = defaultdict(lambda: [1, 1])
        # [ind_j(x,e), p_j(e)]
        parent = 0 # node 0 is the tree's root
        for dir in path[:-1]:
            feature = self.tree.tree_.feature[parent]
            threshold = self.tree.tree_.threshold[parent]
            self.weights[feature][0] *= 1 if (x[feature] <= threshold)\
                 == (dir == 'l') else 0
            self.weights[feature][1] *= self.branching_prob_[parent]
            if dir == 'l':
                parent = self.tree.tree_.children_left[parent]
            else:
                parent = self.tree.tree_.children_right[parent]

    @cached
    def dp(self, k_order, K, T):
        if (K< 0) or (T< 0):
            raise ValueError('value of K or T is negative'+
            f'\n{K=}\t{T=}')
        if K < T:
            return 0
        elif (T == 0): # and (K == 0):
            # this branch has to be trigged when T=0 itself
            return self.weights[k_order[0]][0] -\
                self.weights[k_order[0]][1]
        elif K == T:
            return self.dp(k_order, K-1, T-1) *\
                self.weights[k_order[K]][0]
        elif K > T:
            return self.dp(k_order, K-1, T-1) *\
                self.weights[k_order[K]][0] + \
                self.dp(k_order, K-1, T) * \
                    self.weights[k_order[K]][1]
        else:
            raise ValueError(f'Given K and T values : {K} {T},'+
                'does not match any of the defined conditions')


# path -> sequence of features, threshold and connections between nodes
# sequences of directions from root node
# hash map {feaure: [<node_id, direction>]}

# weights function to compute the Ind_j and P_j for every variable i in the path to leaf j.

# Questions:

# K - a specific ordering, where i_th feature is excluded in it; when computing feature
# importance of i_th feature
# i_th feature might occur more than one in a path. 
# should we exclude all the occurances of i? then final |k| will not be equal to |F|
# => F is the unique set of features

# for V(0,0),  prod of missing features' p_j (probability) is not accounted?
# which is comming for both presence and absence of i_th feature.