# extracts all paths in a decision tree
# apply dp for each path after that.
def leaf_path(self, X):
    ans = []
    stack = [0, []]
    while stack:
        node, path = stack.pop()
        if node == 'leaf':
            weights = self._weights(path, X)
            feature_contri  = self.dp(
                k_order=list(weights.keys()),
                K=len(weights),
                T=len(weights), weights=weights)
            ans.append(feature_contri)
        else:
            left_child = self.tree.tree_.children_left[node]
            right_child = self.tree.tree_.children_right[node]
            if  left_child== -2:
                stack.append('leaf', path+['l'])
            else:
                stack.append(left_child, path+ ['l'])
            if right_child == -2:
                stack.append('leaf', path+['r'])
            else:
                stack.append(right_child, path+['r'])
