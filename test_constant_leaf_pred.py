#%%
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression

from constant_leaf_pred import Shap

class TestShap():
    def setup(self):
        self.rng = np.random.RandomState(9)
        X, y = make_regression(n_samples=1000, n_features=6,
                n_informative=6, random_state=self.rng)
        tree = DecisionTreeRegressor(   
            max_depth=4, random_state=self.rng).fit(X, y)
        self.shap = Shap().fit(tree)


    def test_node_branching_prob(self,):
        np.testing.assert_array_almost_equal(
            self.shap.branching_prob_,
            [0.519, 0.37379576, -2., -2., 0.51767152, -2., -2.])


    def test_shap_explain(self,):
        test_x = self.rng.rand(1,3)
        total_contrib = 0
        for i in range(3):
            contrib = self.shap.explain(test_x, i)
            print(f'feature-{i} : contrib {contrib}')
            total_contrib += contrib
        assert total_contrib == \
            self.shap.tree.predict(test_x)[0]

    def test_dp(self,):
        test_x = self.rng.rand(1,6)
        self.shap.explain(test_x)
        assert False