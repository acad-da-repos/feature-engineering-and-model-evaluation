
import unittest
import numpy as np
from assignment import create_polynomial_features, evaluate_model

class TestFeatureEngineeringAndEvaluation(unittest.TestCase):
    def test_create_polynomial_features(self):
        X = np.array([[1, 2], [3, 4]])
        poly_X = create_polynomial_features(X)
        self.assertEqual(poly_X.shape, (2, 5)) # Original 2 features + 3 polynomial features (x1^2, x2^2, x1*x2)

    def test_evaluate_model(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        mse, r2 = evaluate_model(y_true, y_pred)
        self.assertIsInstance(mse, float)
        self.assertIsInstance(r2, float)

if __name__ == '__main__':
    unittest.main()
