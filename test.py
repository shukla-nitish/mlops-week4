import unittest
import numpy as np
from sklearn.metrics import accuracy_score
from joblib import load

class TestIrisModel(unittest.TestCase):
    def setUp(self):
        # Load the trained model
        cls.model = load('model.joblib')
        # Load the Iris dataset
        iris = load_iris()
        cls.X = iris.data
        cls.y = iris.target

    # --- Data Validation Tests ---
    def test_feature_shape(self):
        """Check that the feature matrix has correct shape (150, 4)."""
        self.assertEqual(self.X.shape, (150, 4), "Feature matrix shape mismatch.")

    def test_label_values(self):
        """Check that all labels are within expected range (0, 1, 2)."""
        unique_labels = np.unique(self.y)
        self.assertTrue(np.all(np.isin(unique_labels, [0, 1, 2])), "Unexpected label values.")

    # --- Model Evaluation Tests ---
    def test_model_accuracy(self):
        """Check that model accuracy on the iris dataset is above 90%."""
        y_pred = self.model.predict(self.X)
        acc = accuracy_score(self.y, y_pred)
        self.assertGreaterEqual(acc, 0.90, f"Model accuracy too low: {acc}")

    def test_single_prediction(self):
        """Test that the model predicts a known class for a sample."""
        sample = self.X[0].reshape(1, -1)
        pred = self.model.predict(sample)[0]
        self.assertIn(pred, [0, 1, 2], "Prediction not in expected classes.")

if __name__ == '__main__':
    unittest.main()

