import unittest
from sklearn.metrics import accuracy_score


class TestIrisModel(unittest.TestCase):
    model = None
    model_path = 'model.joblib'
    X = None
    y = None
    sample_path = "samples"
    
    def setUp(self):
        from joblib import load
        import pandas as pd
        self.model = load(self.model_path)
        data = pd.read_csv(self.sample_path + "/sample.csv")
        self.X = data.iloc[:,:4]
        self.y = data.iloc[:,4]

    # --- Data Validation Tests ---
    def test_feature_shape(self):
        """Check that the feature matrix has correct number of features."""
        self.assertEqual(self.X.shape[1],  4, "Feature matrix shape mismatch.")

    def test_label_values(self):
        import numpy as np
        """Check that all labels are among expected values."""
        unique_labels = np.unique(self.y)
        self.assertTrue(np.all(np.isin(unique_labels, ['setosa', 'versicolor', 'virginica'])), "Unexpected label values.")

    # --- Model Evaluation Tests ---
    def test_model_accuracy(self):
        """Check that model accuracy on the iris dataset is above 90%."""
        y_pred = self.model.predict(self.X)
        acc = accuracy_score(self.y, y_pred)
        self.assertGreaterEqual(acc, 0.90, f"Model accuracy too low: {acc}")

    def test_single_prediction(self):
        """Test that the model predicts a known class for a sample."""
        sample = self.X.iloc[[0]]
        pred = self.model.predict(sample)[0]
        self.assertIn(pred, ['setosa', 'versicolor', 'virginica'], "Prediction not in expected classes.")

if __name__ == '__main__':
    unittest.main()

