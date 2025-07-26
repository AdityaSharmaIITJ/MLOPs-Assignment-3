import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os

def load_model_and_data():
    """Load the trained model and test data"""
    try:
        model_path = 'models/california_housing_model.joblib'
        test_data_path = 'models/test_data.joblib'
        
        # Load model
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        
        # Load test data
        test_data = joblib.load(test_data_path)
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        print(f"Test data loaded successfully from {test_data_path}")
        
        return model, X_test, y_test
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def verify_model(model, X_test, y_test):
    """Verify the model by making predictions"""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"Model verification results:")
        print(f"R² Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"Predictions shape: {y_pred.shape}")
        print(f"Sample predictions: {y_pred[:5]}")
        
        # Verify model parameters
        print(f"Model coefficients shape: {model.coef_.shape}")
        print(f"Model intercept: {model.intercept_:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Model verification failed: {e}")
        return False

def main():
    print("Starting model verification in Docker container...")
    
    # Check if required files exist
    required_files = ['models/california_housing_model.joblib', 'models/test_data.joblib']
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Required file not found: {file_path}")
            return False
    
    try:
        # Load model and test data
        model, X_test, y_test = load_model_and_data()
        
        # Verify model
        success = verify_model(model, X_test, y_test)
        
        if success:
            print("Model verification completed successfully!")
            return True
        else:
            print("Model verification failed!")
            return False
            
    except Exception as e:
        print(f"Verification process failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
