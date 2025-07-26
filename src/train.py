import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

def load_data():
    """Load and prepare the California Housing dataset"""
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {feature_names}")
    
    return X, y, feature_names

def train_model(X_train, y_train):
    """Train a Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"Model coefficients shape: {model.coef_.shape}")
    print(f"Model intercept: {model.intercept_}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    
    return r2, mse

def save_model(model, X_test, y_test, filename='california_housing_model.joblib'):
    """Save the trained model and test data"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = os.path.join('models', filename)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save test data for verification
    test_data = {'X_test': X_test, 'y_test': y_test}
    test_path = os.path.join('models', 'test_data.joblib')
    joblib.dump(test_data, test_path)
    print(f"Test data saved to {test_path}")
    
    return model_path

def main():
    print("Starting California Housing model training...")
    
    # Load data
    X, y, feature_names = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    r2, mse = evaluate_model(model, X_test, y_test)
    
    # Save model and test data
    model_path = save_model(model, X_test, y_test)
    
    print("Training completed successfully!")
    
    return model, r2, mse

if __name__ == "__main__":
    main()
