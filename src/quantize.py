import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import os

class SimpleLinearNet(nn.Module):
    """Single-layer PyTorch neural network"""
    def __init__(self, input_size):
        super(SimpleLinearNet, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.linear(x).squeeze()

def load_sklearn_model():
    """Load the trained scikit-learn model"""
    model_path = 'models/california_housing_model.joblib'
    test_data_path = 'models/test_data.joblib'
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please run train.py first to generate the model.")
        return None, None, None
    
    model = joblib.load(model_path)
    test_data = joblib.load(test_data_path)
    
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    print(f"Loaded model with coefficients shape: {model.coef_.shape}")
    print(f"Loaded model intercept: {model.intercept_}")
    
    return model, X_test, y_test

def extract_parameters(sklearn_model):
    """Extract parameters from scikit-learn model"""
    coef = sklearn_model.coef_.copy()
    intercept = sklearn_model.intercept_
    
    params = {
        'weights': coef,
        'bias': intercept
    }
    
    print(f"Extracted weights shape: {coef.shape}")
    print(f"Extracted bias: {intercept}")
    
    return params

def manual_quantization(params, num_bits=8):
    """Perform manual quantization to 8-bit unsigned integers"""
    quantized_params = {}
    quantization_info = {}

    for param_name, param_values in params.items():
        values = np.array(param_values, dtype=np.float32)

        min_val = float(np.min(values))
        max_val = float(np.max(values))

        if max_val == min_val:
            print(f"All values for '{param_name}' are constant ({min_val}). Quantizing to zeros.")
            scale = 1.0  # Arbitrary non-zero scale to avoid divide-by-zero
            zero_point = 0
            quantized = np.zeros_like(values, dtype=np.uint8)
        else:
            scale = (max_val - min_val) / (2**num_bits - 1)
            zero_point = int(np.round(-min_val / scale))
            zero_point = np.clip(zero_point, 0, 2**num_bits - 1)
            quantized = np.round(values / scale + zero_point)
            quantized = np.clip(quantized, 0, 2**num_bits - 1).astype(np.uint8)

        quantized_params[param_name] = quantized
        quantization_info[param_name] = {
            'scale': scale,
            'zero_point': zero_point,
            'min_val': min_val,
            'max_val': max_val
        }

        print(f"Quantized {param_name}:")
        print(f"  Original range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  Scale: {scale:.6f}")
        print(f"  Zero point: {zero_point}")
        print(f"  Quantized shape: {quantized.shape}")

    return quantized_params, quantization_info

def dequantize_parameters(quantized_params, quantization_info):
    """Dequantize parameters back to float32"""
    dequantized_params = {}
    
    for param_name, quantized_values in quantized_params.items():
        info = quantization_info[param_name]
        
        # Dequantize: (quantized - zero_point) * scale
        dequantized = (quantized_values.astype(np.float32) - info['zero_point']) * info['scale']
        dequantized_params[param_name] = dequantized
        
        print(f"Dequantized {param_name} shape: {dequantized.shape}")
    
    return dequantized_params

def create_pytorch_model(dequantized_params, input_size):
    """Create PyTorch model with dequantized weights"""
    model = SimpleLinearNet(input_size)

    # Set weights and bias explicitly as float32
    with torch.no_grad():
        model.linear.weight.data = torch.tensor(
            dequantized_params['weights'].astype(np.float32)
        ).unsqueeze(0)
        model.linear.bias.data = torch.tensor(
            [dequantized_params['bias']], dtype=torch.float32
        )

    return model

def evaluate_models(original_model, pytorch_model, X_test, y_test):
    """Evaluate both original and quantized models"""
    # Original sklearn model predictions
    y_pred_original = original_model.predict(X_test)
    r2_original = r2_score(y_test, y_pred_original)
    
    # PyTorch model predictions
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred_quantized = pytorch_model(X_test_tensor).numpy()
    r2_quantized = r2_score(y_test, y_pred_quantized)
    
    print(f"\n Model Comparison:")
    print(f"Original R² Score: {r2_original:.6f}")
    print(f"Quantized R² Score: {r2_quantized:.6f}")
    print(f"R² Difference: {abs(r2_original - r2_quantized):.6f}")
    
    return r2_original, r2_quantized

def calculate_model_sizes():
    """Calculate and return model sizes"""
    unquant_size = os.path.getsize('models/unquant_params.joblib') / 1024  # KB
    quant_size = os.path.getsize('models/quant_params.joblib') / 1024  # KB
    
    print(f"\n Model Sizes:")
    print(f"Unquantized model: {unquant_size:.2f} KB")
    print(f"Quantized model: {quant_size:.2f} KB")
    print(f"Size reduction: {(1 - quant_size/unquant_size)*100:.1f}%")
    
    return unquant_size, quant_size

def main():
    print("Starting quantization process...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load scikit-learn model
    sklearn_model, X_test, y_test = load_sklearn_model()
    if sklearn_model is None:
        return
    
    # Extract parameters
    original_params = extract_parameters(sklearn_model)
    
    # Save unquantized parameters
    joblib.dump(original_params, 'models/unquant_params.joblib')
    print("Saved unquantized parameters to models/unquant_params.joblib")
    
    # Perform quantization
    quantized_params, quantization_info = manual_quantization(original_params)
    
    # Save quantized parameters
    quant_data = {
        'quantized_params': quantized_params,
        'quantization_info': quantization_info
    }
    joblib.dump(quant_data, 'models/quant_params.joblib')
    print("Saved quantized parameters to models/quant_params.joblib")
    
    # Dequantize for inference
    dequantized_params = dequantize_parameters(quantized_params, quantization_info)
    
    # Create PyTorch model with dequantized weights
    input_size = X_test.shape[1]
    pytorch_model = create_pytorch_model(dequantized_params, input_size)
    
    # Evaluate models
    r2_original, r2_quantized = evaluate_models(sklearn_model, pytorch_model, X_test, y_test)
    
    # Calculate model sizes
    unquant_size, quant_size = calculate_model_sizes()
    
    # Create results summary
    results = {
        'r2_original': r2_original,
        'r2_quantized': r2_quantized,
        'unquant_size_kb': unquant_size,
        'quant_size_kb': quant_size,
        'size_reduction_percent': (1 - quant_size/unquant_size)*100
    }
    
    joblib.dump(results, 'models/quantization_results.joblib')
    
    print("\n Quantization process completed successfully!")
    print(f"Results saved to models/quantization_results.joblib")
    
    return results

if __name__ == "__main__":
    main()
