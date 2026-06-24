#!/usr/bin/env python3
"""
Script to properly load and convert the trained model with actual weights
"""
import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def inspect_original_model():
    """Inspect the original H5 model to understand its structure"""
    print("Inspecting original model structure...")
    
    with h5py.File("model/tomato_leaf_disease_model.h5", 'r') as f:
        print("Model structure:")
        
        # Check if it has model_config
        if 'model_config' in f.attrs:
            print("Model config found")
            config = f.attrs['model_config']
            print(f"Config type: {type(config)}")
        
        # Check layer names
        if 'model_weights' in f:
            weights_group = f['model_weights']
            print("Layer names in weights:")
            for name in weights_group.keys():
                print(f"  - {name}")
                
        # Try to get the actual model architecture
        try:
            # This might work for some H5 files
            import json
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config'].decode('utf-8') if isinstance(f.attrs['model_config'], bytes) else f.attrs['model_config']
                config = json.loads(config_str)
                print("Model architecture:")
                print(json.dumps(config, indent=2))
        except Exception as e:
            print(f"Could not parse model config: {e}")

def create_model_from_weights():
    """Try to create a model that matches the original architecture"""
    print("Creating model to match original architecture...")
    
    # Based on the layer names we saw, create a matching architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3), name='conv2d'),
        MaxPooling2D(2, 2, name='max_pooling2d'),
        Conv2D(64, (3, 3), activation='relu', name='conv2d_1'),
        MaxPooling2D(2, 2, name='max_pooling2d_1'),
        Conv2D(128, (3, 3), activation='relu', name='conv2d_2'),
        MaxPooling2D(2, 2, name='max_pooling2d_2'),
        Flatten(name='flatten'),
        Dense(512, activation='relu', name='dense'),
        Dropout(0.5, name='dropout'),
        Dense(10, activation='softmax', name='dense_1')
    ])
    
    return model

def try_load_weights_directly():
    """Try to load weights directly from the H5 file"""
    print("Attempting to load weights directly...")
    
    try:
        # Create model with matching architecture
        model = create_model_from_weights()
        
        # Try to load weights
        model.load_weights("model/tomato_leaf_disease_model.h5")
        print("SUCCESS: Weights loaded directly!")
        
        # Test the model
        test_input = np.random.random((1, 128, 128, 3))
        prediction = model.predict(test_input, verbose=0)
        print(f"Model output shape: {prediction.shape}")
        print(f"Sample prediction: {prediction[0][:5]}")  # First 5 values
        
        return model
        
    except Exception as e:
        print(f"Direct weight loading failed: {e}")
        return None

def try_alternative_loading():
    """Try alternative methods to load the model"""
    print("Trying alternative loading methods...")
    
    try:
        # Method 1: Try with custom objects that handle the batch_shape issue
        class CustomInputLayer:
            def __init__(self, **kwargs):
                # Remove problematic parameters
                kwargs.pop('batch_shape', None)
                self.input_shape = kwargs.get('input_shape', (128, 128, 3))
                self.name = kwargs.get('name', 'input_layer')
                self.dtype = kwargs.get('dtype', 'float32')
        
        custom_objects = {
            'InputLayer': CustomInputLayer,
        }
        
        model = tf.keras.models.load_model(
            "model/tomato_leaf_disease_model.h5", 
            compile=False, 
            custom_objects=custom_objects
        )
        
        print("SUCCESS: Model loaded with custom objects!")
        
        # Test the model
        test_input = np.random.random((1, 128, 128, 3))
        prediction = model.predict(test_input, verbose=0)
        print(f"Model output shape: {prediction.shape}")
        print(f"Sample prediction: {prediction[0][:5]}")
        
        return model
        
    except Exception as e:
        print(f"Alternative loading failed: {e}")
        return None

def save_working_model(model):
    """Save the working model in compatible formats"""
    if model is None:
        print("No model to save")
        return False
    
    try:
        # Save as .keras format
        model.save("model/tomato_leaf_disease_model_fixed.keras")
        print("SUCCESS: Saved as .keras format")
        
        # Save as SavedModel format
        model.save("model/tomato_leaf_disease_model_fixed_savedmodel")
        print("SUCCESS: Saved as SavedModel format")
        
        return True
        
    except Exception as e:
        print(f"Failed to save model: {e}")
        return False

def main():
    print("Tomato Leaf Disease Model Weight Loader")
    print("=" * 50)
    
    # Step 1: Inspect original model
    inspect_original_model()
    
    # Step 2: Try to load weights directly
    model = try_load_weights_directly()
    
    if model is None:
        # Step 3: Try alternative loading
        model = try_alternative_loading()
    
    if model is not None:
        # Step 4: Save the working model
        if save_working_model(model):
            print("\nSUCCESS: Model with trained weights saved!")
            print("Update app.py to use: model/tomato_leaf_disease_model_fixed.keras")
        else:
            print("ERROR: Could not save the model")
    else:
        print("ERROR: Could not load the trained model")
        print("The model may need to be retrained with current TensorFlow")

if __name__ == "__main__":
    main()

