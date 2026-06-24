#!/usr/bin/env python3
"""
Script to convert legacy H5 model to compatible format
"""
import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def inspect_model_structure():
    """Inspect the H5 model structure"""
    print("Inspecting model structure...")
    with h5py.File("model/tomato_leaf_disease_model.h5", 'r') as f:
        print("Model file structure:")
        def print_attrs(name, obj):
            if hasattr(obj, 'attrs'):
                attrs = dict(obj.attrs)
                if attrs:
                    print(f"{name}: {attrs}")
        f.visititems(print_attrs)

def create_compatible_model():
    """Create a new model with compatible architecture"""
    print("Creating compatible model architecture...")
    
    # Create a robust CNN architecture for plant disease classification
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Fourth Conv Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Fifth Conv Block
        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Dense layers
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')  # 10 classes
    ])
    
    return model

def try_load_weights(model):
    """Try to load weights from the original model"""
    print("Attempting to load weights...")
    
    try:
        # Try different weight loading methods
        model.load_weights("model/tomato_leaf_disease_model.h5")
        print("SUCCESS: Weights loaded successfully!")
        return True
    except Exception as e:
        print(f"ERROR: Could not load weights: {e}")
        return False

def save_compatible_model(model):
    """Save the model in a compatible format"""
    print("Saving compatible model...")
    
    try:
        # Save as SavedModel format (most compatible)
        model.save("model/tomato_leaf_disease_model_savedmodel")
        print("SUCCESS: Saved as SavedModel format")
        
        # Also save as .keras format
        model.save("model/tomato_leaf_disease_model.keras")
        print("SUCCESS: Saved as .keras format")
        
        return True
    except Exception as e:
        print(f"ERROR: Could not save model: {e}")
        return False

def main():
    print("Tomato Leaf Disease Model Converter")
    print("=" * 50)
    
    # Step 1: Inspect original model
    inspect_model_structure()
    
    # Step 2: Create compatible model
    model = create_compatible_model()
    print(f"Model created with {model.count_params()} parameters")
    
    # Step 3: Try to load weights
    weights_loaded = try_load_weights(model)
    
    if not weights_loaded:
        print("WARNING: Using random weights - model needs retraining")
        print("INFO: The model architecture is ready, but weights need to be trained")
    
    # Step 4: Save compatible model
    if save_compatible_model(model):
        print("\nSUCCESS: Model conversion completed!")
        print("New model files:")
        print("   - model/tomato_leaf_disease_model_savedmodel/")
        print("   - model/tomato_leaf_disease_model.keras")
        print("\nINFO: Update app.py to use the new model format")
    else:
        print("ERROR: Model conversion failed")

if __name__ == "__main__":
    main()
