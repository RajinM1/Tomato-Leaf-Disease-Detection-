#!/usr/bin/env python3
"""
Script to load the exact trained model architecture and weights
"""
import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_exact_model():
    """Create the exact model architecture from the H5 file"""
    print("Creating exact model architecture...")
    
    # Based on the H5 file inspection, create the exact architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3), name='conv2d'),
        MaxPooling2D(2, 2, name='max_pooling2d'),
        Conv2D(64, (3, 3), activation='relu', name='conv2d_1'),
        MaxPooling2D(2, 2, name='max_pooling2d_1'),
        Conv2D(128, (3, 3), activation='relu', name='conv2d_2'),
        MaxPooling2D(2, 2, name='max_pooling2d_2'),
        Flatten(name='flatten'),
        Dense(128, activation='relu', name='dense'),  # 128 units, not 512!
        Dropout(0.5, name='dropout'),
        Dense(10, activation='softmax', name='dense_1')
    ])
    
    return model

def load_trained_weights():
    """Load the trained weights into the model"""
    print("Loading trained weights...")
    
    try:
        # Create the exact model architecture
        model = create_exact_model()
        
        # Load weights from the H5 file
        model.load_weights("model/tomato_leaf_disease_model.h5")
        print("SUCCESS: Trained weights loaded!")
        
        # Test the model
        test_input = np.random.random((1, 128, 128, 3))
        prediction = model.predict(test_input, verbose=0)
        print(f"Model output shape: {prediction.shape}")
        print(f"Sample prediction: {prediction[0]}")
        
        return model
        
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return None

def save_trained_model(model):
    """Save the model with trained weights"""
    if model is None:
        print("No model to save")
        return False
    
    try:
        # Save as .keras format
        model.save("model/tomato_leaf_disease_model_trained.keras")
        print("SUCCESS: Saved trained model as .keras format")
        
        # Save as SavedModel format
        model.save("model/tomato_leaf_disease_model_trained_savedmodel")
        print("SUCCESS: Saved trained model as SavedModel format")
        
        return True
        
    except Exception as e:
        print(f"Failed to save model: {e}")
        return False

def main():
    print("Loading Trained Tomato Leaf Disease Model")
    print("=" * 50)
    
    # Load the trained model
    model = load_trained_weights()
    
    if model is not None:
        # Save the working model
        if save_trained_model(model):
            print("\nSUCCESS: Trained model with actual weights saved!")
            print("Files created:")
            print("  - model/tomato_leaf_disease_model_trained.keras")
            print("  - model/tomato_leaf_disease_model_trained_savedmodel/")
            print("\nUpdate app.py to use the trained model!")
        else:
            print("ERROR: Could not save the trained model")
    else:
        print("ERROR: Could not load the trained model")

if __name__ == "__main__":
    main()

