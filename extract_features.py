#!/usr/bin/env python3
"""
Script to extract and save the feature column names from the original training process.
This ensures exact feature alignment between training and inference.
"""

import pandas as pd
import joblib
import numpy as np

def extract_training_features():
    """Extract the exact feature configuration used during training."""
    
    # Load the model to get feature count
    try:
        model = joblib.load('logistic_regression_model.pkl')
        n_features = model.coef_.shape[1]
        print(f"Model expects {n_features} features")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Load and process data exactly as done in training
    try:
        train_df = pd.read_csv('nlp-getting-started/train.csv')
        test_df = pd.read_csv('nlp-getting-started/test.csv')
        
        # Fill missing values with 'unknown' as done in training
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df['keyword'] = train_df['keyword'].fillna('unknown')
        train_df['location'] = train_df['location'].fillna('unknown')
        test_df['keyword'] = test_df['keyword'].fillna('unknown')
        test_df['location'] = test_df['location'].fillna('unknown')
        
        # Create combined dataframe as done in training
        combined_df = pd.concat([train_df.drop('target', axis=1), test_df], ignore_index=True)
        
        # Get one-hot encoded columns
        encoded_df = pd.get_dummies(combined_df[['keyword', 'location']])
        categorical_features = encoded_df.shape[1]
        
        print(f"Categorical features: {categorical_features}")
        print(f"Expected Word2Vec features: {n_features - categorical_features}")
        
        # Save the column configuration
        feature_config = {
            'categorical_columns': encoded_df.columns.tolist(),
            'n_categorical_features': categorical_features,
            'n_total_features': n_features,
            'word2vec_features': n_features - categorical_features
        }
        
        # Save to file
        joblib.dump(feature_config, 'feature_config.pkl')
        print(f"Feature configuration saved to 'feature_config.pkl'")
        print(f"Total categorical columns: {len(feature_config['categorical_columns'])}")
        
        return feature_config
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    config = extract_training_features()
    if config:
        print("✅ Feature configuration extracted successfully!")
        print(f"Save this file alongside your model files for consistent inference.")
    else:
        print("❌ Failed to extract feature configuration.")