#!/usr/bin/env python3
"""
Advanced diagnostic script to identify exact feature mismatch cause
"""

import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path

def clean_text(text):
    """Clean tweet text - exact copy from app.py"""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = text.lower().strip()
    return text

def tweet_vector(text, model):
    """Convert text to vector - exact copy from app.py"""
    if not text or not isinstance(text, str):
        return np.zeros(model.vector_size)
    
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def investigate_feature_mismatch():
    """Deep dive into the feature mismatch issue"""
    print("🔍 Advanced Feature Mismatch Investigation")
    print("=" * 50)
    
    # Load all components
    try:
        model_combined = joblib.load('logistic_regression_model.pkl')
        word2vec_model = joblib.load('word2vec_model.pkl')
        feature_config = joblib.load('feature_config.pkl')
    except Exception as e:
        print(f"❌ Error loading components: {e}")
        return
    
    # Load actual training data to compare
    try:
        data_dir = Path("nlp-getting-started")
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        
        # Fill missing values exactly as in notebook
        train_df['keyword'] = train_df['keyword'].fillna('unknown')
        train_df['location'] = train_df['location'].fillna('unknown')
        test_df['keyword'] = test_df['keyword'].fillna('unknown')
        test_df['location'] = test_df['location'].fillna('unknown')
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print(f"📊 Model Configuration:")
    print(f"   Expected features: {model_combined.n_features_in_}")
    print(f"   Word2Vec vector size: {word2vec_model.vector_size}")
    print(f"   Saved categorical features: {len(feature_config['categorical_columns'])}")
    
    # Test with actual training data sample
    sample_tweet = train_df.iloc[0]
    print(f"\n🧪 Testing with training sample:")
    print(f"   Text: '{sample_tweet['text'][:50]}...'")
    print(f"   Keyword: '{sample_tweet['keyword']}'")
    print(f"   Location: '{sample_tweet['location']}'")
    
    # Generate embeddings exactly as in the app
    cleaned_text = clean_text(sample_tweet['text'])
    text_embedding = tweet_vector(cleaned_text, word2vec_model)
    print(f"   Cleaned text: '{cleaned_text[:50]}...'")
    print(f"   Text embedding shape: {text_embedding.shape}")
    
    # Generate categorical features exactly as in the app
    temp_df = pd.DataFrame({
        'keyword': [sample_tweet['keyword']], 
        'location': [sample_tweet['location']]
    })
    temp_encoded = pd.get_dummies(temp_df[['keyword', 'location']])
    temp_encoded = temp_encoded.reindex(columns=feature_config['categorical_columns'], fill_value=0)
    print(f"   Categorical features shape: {temp_encoded.shape}")
    print(f"   Categorical features sum: {temp_encoded.sum().sum()}")
    
    # Compare with training approach
    print(f"\n🔬 Training Data Approach Comparison:")
    
    # Create combined dataframe like in training
    combined_df = pd.concat([train_df[['keyword', 'location']], test_df[['keyword', 'location']]], ignore_index=True)
    training_encoded = pd.get_dummies(combined_df[['keyword', 'location']])
    print(f"   Training encoded shape: {training_encoded.shape}")
    print(f"   Training columns count: {len(training_encoded.columns)}")
    
    # Check if columns match
    saved_cols = set(feature_config['categorical_columns'])
    training_cols = set(training_encoded.columns.tolist())
    
    missing_in_saved = training_cols - saved_cols
    extra_in_saved = saved_cols - training_cols
    
    if missing_in_saved:
        print(f"   ⚠️  Columns in training but not in saved config: {missing_in_saved}")
    if extra_in_saved:
        print(f"   ⚠️  Columns in saved config but not in training: {extra_in_saved}")
    
    # Final feature combination
    features = np.hstack((text_embedding.reshape(1, -1), temp_encoded))
    print(f"\n📏 Final Feature Analysis:")
    print(f"   Text features: {text_embedding.shape[0]}")
    print(f"   Categorical features: {temp_encoded.shape[1]}")
    print(f"   Total generated: {features.shape[1]}")
    print(f"   Expected by model: {model_combined.n_features_in_}")
    print(f"   Difference: {features.shape[1] - model_combined.n_features_in_}")
    
    # Try to identify the exact issue
    if features.shape[1] == model_combined.n_features_in_ + 2:
        print(f"\n🎯 DIAGNOSIS: Exactly 2 extra features generated")
        print(f"   This suggests either:")
        print(f"   1. Word2Vec should use 98 dimensions instead of 100")
        print(f"   2. Or 2 categorical features are incorrectly added")
        
        # Test prediction with truncated Word2Vec
        text_emb_98 = text_embedding[:98]  # Take only first 98 dimensions
        features_98 = np.hstack((text_emb_98.reshape(1, -1), temp_encoded))
        print(f"   Testing with 98 Word2Vec features: {features_98.shape[1]} features")
        
        if features_98.shape[1] == model_combined.n_features_in_:
            print(f"   ✅ SUCCESS! Using 98 Word2Vec dimensions fixes the issue")
            
            # Test prediction
            try:
                prediction = model_combined.predict(features_98)[0]
                prediction_proba = model_combined.predict_proba(features_98)[0]
                print(f"   🎉 Prediction works: {'Disaster' if prediction == 1 else 'Not Disaster'}")
                print(f"   Confidence: {max(prediction_proba):.3f}")
                return True
            except Exception as e:
                print(f"   ❌ Prediction failed even with 98 features: {e}")
    
    return False

if __name__ == "__main__":
    success = investigate_feature_mismatch()
    if success:
        print(f"\n🔧 SOLUTION: Modify the app to use only the first 98 dimensions of Word2Vec vectors")
    else:
        print(f"\n❌ Further investigation needed")