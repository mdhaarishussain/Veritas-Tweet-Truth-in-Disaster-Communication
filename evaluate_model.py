#!/usr/bin/env python3
"""
Model Evaluation Script for Veritas Disaster Tweet Classifier

This script evaluates the trained model and generates real performance metrics
to replace placeholder values in documentation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import re
from pathlib import Path

def clean_text(text):
    """Clean tweet text by removing URLs, mentions, hashtags, and special characters."""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove special characters and keep only alphanumeric and spaces
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()
    return text

def tweet_vector(text, model):
    """Convert text to vector representation using Word2Vec model."""
    if not text or not isinstance(text, str):
        return np.zeros(model.vector_size)
    
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def prepare_features(df, word2vec_model, encoded_columns):
    """Prepare features exactly as done during training."""
    features_list = []
    
    for idx, row in df.iterrows():
        # Clean and vectorize text
        cleaned_text = clean_text(row['text'])
        text_embedding = tweet_vector(cleaned_text, word2vec_model)
        
        # Create DataFrame for one-hot encoding (single row)
        temp_df = pd.DataFrame({
            'keyword': [row.get('keyword', 'unknown')], 
            'location': [row.get('location', 'unknown')]
        })
        temp_encoded = pd.get_dummies(temp_df[['keyword', 'location']])
        
        # Align columns with training data
        temp_encoded = temp_encoded.reindex(columns=encoded_columns, fill_value=0)
        
        # Combine features
        combined_features = np.hstack((text_embedding.reshape(1, -1), temp_encoded.values))
        features_list.append(combined_features[0])
    
    return np.array(features_list)

def load_models_and_config():
    """Load trained models and feature configuration."""
    try:
        # Load models
        model_combined = joblib.load('logistic_regression_model.pkl')
        word2vec_model = joblib.load('word2vec_model.pkl')
        
        # Load feature configuration
        try:
            feature_config = joblib.load('feature_config.pkl')
            encoded_columns = feature_config['categorical_columns']
            print(f"✅ Loaded feature configuration: {len(encoded_columns)} categorical features")
        except FileNotFoundError:
            print("⚠️ Feature config not found, generating from training data...")
            # Fallback: recreate from training data
            train_df = pd.read_csv('nlp-getting-started/train.csv')
            test_df = pd.read_csv('nlp-getting-started/test.csv')
            
            # Fill missing values
            train_df['keyword'] = train_df['keyword'].fillna('unknown')
            train_df['location'] = train_df['location'].fillna('unknown')
            test_df['keyword'] = test_df['keyword'].fillna('unknown')
            test_df['location'] = test_df['location'].fillna('unknown')
            
            # Create combined dataframe for consistent encoding
            combined_df = pd.concat([train_df.drop('target', axis=1), test_df], ignore_index=True)
            encoded_columns = pd.get_dummies(combined_df[['keyword', 'location']]).columns.tolist()
        
        return model_combined, word2vec_model, encoded_columns
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None, None

def evaluate_model():
    """Comprehensive model evaluation."""
    print("🔍 Starting Model Evaluation...")
    print("=" * 50)
    
    # Load models and configuration
    model, word2vec_model, encoded_columns = load_models_and_config()
    if model is None:
        print("❌ Failed to load models. Exiting.")
        return None
    
    # Load training data
    try:
        train_df = pd.read_csv('nlp-getting-started/train.csv')
        print(f"✅ Loaded training data: {len(train_df)} samples")
        
        # Fill missing values
        train_df['keyword'] = train_df['keyword'].fillna('unknown')
        train_df['location'] = train_df['location'].fillna('unknown')
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None
    
    # Display model information
    print(f"\n📊 Model Information:")
    print(f"   Expected features: {model.n_features_in_}")
    print(f"   Word2Vec dimensions: {word2vec_model.vector_size}")
    print(f"   Vocabulary size: {len(word2vec_model.wv)}")
    print(f"   Categorical features: {len(encoded_columns)}")
    
    # Prepare features
    print(f"\n🔧 Preparing features...")
    X = prepare_features(train_df, word2vec_model, encoded_columns)
    y = train_df['target'].values
    
    print(f"   Generated feature shape: {X.shape}")
    print(f"   Target distribution: {np.bincount(y)} (0=non-disaster, 1=disaster)")
    
    # Check feature alignment
    if X.shape[1] != model.n_features_in_:
        print(f"⚠️ Feature dimension mismatch!")
        print(f"   Generated: {X.shape[1]}, Expected: {model.n_features_in_}")
        print(f"   Difference: {X.shape[1] - model.n_features_in_}")
        
        # Handle mismatch
        if X.shape[1] > model.n_features_in_:
            X = X[:, :model.n_features_in_]
            print(f"   ✂️ Trimmed to {X.shape[1]} features")
        else:
            padding = np.zeros((X.shape[0], model.n_features_in_ - X.shape[1]))
            X = np.hstack([X, padding])
            print(f"   📝 Padded to {X.shape[1]} features")
    else:
        print(f"✅ Feature dimensions match perfectly!")
    
    # Make predictions
    print(f"\n🔮 Making predictions...")
    try:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        print(f"✅ Predictions completed successfully")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Class distribution
    disaster_count = np.sum(y)
    non_disaster_count = len(y) - disaster_count
    disaster_pct = disaster_count / len(y) * 100
    non_disaster_pct = non_disaster_count / len(y) * 100
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_samples': len(y),
        'disaster_samples': disaster_count,
        'non_disaster_samples': non_disaster_count,
        'confusion_matrix': cm,
        'classification_report': classification_report(y, y_pred),
        'model_info': {
            'expected_features': model.n_features_in_,
            'word2vec_size': word2vec_model.vector_size,
            'vocabulary_size': len(word2vec_model.wv),
            'categorical_features': len(encoded_columns)
        }
    }
    
    # Display comprehensive results
    print("\n" + "=" * 60)
    print("📊 REAL MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"   Recall:    {recall:.3f} ({recall*100:.1f}%)")
    print(f"   F1-Score:  {f1:.3f} ({f1*100:.1f}%)")
    
    print(f"\n📊 Dataset Information:")
    print(f"   Total Samples: {len(y):,}")
    print(f"   Disaster Tweets: {disaster_count:,} ({disaster_pct:.1f}%)")
    print(f"   Non-Disaster Tweets: {non_disaster_count:,} ({non_disaster_pct:.1f}%)")
    
    print(f"\n🔢 Confusion Matrix:")
    print(f"                     Predicted")
    print(f"                Non-Disaster  Disaster")
    print(f"   True Non-Disaster:   {tn:4d}     {fp:4d}")
    print(f"   True Disaster:       {fn:4d}     {tp:4d}")
    
    print(f"\n📋 Detailed Classification Report:")
    print(results['classification_report'])
    
    print(f"\n🔧 Technical Details:")
    print(f"   Model Features: {model.n_features_in_:,}")
    print(f"   Word2Vec Dimensions: {word2vec_model.vector_size}")
    print(f"   Vocabulary Size: {len(word2vec_model.wv):,} words")
    print(f"   Categorical Features: {len(encoded_columns)}")
    
    # Model performance interpretation
    print(f"\n💡 Performance Interpretation:")
    if precision > 0.85:
        print(f"   ✅ High Precision ({precision:.1%}): Few false disaster alerts")
    elif precision > 0.70:
        print(f"   ⚠️ Moderate Precision ({precision:.1%}): Some false disaster alerts")
    else:
        print(f"   ❌ Low Precision ({precision:.1%}): Many false disaster alerts")
    
    if recall > 0.85:
        print(f"   ✅ High Recall ({recall:.1%}): Catches most real disasters")
    elif recall > 0.70:
        print(f"   ⚠️ Moderate Recall ({recall:.1%}): Misses some real disasters")
    else:
        print(f"   ❌ Low Recall ({recall:.1%}): Misses many real disasters")
    
    if f1 > 0.80:
        print(f"   ✅ Strong Overall Performance (F1: {f1:.1%})")
    elif f1 > 0.70:
        print(f"   ⚠️ Good Performance (F1: {f1:.1%})")
    else:
        print(f"   ❌ Needs Improvement (F1: {f1:.1%})")
    
    print(f"\n" + "=" * 60)
    print("✅ Evaluation completed successfully!")
    print("📝 Use these metrics to update your README.md")
    print("=" * 60)
    
    return results

def generate_readme_metrics(results):
    """Generate formatted metrics for README update."""
    if not results:
        return None
    
    readme_text = f"""
### Model Performance (Evaluated on Training Data)
- **Accuracy**: {results['accuracy']:.1%}
- **Precision**: {results['precision']:.1%} 
- **Recall**: {results['recall']:.1%}
- **F1-Score**: {results['f1_score']:.1%}
- **Total Samples**: {results['total_samples']:,}
- **Class Distribution**: {results['disaster_samples']:,} disaster ({results['disaster_samples']/results['total_samples']:.1%}), {results['non_disaster_samples']:,} non-disaster ({results['non_disaster_samples']/results['total_samples']:.1%})

### Technical Specifications
- **Model Features**: {results['model_info']['expected_features']:,} total dimensions
- **Word2Vec**: {results['model_info']['word2vec_size']} dimensions, {results['model_info']['vocabulary_size']:,} vocabulary
- **Categorical Features**: {results['model_info']['categorical_features']} encoded features
"""
    
    print("\n📋 README Metrics Section:")
    print(readme_text)
    return readme_text

if __name__ == "__main__":
    print("🚨 Veritas Model Evaluation")
    print("Generating real performance metrics...\n")
    
    results = evaluate_model()
    
    if results:
        generate_readme_metrics(results)
        print(f"\n💡 Next steps:")
        print(f"   1. Fix any feature alignment issues if reported above")
        print(f"   2. Update README.md with these real metrics")
        print(f"   3. Consider cross-validation for more robust evaluation")
    else:
        print(f"\n❌ Evaluation failed. Check your model files and data.")