"""
Veritas - Tweet Truth in Disaster Communication
A Streamlit application for classifying tweets as real disasters or not using ML models.

Competition: Natural Language Processing with Disaster Tweets (Kaggle)
"""

import streamlit as st
import joblib
import numpy as np
import re
import pandas as pd
import os
from pathlib import Path

# Configuration and Setup
@st.cache_data
def load_data_files():
    """Load and cache the training data for one-hot encoding recreation."""
    data_dir = Path("nlp-getting-started")
    
    try:
        # Try to load the saved feature configuration first
        try:
            feature_config = joblib.load('feature_config.pkl')
            encoded_cols = feature_config['categorical_columns']
            st.success(f"✅ Loaded training feature configuration: {len(encoded_cols)} categorical features")
        except FileNotFoundError:
            # Fallback to regenerating from data files
            st.warning("⚠️ Feature configuration file not found. Regenerating from data files...")
            train_df = pd.read_csv(data_dir / "train.csv")
            test_df = pd.read_csv(data_dir / "test.csv")
            
            # Fill missing values with 'unknown' as done in the notebook
            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df['keyword'] = train_df['keyword'].fillna('unknown')
            train_df['location'] = train_df['location'].fillna('unknown')
            test_df['keyword'] = test_df['keyword'].fillna('unknown')
            test_df['location'] = test_df['location'].fillna('unknown')
            
            # Create combined dataframe for consistent one-hot encoding
            combined_df = pd.concat([train_df.drop('target', axis=1), test_df], ignore_index=True)
            encoded_cols = pd.get_dummies(combined_df[['keyword', 'location']]).columns.tolist()
        
        # Get data sizes
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        
        return encoded_cols, len(train_df), len(test_df)
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        return None, 0, 0

@st.cache_resource
def load_models():
    """Load and cache the trained ML models."""
    try:
        model_combined = joblib.load('logistic_regression_model.pkl')
        word2vec_model = joblib.load('word2vec_model.pkl')
        return model_combined, word2vec_model
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None
# Text Processing Functions
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

# Function to average word vectors for a tweet (must be the same as used in the notebook)
def tweet_vector(text, model):
    """Convert text to vector representation using Word2Vec model."""
    if not text or not isinstance(text, str):
        return np.zeros(model.vector_size)
    
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        # Return a vector of zeros with the correct dimension if no words are in the vocabulary
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def prepare_features(text, word2vec_model, encoded_columns, keyword='unknown', location='unknown'):
    """Prepare features for prediction by combining embeddings and one-hot encoded features."""
    # Clean and vectorize text
    cleaned_text = clean_text(text)
    text_embedding = tweet_vector(cleaned_text, word2vec_model)
    
    # Create DataFrame for one-hot encoding
    temp_df = pd.DataFrame({'keyword': [keyword], 'location': [location]})
    temp_encoded = pd.get_dummies(temp_df[['keyword', 'location']])
    
    # Efficiently align columns with training data using reindex
    temp_encoded = temp_encoded.reindex(columns=encoded_columns, fill_value=0)
    
    # Debug: Print feature dimensions
    text_features = text_embedding.shape[0] if text_embedding.ndim > 0 else 0
    categorical_features = temp_encoded.shape[1] if len(temp_encoded.shape) > 1 else temp_encoded.shape[0]
    total_features = text_features + categorical_features
    
    # Combine features
    features = np.hstack((text_embedding.reshape(1, -1), temp_encoded))
    
    # Final debug check
    final_feature_count = features.shape[1]
    if final_feature_count != 4842:  # Expected by model
        st.error(f"⚠️ Feature dimension mismatch: Generated {final_feature_count} features, expected 4842")
        st.error(f"Text embedding: {text_features}, Categorical: {categorical_features}")
    
    return features, cleaned_text

def predict_tweet(text, model, word2vec_model, encoded_columns, keyword='unknown', location='unknown'):
    """Make prediction for a single tweet."""
    try:
        features, cleaned_text = prepare_features(text, word2vec_model, encoded_columns, keyword, location)
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        return {
            'prediction': prediction,
            'confidence': max(prediction_proba),
            'cleaned_text': cleaned_text,
            'probabilities': {
                'not_disaster': prediction_proba[0],
                'disaster': prediction_proba[1]
            }
        }
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Main Application
def main():
    # Page configuration
    st.set_page_config(
        page_title="Veritas - Disaster Tweet Classifier",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🚨 Veritas - Tweet Truth in Disaster Communication")
    st.markdown("""
    **Classify tweets to determine if they indicate real disasters or not.**
    
    This application uses machine learning models trained on the Kaggle "Natural Language Processing with Disaster Tweets" 
    competition dataset to classify tweets as either indicating a real disaster or not.
    """)
    
    # Load models and data
    with st.spinner("Loading models and data..."):
        model_combined, word2vec_model = load_models()
        encoded_columns, train_size, test_size = load_data_files()
    
    if model_combined is None or word2vec_model is None or encoded_columns is None:
        st.error("Failed to load required models or data. Please check if all files are present.")
        return
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Training Data Size:** {train_size:,} tweets")
        st.write(f"**Test Data Size:** {test_size:,} tweets")
        st.write(f"**Model Type:** Logistic Regression")
        st.write(f"**Word Embedding:** Word2Vec")
        st.write(f"**Vector Size:** {word2vec_model.vector_size}")
        st.write(f"**Vocabulary Size:** {len(word2vec_model.wv):,} words")
        
        st.markdown("---")
        st.markdown("**About the Competition:**")
        st.markdown("""
        Twitter has become an important communication channel in times of emergency. 
        This app helps distinguish between tweets that announce real disasters versus 
        those that use disaster-related words metaphorically.
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Single Tweet", "📂 Batch Processing", "ℹ️ Examples"])
    
    with tab1:
        single_tweet_interface(model_combined, word2vec_model, encoded_columns)
    
    with tab2:
        batch_processing_interface(model_combined, word2vec_model, encoded_columns)
    
    with tab3:
        examples_interface()

def single_tweet_interface(model_combined, word2vec_model, encoded_columns):
    """Interface for classifying a single tweet."""
    st.header("Classify a Single Tweet")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        tweet_input = st.text_area(
            "Enter a tweet to classify:",
            placeholder="e.g., 'Forest fire near my house, evacuating now!'",
            height=100
        )
    
    with col2:
        st.markdown("**Optional Context:**")
        keyword_input = st.text_input("Keyword", value="unknown", help="Related keyword (if any)")
        location_input = st.text_input("Location", value="unknown", help="Location mentioned (if any)")
    
    if st.button("🔍 Classify Tweet", type="primary"):
        if tweet_input.strip():
            with st.spinner("Analyzing tweet..."):
                result = predict_tweet(
                    tweet_input, 
                    model_combined, 
                    word2vec_model, 
                    encoded_columns,
                    keyword_input,
                    location_input
                )
            
            if result:
                display_prediction_result(tweet_input, result)
        else:
            st.warning("⚠️ Please enter a tweet to classify.")

def batch_processing_interface(model_combined, word2vec_model, encoded_columns):
    """Interface for batch processing multiple tweets."""
    st.header("Batch Processing")
    st.markdown("Upload a CSV file containing tweets for batch classification.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="CSV file must contain a 'text' column with tweets. Optional 'keyword' and 'location' columns are supported."
    )
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            if 'text' not in df_upload.columns:
                st.error("❌ CSV file must contain a 'text' column.")
                return
            
            st.success(f"✅ File uploaded successfully! Found {len(df_upload)} tweets.")
            
            # Show preview
            with st.expander("📋 Preview Data"):
                st.dataframe(df_upload.head())
            
            if st.button("🚀 Process All Tweets", type="primary"):
                process_batch_tweets(df_upload, model_combined, word2vec_model, encoded_columns)
                
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

def process_batch_tweets(df, model_combined, word2vec_model, encoded_columns):
    """Process multiple tweets and display results."""
    with st.spinner("Processing tweets..."):
        progress_bar = st.progress(0)
        results = []
        
        for idx, row in df.iterrows():
            # Update progress
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            
            # Get keyword and location if available
            keyword = row.get('keyword', 'unknown')
            location = row.get('location', 'unknown')
            
            # Handle NaN values
            if pd.isna(keyword):
                keyword = 'unknown'
            if pd.isna(location):
                location = 'unknown'
            
            # Make prediction
            result = predict_tweet(
                row['text'], 
                model_combined, 
                word2vec_model, 
                encoded_columns,
                keyword,
                location
            )
            
            if result:
                results.append({
                    'text': row['text'],
                    'prediction': 'Disaster' if result['prediction'] == 1 else 'Not Disaster',
                    'confidence': result['confidence'],
                    'disaster_probability': result['probabilities']['disaster'],
                    'cleaned_text': result['cleaned_text']
                })
        
        progress_bar.empty()
    
    # Display results
    if results:
        results_df = pd.DataFrame(results)
        
        st.success(f"✅ Processed {len(results)} tweets successfully!")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        disaster_count = len(results_df[results_df['prediction'] == 'Disaster'])
        
        with col1:
            st.metric("Total Tweets", len(results_df))
        with col2:
            st.metric("Disaster Tweets", disaster_count)
        with col3:
            st.metric("Non-Disaster Tweets", len(results_df) - disaster_count)
        
        # Results table
        st.dataframe(
            results_df,
            use_container_width=True,
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    help="Model confidence (0-1)",
                    min_value=0,
                    max_value=1,
                ),
                "disaster_probability": st.column_config.ProgressColumn(
                    "Disaster Prob.",
                    help="Probability of being a disaster tweet",
                    min_value=0,
                    max_value=1,
                ),
            }
        )
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="disaster_tweet_predictions.csv",
            mime="text/csv"
        )

def display_prediction_result(original_text, result):
    """Display the prediction result for a single tweet."""
    prediction = result['prediction']
    confidence = result['confidence']
    probabilities = result['probabilities']
    cleaned_text = result['cleaned_text']
    
    # Main prediction display
    if prediction == 1:
        st.error(f"🚨 **DISASTER TWEET DETECTED**")
        st.error(f"Confidence: {confidence:.1%}")
    else:
        st.success(f"✅ **NOT A DISASTER TWEET**")
        st.success(f"Confidence: {confidence:.1%}")
    
    # Detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Prediction Probabilities:**")
        st.write(f"• Not Disaster: {probabilities['not_disaster']:.1%}")
        st.write(f"• Disaster: {probabilities['disaster']:.1%}")
    
    with col2:
        st.markdown("**🧹 Text Processing:**")
        with st.expander("See cleaned text"):
            st.write(f"**Original:** {original_text}")
            st.write(f"**Cleaned:** {cleaned_text}")

def examples_interface():
    """Display example tweets and explanations."""
    st.header("📚 Example Classifications")
    
    examples = [
        {
            'text': "Forest fire near La Ronge Sask. Canada",
            'expected': "Disaster",
            'explanation': "Direct report of an actual forest fire with specific location."
        },
        {
            'text': "What's up man?",
            'expected': "Not Disaster", 
            'explanation': "Casual conversation with no disaster-related content."
        },
        {
            'text': "The new iPhone is on fire! Best phone ever!",
            'expected': "Not Disaster",
            'explanation': "Uses 'fire' metaphorically to express enthusiasm, not literal fire."
        },
        {
            'text': "Emergency evacuation happening now in the building across the street",
            'expected': "Disaster",
            'explanation': "Reports ongoing emergency evacuation, indicating real danger."
        },
        {
            'text': "This movie is the bomb!",
            'expected': "Not Disaster",
            'explanation': "Uses 'bomb' metaphorically to express quality, not literal explosive."
        }
    ]
    
    st.markdown("""
    Here are some example tweets that demonstrate the difference between real disaster reports 
    and metaphorical usage of disaster-related words:
    """)
    
    for i, example in enumerate(examples, 1):
        with st.expander(f"Example {i}: {example['expected']}"):
            st.write(f"**Tweet:** {example['text']}")
            st.write(f"**Expected Classification:** {example['expected']}")
            st.write(f"**Explanation:** {example['explanation']}")

if __name__ == "__main__":
    main()