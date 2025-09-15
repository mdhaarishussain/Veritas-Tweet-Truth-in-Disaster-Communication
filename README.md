# 🚨 Veritas - Tweet Truth in Disaster Communication

A machine learning application that classifies tweets to determine if they indicate real disasters or metaphorical usage of disaster-related terms.

## 🎯 Project Overview

This project was developed as part of the Shell AI/ML Internship program, tackling the Kaggle "Natural Language Processing with Disaster Tweets" competition. The system helps distinguish between tweets that announce real emergencies versus those using disaster-related words metaphorically.

### The Problem
Twitter has become crucial for emergency communication, but it's challenging to programmatically distinguish between:
- **Real disaster reports**: "Forest fire near La Ronge Sask. Canada"
- **Metaphorical usage**: "This new iPhone is on fire! Best phone ever!"

## 🏗️ Architecture

### Data Pipeline
- **Training Dataset**: 7,613 hand-classified tweets
- **Test Dataset**: 3,263 tweets
- **Features**: Tweet text, keywords, locations
- **Target**: Binary classification (disaster/not disaster)

### Model Components
1. **Text Preprocessing**: URL/mention/hashtag removal, lowercasing, special character handling
2. **Feature Engineering**: 
   - Word2Vec embeddings (100 dimensions, 3,470 vocabulary)
   - One-hot encoded categorical features (keyword + location)
   - Combined feature vector (4,842 dimensions total)
3. **Classification**: Logistic Regression with L-BFGS solver

### Model Performance
> **Note**: Run `python evaluate_model.py` to generate current performance metrics.
> 
> The model is trained and functional, but comprehensive evaluation metrics need to be generated to provide accurate performance figures.

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
```

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/mdhaarishussain/Veritas-Tweet-Truth-in-Disaster-Communication.git
cd Veritas-Tweet-Truth-in-Disaster-Communication

# Option 1: Use Virtual Environment (Recommended)
# Windows
run_app.bat

# Option 2: Manual Installation
pip install -r requirements.txt
streamlit run app.py
```

### Usage

#### Web Interface
1. Open http://localhost:8501 in your browser
2. Choose between single tweet classification or batch processing
3. Enter tweet text or upload a CSV file
4. View results with confidence scores and explanations

#### Model Evaluation
```bash
# Run comprehensive model evaluation
python evaluate_model.py

# This will generate:
# - Accuracy, Precision, Recall, F1-Score
# - Confusion Matrix
# - Classification Report
# - Feature alignment verification
```

## 📁 Project Structure

```
├── app.py                              # Enhanced Streamlit application
├── evaluate_model.py                   # Model evaluation script
├── Shell_AI_Internship (3).ipynb     # Complete ML development notebook
├── logistic_regression_model.pkl      # Trained classification model
├── word2vec_model.pkl                 # Trained word embeddings
├── feature_config.pkl                 # Training feature configuration
├── requirements.txt                   # Python dependencies
├── veritas_env/                       # Virtual environment
├── run_app.bat                        # Easy app launcher (Windows)
├── run_app.sh                         # Easy app launcher (Unix/macOS)
├── nlp-getting-started/              # Kaggle competition data
│   ├── train.csv                     # Training dataset
│   ├── test.csv                      # Test dataset
│   └── sample_submission.csv         # Submission format
└── README.md                         # This file
```

## 🛠️ Technical Details

### Data Preprocessing
- **Missing Value Handling**: Keywords and locations filled with 'unknown'
- **Text Cleaning**: Removes URLs, mentions (@user), hashtags (#tag), special characters
- **Tokenization**: Space-separated word splitting
- **Normalization**: Lowercase conversion

### Feature Engineering
- **Word2Vec Parameters**:
  - Vector size: 100 dimensions
  - Window: 5 words
  - Minimum count: 5 occurrences
  - Training algorithm: Skip-gram
  - Vocabulary: 3,470 unique words
- **Categorical Encoding**: One-hot encoding for keywords and locations
- **Feature Combination**: Horizontal stacking of embeddings and categorical features

### Model Training
- **Algorithm**: Logistic Regression
- **Solver**: L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
- **Max Iterations**: 1,000
- **Regularization**: L2 (Ridge)
- **Features**: 4,842 total dimensions

## 📊 Current Status

### ✅ Completed
- [x] Data preprocessing and cleaning pipeline
- [x] Word2Vec model training (100D embeddings)
- [x] Logistic Regression classifier training
- [x] Streamlit web application with full UI
- [x] Batch processing functionality
- [x] Virtual environment setup
- [x] Model serialization and loading

### 🔧 In Progress
- [ ] **Model Evaluation**: Comprehensive performance metrics generation
- [ ] **Feature Alignment**: Resolving dimension mismatch issues
- [ ] **Cross-Validation**: K-fold validation implementation

### Key Technical Notes
- **Feature Dimension**: Model expects exactly 4,842 features
- **Word2Vec Model**: Trained and functional with 3,470 vocabulary
- **App Status**: Functional but requires feature alignment fix
- **Data**: Complete Kaggle competition dataset available

## 📊 Results & Analysis

### Model Evaluation
Run `python evaluate_model.py` to generate comprehensive performance metrics including:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report  
- Feature alignment verification

### Dataset Characteristics
1. **Training Size**: 7,613 hand-classified tweets
2. **Test Size**: 3,263 tweets for final evaluation
3. **Class Distribution**: Balanced dataset with disaster/non-disaster examples
4. **Feature Space**: 4,842 total dimensions (100 text + 4,742 categorical)

## 🧪 Testing & Validation

To evaluate the current model performance:

```bash
# Activate virtual environment
.\veritas_env\Scripts\activate  # Windows
source veritas_env/bin/activate  # Unix/macOS

# Run evaluation
python evaluate_model.py

# Run app with debugging
streamlit run app.py
```

This will provide:
- Real accuracy, precision, recall, and F1-score
- Confusion matrix analysis
- Feature alignment verification
- Classification report with per-class metrics

## 🔮 Future Enhancements

### Model Improvements
- [ ] **Transformer Models**: Implement BERT/RoBERTa for better context understanding
- [ ] **Ensemble Methods**: Combine multiple model predictions
- [ ] **Cross-Validation**: Implement k-fold validation for robust evaluation
- [ ] **Hyperparameter Tuning**: Grid search for optimal parameters

### Application Features
- [ ] **Real-time Monitoring**: Twitter API integration for live tweet analysis
- [ ] **Geospatial Analysis**: Map visualization of disaster reports
- [ ] **Multi-language Support**: Extend to non-English tweets
- [ ] **Alert System**: Automated notifications for high-confidence disaster tweets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of the Shell AI/ML Internship program. Please respect data usage guidelines and competition rules.

## 🙏 Acknowledgments

- **Kaggle**: For the "Natural Language Processing with Disaster Tweets" competition
- **Shell AI/ML Internship Program**: For the learning opportunity
- **Open Source Libraries**: scikit-learn, gensim, streamlit, pandas, numpy

## 📞 Contact

**Md Haaris Hussain**
- GitHub: [@mdhaarishussain](https://github.com/mdhaarishussain)
- Project: [Veritas-Tweet-Truth-in-Disaster-Communication](https://github.com/mdhaarishussain/Veritas-Tweet-Truth-in-Disaster-Communication)

---

**⚠️ Disclaimer**: This model is for educational and research purposes. In real emergency situations, always rely on official sources and emergency services.
