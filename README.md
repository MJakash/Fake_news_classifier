## **Fake News Classifier**

### **Overview**
The Fake News Classifier is a machine learning-based solution designed to identify whether a given news article is fake or real based on its textual content. This repository includes the implementation, preprocessing pipeline, and a high-performing classifier using the XGBoost algorithm, along with other baseline models for comparison.

### **Features**

- *High Accuracy*: Achieves 99% testing accuracy with the XGBoost model.
- *Text Preprocessing*: Robust pipeline for cleaning and vectorizing text.
- *Batch Classification*: Classify multiple news articles at once and export results.
- *Single Input Prediction*: Interactive function for testing individual news articles.
- *Evaluation Metrics*: Includes AUC-ROC and classification reports.

### **Dataset**

The dataset used for training and testing consists of:

- *Columns*: title, text, subject, date, label (0 = fake, 1 = real)
- *Preprocessing Steps*:
  - *Dropped columns*: title, subject, date.
  - *Text cleaning*:
    - Convert to lowercase.
    - Remove non-word characters, punctuation, numbers, HTML tags, URLs, and square bracket content./n
  - Transformed cleaned text into feature vectors using TF-IDF vectorization.

### **Models and Performance**

|Model	                     |Validation Accuracy|Testing Accuracy|F1-Score (Overall)|
|----------------------------|-------------------|----------------|------------------|
|Logistic Regression	       |      98%	         |     98%        |     	0.98       | 
|Decision Tree Classifier	   |      94%	         |     94%	      |       0.94       | 
|Gradient Boosting Classifier|	    97%	         |     97%	      |       0.97       |
|Random Forest Classifier	   |      97%	         |     97%	      |       0.97       | 
|XGBoost Classifier	         |      99%	         |     99%        |     	0.99       |

### **AUC-ROC Metrics**

- Training/Validation AUC-ROC: 0.99
- Testing AUC-ROC: 0.99
- *Insights*:

The XGBoost Classifier consistently outperforms all other models with superior accuracy and AUC scores.

### **Future Improvements**
- Dataset Expansion: Include more diverse and larger datasets.
- Feature Engineering: Add metadata features like subject and date.
- Deep Learning: Explore transformer-based models (e.g., BERT, RoBERTa) for improved performance.

### **Acknowledgements**
The dataset used in this project was obtained from publicly available sources.
Libraries used include scikit-learn, pandas, numpy, and xgboost.
For questions or contributions, feel free to open an issue or submit a pull request. 🚀













