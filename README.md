**Fake News Classifier**

**Overview**
The Fake News Classifier is a machine learning-based solution designed to identify whether a given news article is fake or real based on its textual content. This repository includes the implementation, preprocessing pipeline, and a high-performing classifier using the XGBoost algorithm, along with other baseline models for comparison.

**Features**

*High Accuracy*: Achieves 99% testing accuracy with the XGBoost model.
*Text Preprocessing*: Robust pipeline for cleaning and vectorizing text.
*Batch Classification*: Classify multiple news articles at once and export results.
*Single Input Prediction*: Interactive function for testing individual news articles.
*Evaluation Metrics*: Includes AUC-ROC and classification reports.

**Dataset**

The dataset used for training and testing consists of:

*Columns*: title, text, subject, date, label (0 = fake, 1 = real)
*Preprocessing Steps*:
*Dropped columns*: title, subject, date.
*Text cleaning*:

-Convert to lowercase.
-Remove non-word characters, punctuation, numbers, HTML tags, URLs, and square bracket content.
Transformed cleaned text into feature vectors using TF-IDF vectorization.

**Models and Performance**

Model	                   Validation Accuracy	Testing Accuracy	F1-Score (Overall)
Logistic Regression	             98%	              98%            	0.98
Decision Tree Classifier	       94%	              94%	            0.94
Gradient Boosting Classifier	   97%	              97%	            0.97
Random Forest Classifier	       97%	              97%	            0.97
XGBoost Classifier	             99%	              99%           	0.99

**AUC-ROC Metrics**

-Training/Validation AUC-ROC: 0.99
-Testing AUC-ROC: 0.99
*Insights*:

The XGBoost Classifier consistently outperforms all other models with superior accuracy and AUC scores.

Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/your_username/Fake-News-Classifier.git
cd Fake-News-Classifier
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Prepare the Dataset:

Place the dataset files (train.tsv and test.tsv) in the data/ directory.
Usage
Training the Model
Run the script to preprocess the data and train the models:

bash
Copy code
python train.py
Predicting with the Classifier
Single Input Prediction
Use the testing() function in predict.py to classify a single news article:

python
Copy code
from predict import testing
result = testing("Nepal bombed Nagasaki in 1947.")
print("Prediction:", result)  # Output: Fake News
Batch Prediction
Run batch classification with the classify_and_save() function:

bash
Copy code
python batch_predict.py --input data/news.json --output results.json
Outputs
For single predictions: Label ("Fake News" or "Real News").
For batch predictions: Results saved in JSON format:
json
Copy code
{
  "results": [
    ["News text 1", "1"],
    ["News text 2", "0"]
  ]
}
Files and Directories
data/: Directory for storing the dataset files.
train.py: Script for training and evaluating models.
predict.py: Functions for single-input prediction.
batch_predict.py: Script for batch classification.
requirements.txt: Python dependencies.
Future Improvements
Dataset Expansion: Include more diverse and larger datasets.
Feature Engineering: Add metadata features like subject and date.
Deep Learning: Explore transformer-based models (e.g., BERT, RoBERTa) for improved performance.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
The dataset used in this project was obtained from publicly available sources.
Libraries used include scikit-learn, pandas, numpy, and xgboost.
For questions or contributions, feel free to open an issue or submit a pull request. 🚀













