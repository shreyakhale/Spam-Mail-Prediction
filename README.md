# SMS Spam Detection Project

This project focuses on detecting spam SMS messages using machine learning techniques. The dataset used for this project contains SMS messages labeled as spam or ham (non-spam).

## Project Overview

The project involves the following key steps:

1. **Data Collection**: The dataset `mail_data.csv` contains SMS messages labeled as spam or ham.

2. **Data Preprocessing**: The `Spam_Mail_Prediction_using_ML.ipynb` notebook includes data cleaning and preprocessing steps to prepare the data for modeling.

3. **Feature Extraction**: The notebook uses the TF-IDF vectorization technique to convert text data into numerical features that can be used by machine learning models.

4. **Model Training**: A logistic regression model is trained on the preprocessed data to classify messages as spam or ham.

5. **Model Evaluation**: The trained model is evaluated using accuracy metrics on both the training and test datasets.

6. **Building a Predictive System**: The project includes a script `predictive_system.py` that demonstrates how the trained model can be used to classify new SMS messages.

## Technology and Algorithms Used

- **Python**: The project is implemented using Python programming language.
- **Machine Learning**: Logistic regression is used as the classification algorithm for this project.
- **Scikit-learn**: The `TfidfVectorizer` class from scikit-learn is used for feature extraction, and the `LogisticRegression` class is used for model training.
- **Pandas**: Pandas is used for data manipulation and analysis.
- **Jupyter Notebook**: Jupyter notebooks are used for interactive data analysis and model development.

## Usage

1. Clone the repository:
https://github.com/shreyakhale/Spam-Mail-Prediction.git
```bash
git clone https://github.com/shreyakhale/Spam-Mail-Prediction.git
```

2. Open and run the `Spam_Mail_Prediction_using_ML.ipynb` notebook in Jupyter to see the project implementation and results.

## Results and Conclusion

- The logistic regression model achieved an accuracy of 96.70% on the training data and 96.59% on the test data, indicating its effectiveness in classifying spam and ham messages.
- The project demonstrates how machine learning can be used to effectively classify SMS messages and highlights the importance of data preprocessing and feature extraction in text classification tasks.

## Future Improvements

- Experiment with other machine learning algorithms and hyperparameters to improve model performance.
- Explore more advanced text processing techniques for better feature extraction.
- Enhance the predictive system with a user-friendly interface.
