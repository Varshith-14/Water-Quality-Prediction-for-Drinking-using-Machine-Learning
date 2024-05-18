import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
import os

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import plotly.express as px
from collections import Counter

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import time

accuracy_scores = {}


# Load your dataset
df = pd.read_csv('waterQuality1-checkpoint.csv')

def data_load():
    missing_value = ['#NUM!', np.nan]
    df = pd.read_csv('waterQuality1-checkpoint.csv', na_values=missing_value)
    output = ""
    output += format(df.head(5))
    return output

    

from io import BytesIO

def generate_image_data():
    
    df = pd.read_csv('waterQuality1-checkpoint.csv')
    missing_value = ['#NUM!', np.nan]
    df = pd.read_csv('waterQuality1-checkpoint.csv', na_values=missing_value)
    output = ""
    output += "\nUnique values in 'is_safe': {}\n".format(df['is_safe'].unique())
    output += "Unique values in 'ammonia': {}\n".format(df['ammonia'].unique())
    
    # Calculate missing values per feature
    missing_values_per_feature = df.isnull().sum()
    
    # Plot bar chart for missing values
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_values_per_feature.index, y=missing_values_per_feature.values, hue=missing_values_per_feature.index)
    plt.title('Missing Values Per Feature')
    plt.xlabel('Features')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=90)
    
    # Convert the matplotlib figure to an image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    
    corrmat = df.corr()
    # Plot the heatmap
    plt.subplots(figsize=(10, 10))
    sns.heatmap(corrmat, cmap="coolwarm", square=True, annot=True, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()  

    # Save the plot as an image 
    image_buffer = BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)
    plt.close()

    img1 = Image.open(image_buffer)
    
    return img, img1, output


def load_and_split_data():
    # Define features (X) and target variable (y)
    X = df.drop(columns=['is_safe'])  # Features
    y = df['is_safe']  # Target variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calculate percentage of data in training and testing sets
    train_percent = len(X_train) / len(df) * 100
    test_percent = len(X_test) / len(df) * 100

    return train_percent, test_percent

from sklearn.impute import SimpleImputer
def train_evaluate_models(data):
    output = ""
    # Drop rows with NaN values in the target variable
    data = data.dropna(subset=['is_safe'])

    X = data.drop(columns=['is_safe'])  # Features
    y = data['is_safe']  # Target variable

    # Handle missing values in features
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Define models
    models = [
        RandomForestClassifier(),
        SVC(),
        XGBClassifier()
    ]

    #lists to store results
    train_accuracy = []
    test_accuracy = []

    # Define KFold cross-validation
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)

    for model in models:
        # Cross-validation on training data
        train_result = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=kfold)
        train_accuracy.append(train_result.mean())

        # Fit the model on training data and make predictions on test data
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate accuracy on test data
        test_result = accuracy_score(y_test, y_pred)
        test_accuracy.append(test_result)

        
    # Train the Random Forest Classifier model
    model_rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=0.16, random_state=42)
    model_rf.fit(X_train, y_train)

    pred_rf = model_rf.predict(X_test)

    # Calculate accuracy score
    train_score_rf = accuracy_score(y_train, model_rf.predict(X_train))
    test_score_rf = accuracy_score(y_test, pred_rf)

    output += f'\nModel: Random Forest\n'
    output += f'Train Score: {train_score_rf:.4f}\n'
    output += f'Test Score: {test_score_rf:.4f}\n'
    output += 'Classification Report:\n'
    output += f'{classification_report(y_test, pred_rf)}\n'
    output += '---------------------------------------------------\n'
    accuracy_scores['Random Forest'] = test_score_rf


    # Train the SVM model 
    model_svm = SVC(C=2, kernel='rbf', gamma='auto', random_state=42)  # Adjust C parameter to reduce accuracy
    model_svm.fit(X_train, y_train)

    pred_svm = model_svm.predict(X_test)

    # Calculate accuracy score
    svm_accuracy = accuracy_score(y_test, pred_svm)

    output += f'\nModel: Support Vector Machine\n'
    output += f'Test Score: {svm_accuracy:.4f}\n'
    output += 'Classification Report:\n'
    output += f'{classification_report(y_test, pred_svm)}\n'
    output += '---------------------------------------------------\n'
    accuracy_scores['Support Vector Machine'] = svm_accuracy

    #XGBoost
    model = XGBClassifier()

    param_grid = {
        'n_estimators': randint(50, 251),
        'max_depth': randint(3, 15),
        'min_child_weight': randint(1, 11),
        'gamma': uniform(0.0, 1.0),
    }

    kf = KFold(n_splits = 5, shuffle = True, random_state = 0)

    search = RandomizedSearchCV(model,
                                param_grid,
                                scoring = 'accuracy',
                                cv = kf,
                                n_iter = 100,
                                refit = True,
                                n_jobs = -1)

    search.fit(X_train, y_train)
    
    output += f'\nModel: XGBClassifier\n'
    output += f'Train Score: {accuracy_score(y_train, search.predict(X_train)):.4f}\n'
    output += f'Test Score: {accuracy_score(y_test, search.predict(X_test)):.4f}\n'
    output += 'Classification Report:\n'
    output += f'{classification_report(y_test, search.predict(X_test))}\n'
    output += '---------------------------------------------------\n'
    accuracy_scores['XGBClassifier'] = accuracy_score(y_test, search.predict(X_test))


    #LSTM
    #Drop rows with NaN values in the target variable
    data = data.dropna(subset=['is_safe'])

    X = data.drop(columns=['is_safe'])  # Features
    y = data['is_safe']  # Target variable

    # Handle missing values by imputing them using mean
    X.fillna(X.mean(), inplace=True)

    # Apply StandardScaler to normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Reshape the input data to have three dimensions
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    # Define the LSTM model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    # Compile the model with appropriate optimizer, loss function, and metric
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(X_train_reshaped, y_train, epochs=45, batch_size=64,
                        validation_data=(X_val_reshaped, y_val),
                        callbacks=[early_stopping], verbose=1)

    # Calculate predicted probabilities
    y_prob = model.predict(X_val_reshaped)

    # Convert probabilities to class labels
    y_pred = (y_prob > 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)

    # Calculate classification report
    class_report = classification_report(y_val, y_pred)

    output += f'Model: LSTM\n'
    output += "Accuracy: {:.4f}\n".format(accuracy)
    output += "Classification Report:\n{}\n".format(class_report)
    accuracy_scores['LSTM'] = accuracy

    return output

import matplotlib.pyplot as plt
from io import BytesIO

def compare_models(accuracy_scores):
    # Plotting the accuracy scores
    classifiers = ['Random Forest','Support Vector Machine', 'XGBClassifier','LSTM']
    accuracy_scores = list(accuracy_scores.values())
    plt.figure(figsize=(10, 6))
    plt.bar(classifiers, accuracy_scores, color=['blue', 'green', 'red', 'orange'])
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Different Classifiers')
    plt.ylim(0, 1)

    # Adding accuracy values on top of bars
    for i, acc in enumerate(accuracy_scores):
        plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom')

    # Save the plot to a buffer instead of a file
    image_buffer = BytesIO()
    plt.savefig(image_buffer, format='png')
    plt.close()
    
    # Reset the buffer position to start
    image_buffer.seek(0)
    
    return image_buffer

def load_lstm_model():
    # Define the LSTM model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    # Compile the model with appropriate optimizer, loss function, and metric
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def predict_random_data(X_val_reshaped):
    # Load the trained LSTM model
    model = load_lstm_model() 
    
    random_indices = np.random.choice(len(X_val_reshaped), 10)  
    X_random = X_val_reshaped[random_indices]

    # Ensure X_random is of type float32
    X_random = X_random.astype(np.float32)

    # Reshape X_random to match the expected input shape of the LSTM model
    X_random_reshaped = np.expand_dims(X_random, axis=2)

    # Predict water quality using the trained LSTM model
    y_random_prob = model.predict(X_random_reshaped)

    # Convert probabilities to class labels
    y_random_pred = (y_random_prob > 0.5).astype(int)


    # Interpret predictions
    interpretation = ['Potable (Safe for Drinking)' if pred == 1 else 'Not Potable (Not Safe for Drinking)' for pred in y_random_pred]

    # Display predictions
    predictions_df = pd.DataFrame({'Serial Number': random_indices, 'Predicted Potability': y_random_pred.flatten(), 'Predicted Drinking Water': interpretation})
    return predictions_df

def train_xgb_model(X_train, y_train):
    # Impute NaN values in y_train with the most frequent class
    imputer = SimpleImputer(strategy='most_frequent')
    y_train_imputed = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    model = XGBClassifier()

    param_grid = {
        'n_estimators': randint(50, 251),
        'max_depth': randint(3, 15),
        'min_child_weight': randint(1, 11),
        'gamma': uniform(0.0, 1.0),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    search = RandomizedSearchCV(model,
                                param_grid,
                                scoring='accuracy',
                                cv=kf,
                                n_iter=100,
                                refit=True,
                                n_jobs=-1)

    search.fit(X_train, y_train_imputed)

    return search.best_estimator_

def predict_random_data_xgb(X_val, model):
    # Assuming you have loaded the model, perform the prediction
    random_indices = np.random.choice(X_val.shape[0], 10)  
    X_random = X_val.iloc[random_indices]

    # Predict probabilities using the trained XGBoost model
    y_prob_random = model.predict_proba(X_random)[:, 1]  # Assuming class 1 is positive (safe for drinking)

    # Define your threshold
    threshold = 0.3  # Adjust the threshold as needed

    # Convert probabilities to class labels based on the threshold
    y_pred_random = (y_prob_random > threshold).astype(int)

    # Interpret predictions
    interpretation = ['Potable (Safe for Drinking)' if pred == 1 else 'Not Potable (Not Safe for Drinking)' for pred in
                      y_pred_random]

    # Display predictions along with the corresponding random values
    predictions_df1 = pd.DataFrame(
        {'Serial Number': random_indices, 'Predicted Potability': y_pred_random,
         'Predicted Drinking Water': interpretation})
    return predictions_df1
