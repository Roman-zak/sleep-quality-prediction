import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

def load_data():
  # Load the datasets (assuming they are saved as CSV)
  fitbit_sleep_df = pd.read_csv('resources/FitbitSleep.csv')
  fitbit_sleep_df = fitbit_sleep_df[
    ['egoid', 'datadate', 'timetobed', 'timeoutofbed', 'bedtimedur',
     'Efficiency']]
  fitbit_activity_df = pd.read_csv('resources/FitbitActivity(1-30-20).csv')
  basic_survey_df = pd.read_csv('resources/BasicSurvey(3-6-20).csv')
  basic_survey_df = basic_survey_df[['egoid',
                                     'gender_1', 'usebeer_1', 'usewine_1',
                                     'usecaffine_1', 'Happy_1']]
  df = pd.merge(fitbit_sleep_df, fitbit_activity_df,
                         on=['egoid', 'datadate'], how='left')

  df = pd.merge(df, basic_survey_df, on='egoid',
                         how='left').drop(columns=['egoid', 'datadate'])

  # Define mappings for each survey column
  frequency_mapping = {
    "Not at all": 0,
    "Less than 1-2 times a month": 1,
    "1-2 times a month": 2,
    "1-2 times a week": 3,
    "Three times a week or more": 4
  }

  happiness_mapping = {
    "not so happy": 1,
    "happy": 2,
    "pretty happy": 3,
    "very happy": 4
  }

  gender_mapping = {
    "Male": 0,
    "Female": 1
  }

  # Apply mappings to the DataFrame columns
  df['usebeer_1'] = df['usebeer_1'].map(frequency_mapping)
  df['usewine_1'] = df['usewine_1'].map(frequency_mapping)
  df['usecaffine_1'] = df['usecaffine_1'].map(
    frequency_mapping)
  df['Happy_1'] = df['Happy_1'].map(happiness_mapping)
  df['gender_1'] = df['gender_1'].map(gender_mapping)

  # Assuming 'timetobed' and 'timeoutofbed' are the columns with time data in combined_df
  df['timetobed'] = pd.to_datetime(
    df['timetobed']).dt.hour * 3600 + pd.to_datetime(
    df['timetobed']).dt.minute * 60 + pd.to_datetime(
    df['timetobed']).dt.second
  df['timeoutofbed'] = pd.to_datetime(
    df['timeoutofbed']).dt.hour * 3600 + pd.to_datetime(
    df['timeoutofbed']).dt.minute * 60 + pd.to_datetime(
    df['timeoutofbed']).dt.second

  # Define bins and labels for efficiency classes
  bins = [0, 0.6, 0.8, 0.9, 1]  # Define bin edges
  labels = [0, 1, 2, 3]  # Define class indices
  df['Efficiency'] = pd.cut(df['Efficiency'], bins=bins,
                                           labels=labels, include_lowest=True)
  # categorical_columns = df.select_dtypes(
  #   include=['category', 'object']).columns
  # print("Categorical columns:", categorical_columns)
  # # One-hot encode categorical columns
  # df = pd.get_dummies(df, columns=categorical_columns)
  # print(np.isnan(df).sum())
  # print(np.isinf(df).sum())

  return df

def print_hi(name):
  # Use a breakpoint in the code line below to debug your script.
  print(f'Hi, {name}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
  print_hi('PyCharm')
  combined_df = load_data()
  # Preprocess data
  # Extract features and labels (assuming 'sleep_quality' as the label)
  features = combined_df.drop(['Efficiency'], axis=1).values
  labels = combined_df['Efficiency'].values

  # Standardize features
  scaler = StandardScaler()
  features = scaler.fit_transform(features)

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                      test_size=0.2,
                                                      random_state=42)

  # # Build the MLP model
  # model = Sequential([
  #   Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
  #   # Input layer
  #   Dropout(0.3),  # Dropout layer to prevent overfitting
  #   Dense(64, activation='relu'),  # Hidden layer
  #   Dropout(0.2),
  #   Dense(32, activation='relu'),  # Hidden layer
  #   Dense(len(np.unique(y_train)), activation='softmax')
  #   # Output layer for classification
  # ])
  #
  # # Compile the model
  # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
  #               metrics=['accuracy'])
  #
  # # Train the model
  # history = model.fit(X_train, y_train, epochs=50, batch_size=32,
  #                     validation_data=(X_test, y_test))
  #
  # # Evaluate the model
  # loss, accuracy = model.evaluate(X_test, y_test)
  # print("Test Accuracy:", accuracy)

  # Initialize and train the model
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  # Make predictions
  y_pred = model.predict(X_test)

  # Evaluate the model
  print("Accuracy:", accuracy_score(y_test, y_pred))
  print("Classification Report:\n", classification_report(y_test, y_pred))

  # # Display to verify common and unique columns
  # print("Combined Columns:", combined_df.columns)
  # plt.figure(figsize=(10, 8))
  # sns.heatmap(combined_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
  # plt.title("Correlation Matrix")
  # plt.show()
