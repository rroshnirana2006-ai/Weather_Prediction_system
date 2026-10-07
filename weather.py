import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import joblib   

#perform EDA and data cleaning on the weather dataset
# Load the dataset
data = pd.read_csv('weatherHistory.csv')
df = pd.DataFrame(data)

# Understand the dataset
df.info()
df.describe()
df.head()
df.tail()
df.columns
df.shape

# Check for missing values
missing_values = df.isnull().sum()


#handle missing values
df['Precip Type']=df['Precip Type'].fillna(df['Precip Type'].mode()[0])
print("missing values after handling",df.isnull().sum())

#check for duplicates
duplicates = df.duplicated().sum()
print("Number of duplicates:", duplicates)

#handle duplicates
df = df.drop_duplicates()
print("Number of duplicates after handling:", df.duplicated().sum())

#incorrect data types
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'],utc=True)

#outlier detection and removal
# Using IQR method to detect and remove outliers in 'Temperature (C)' column
Q1 = df['Temperature (C)'].quantile(0.25)
Q3 = df['Temperature (C)'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Temperature (C)'] < (Q1 - 1.5 * IQR)) | (df['Temperature (C)'] > (Q3 + 1.5 * IQR)))]


# Create time features
df["Year"] = df["Formatted Date"].dt.year
df["Month"] = df["Formatted Date"].dt.month
df["Day"] = df["Formatted Date"].dt.day
df["Hour"] = df["Formatted Date"].dt.hour

# Create cyclical time features
df["Hour_sin"] = np.sin(
    2 * np.pi * df["Hour"] / 24
)

df["Hour_cos"] = np.cos(
    2 * np.pi * df["Hour"] / 24
)

df["Month_sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)

# -----------------------------
# 1. Temperature Distribution

plt.figure(figsize=(8, 5))

sns.histplot(df["Temperature (C)"], bins=50, kde=True)

plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Number of Records")

plt.show()


# 2. Humidity vs Temperature

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=df["Humidity"],
    y=df["Temperature (C)"],
    alpha=0.4
)

plt.title("Humidity vs Temperature")
plt.xlabel("Humidity")
plt.ylabel("Temperature")

plt.show()


# 3. Weather Correlation

columns = [
    "Temperature (C)",
    "Apparent Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Visibility (km)",
    "Pressure (millibars)",
    "Hour",
    "Month"
]

plt.figure(figsize=(10, 7))

sns.heatmap(
    df[columns].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Weather Correlation")

plt.show()

#Machine Learning Model
FEATURES = [

    "Apparent Temperature (C)",

    "Humidity",

    "Wind Speed (km/h)",

    "Wind Bearing (degrees)",

    "Visibility (km)",

    "Pressure (millibars)",

    "Year",

    "Month",

    "Day",

    "Hour",

    "Hour_sin",

    "Hour_cos",

    "Month_sin",

    "Month_cos"
]
TARGET = "Temperature (C)"

x = df[FEATURES]

y = df[TARGET]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=42)

# check the shape of the training and testing data
print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

model =  RandomForestRegressor(

        n_estimators=300,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        max_features=0.8,

        random_state=42,

        n_jobs=-1
    )


    

# Train the models
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# STEP 4: Calculate model performance

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

#print the model performance metrics

print("R2 Score:", round(r2, 4))

print(
    "R2 Percentage:",
    round(r2 * 100, 2),
    "%"
)

print("MAE:", round(mae, 4))

print("RMSE:", round(rmse, 4))

 #14. SAVE MODEL
 
joblib.dump(model, "weather_model.pkl")

print("Model saved successfully!")
