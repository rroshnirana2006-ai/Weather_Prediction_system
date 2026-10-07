# 🌤️ Weather Prediction System

A **Machine Learning-based Weather Prediction System** that combines a **Random Forest Regression model**, **Streamlit**, and a **live Weather API** to provide real-time weather information and predict temperature.

The application takes a city name from the user, retrieves its current weather data through an online API, processes the required features, and uses a trained Machine Learning model to predict the temperature. 🌡️

---

## ✨ Features

* 🌍 **Live Weather Data** using WeatherAPI
* 🤖 **Machine Learning-based Temperature Prediction**
* 🌲 **Random Forest Regression** model
* 🖥️ Interactive **Streamlit Web Interface**
* 📊 Exploratory Data Analysis (EDA)
* 🧹 Data cleaning and preprocessing
* 🔍 Missing-value handling
* ♻️ Duplicate-value detection and removal
* 📈 Outlier detection using the **IQR method**
* 🕐 Time-based feature engineering
* 🔄 Cyclical time features using **Sine and Cosine transformations**
* 📋 Comparison between actual and predicted temperature
* 💾 Trained model saved using **Joblib**

---

## 🧠 How the Project Works

The project follows these major steps:

```text
Weather Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Train Random Forest Model
       ↓
Evaluate Model
       ↓
Save Model (.pkl)
       ↓
Streamlit Application
       ↓
User Enters City
       ↓
Live Weather API
       ↓
Process Weather Data
       ↓
Machine Learning Prediction
       ↓
Display Actual & Predicted Temperature
```

---

## 📊 1. Dataset

The Machine Learning model is trained using a historical weather dataset named:

```text
weatherHistory.csv
```

The dataset contains weather-related information such as:

* 🌡️ Temperature
* 🌡️ Apparent Temperature
* 💧 Humidity
* 💨 Wind Speed
* 🧭 Wind Bearing
* 👁️ Visibility
* 🌡️ Pressure
* 📅 Date and Time
* 🌦️ Precipitation Type

The target variable used for prediction is:

```text
Temperature (C)
```

---

## 🧹 2. Data Cleaning

Several preprocessing steps are performed before training the model.

### Missing Values

Missing values in the `Precip Type` column are handled using the **mode**:

```python
df['Precip Type'] = df['Precip Type'].fillna(
    df['Precip Type'].mode()[0]
)
```

### Duplicate Records

Duplicate rows are identified and removed:

```python
df = df.drop_duplicates()
```

### Date Conversion

The `Formatted Date` column is converted into a proper datetime format:

```python
df['Formatted Date'] = pd.to_datetime(
    df['Formatted Date'],
    utc=True
)
```

### Outlier Removal

The **Interquartile Range (IQR)** method is used to identify and remove extreme temperature values.

The IQR is calculated as:

```text
IQR = Q3 - Q1
```

Values outside:

```text
Q1 - 1.5 × IQR
Q3 + 1.5 × IQR
```

are treated as outliers.

---

## ⏰ 3. Feature Engineering

Time-related information is extracted from the date.

The following features are created:

```text
Year
Month
Day
Hour
```

For example:

```python
df["Year"] = df["Formatted Date"].dt.year
df["Month"] = df["Formatted Date"].dt.month
df["Day"] = df["Formatted Date"].dt.day
df["Hour"] = df["Formatted Date"].dt.hour
```

### 🔄 Cyclical Features

Time is cyclical. For example, hour 23 and hour 0 are very close in time, but a normal numerical representation would consider them far apart.

Therefore, sine and cosine transformations are used:

```python
Hour_sin = sin(2π × Hour / 24)
Hour_cos = cos(2π × Hour / 24)
```

Similarly, month-based cyclical features are created:

```python
Month_sin = sin(2π × Month / 12)
Month_cos = cos(2π × Month / 12)
```

This helps the Machine Learning model understand the repeating nature of time.

---

## 📈 4. Exploratory Data Analysis

EDA is performed to understand patterns and relationships in the weather dataset.

### 🌡️ Temperature Distribution

A histogram is used to understand how temperature values are distributed.

### 💧 Humidity vs Temperature

A scatter plot is used to study the relationship between humidity and temperature.

### 🔥 Weather Correlation

A correlation heatmap is used to understand relationships between different weather variables.

The project uses:

```text
Matplotlib
Seaborn
```

for visualization.

---

## 🤖 5. Machine Learning Model

The project uses:

### 🌲 Random Forest Regression

`RandomForestRegressor` from Scikit-learn is used to predict temperature.

```python
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
```

### Why Random Forest?

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees to make predictions.

It is useful for this project because weather data can have **complex and non-linear relationships** between variables such as humidity, wind, pressure, visibility, and temperature.

---

## 🎯 6. Features Used by the Model

The model uses the following input features:

| Feature                  | Description                   |
| ------------------------ | ----------------------------- |
| 🌡️ Apparent Temperature | Feels-like temperature        |
| 💧 Humidity              | Amount of moisture in the air |
| 💨 Wind Speed            | Wind speed in km/h            |
| 🧭 Wind Bearing          | Wind direction                |
| 👁️ Visibility           | Visibility distance           |
| 🌡️ Pressure             | Atmospheric pressure          |
| 📅 Year                  | Year                          |
| 📅 Month                 | Month                         |
| 📅 Day                   | Day                           |
| 🕐 Hour                  | Hour                          |
| 🔄 Hour Sin              | Cyclical hour feature         |
| 🔄 Hour Cos              | Cyclical hour feature         |
| 🔄 Month Sin             | Cyclical month feature        |
| 🔄 Month Cos             | Cyclical month feature        |

### 🎯 Target

```text
Temperature (C)
```

---

## 📊 7. Model Evaluation

The model is evaluated using three regression metrics:

### 📌 R² Score

Measures how well the model explains the variation in the target variable.

```python
r2_score(y_test, y_pred)
```

A value closer to **1** generally indicates better performance.

### 📌 MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

```python
mean_absolute_error(y_test, y_pred)
```

Lower MAE is better.

### 📌 RMSE — Root Mean Squared Error

Measures prediction error while giving more weight to larger errors.

```python
np.sqrt(mean_squared_error(y_test, y_pred))
```

Lower RMSE is better.

---

## 💾 8. Saving the Model

After training, the model is saved using **Joblib**:

```python
joblib.dump(model, "weather_model.pkl")
```

This allows the trained model to be loaded later without training it again.

```python
model = joblib.load("weather_model.pkl")
```

---

# 🌐 9. Live Weather API

The application uses **WeatherAPI** to retrieve current weather information from the internet.

The user enters a city name, and the application sends that city to the API.

The API provides information such as:

* 🌡️ Current temperature
* 🌡️ Feels-like temperature
* 💧 Humidity
* 💨 Wind speed
* 🌡️ Atmospheric pressure
* 👁️ Visibility
* 🌦️ Weather condition
* 📍 Location information
* 🕐 Local date and time

This means the application does not depend only on historical data—it can also retrieve **current weather information**.

---

# 🖥️ 10. Streamlit Interface

The user interface is developed using **Streamlit**.

The application provides a simple interface where the user enters a city:

```text
Enter City Name
[ Ludhiana              ]

🔍 Get Weather & Predict
```

After clicking the button, the application:

1. 📍 Gets the city information.
2. 🌐 Sends a request to the Weather API.
3. 📥 Receives live weather data.
4. ⚙️ Extracts the required features.
5. 🔄 Creates the same features used during model training.
6. 🤖 Sends the data to the Random Forest model.
7. 🌡️ Predicts the temperature.
8. 📊 Displays live weather and prediction results.

---

## 🔍 11. Actual vs Predicted Temperature

The application displays both:

### 🌡️ Actual Temperature

The current temperature received directly from the Weather API.

### 🤖 Predicted Temperature

The temperature predicted by the trained Machine Learning model using the weather features received from the API.

The application also calculates the difference:

```python
difference = predicted_temperature - current["temp_c"]
```

This allows the user to compare the model prediction with the current API temperature.

---

## 🛠️ Technologies Used

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| 🐍 Python        | Main programming language |
| 🐼 Pandas        | Data processing           |
| 🔢 NumPy         | Numerical calculations    |
| 📊 Matplotlib    | Data visualization        |
| 📈 Seaborn       | Statistical visualization |
| 🤖 Scikit-learn  | Machine Learning          |
| 🌲 Random Forest | Temperature prediction    |
| 💾 Joblib        | Model saving/loading      |
| 🖥️ Streamlit    | Web application           |
| 🌐 WeatherAPI    | Live weather data         |
| 🔗 Requests      | API requests              |

---

## 📁 Project Structure

```text
Weather-Prediction-System/
│
├── 📄 weatherHistory.csv
├── 📄 weather_model.pkl
├── 📄 app.py
├── 📄 weather.py
├── 📄 README.md
└── 📄 .gitignore
```


---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Weather-Prediction-System.git
```

### 2. Move into the Project Folder

```bash
cd Weather-Prediction-System
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit requests
```

Or, if you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 🔑 API Key

The application uses WeatherAPI.

Before running the project, add your own API key to the application.

```python
API_KEY = "YOUR_API_KEY"
```

⚠️ **Important:** Do not upload your real API key publicly to GitHub.

A better approach is to store the API key using **Streamlit secrets** or an environment variable.

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your web browser.

Then:

```text
1. Enter a city name 🌍
2. Click "Get Weather & Predict" 🔍
3. Wait for live weather data 🌐
4. View current weather 🌤️
5. View ML predicted temperature 🤖
6. Compare actual and predicted temperature 📊
```

---

## 📌 Example Output

For a city such as **Ludhiana**, the application can display:

```text
🌍 Live Weather

Actual Temperature: XX °C
Humidity: XX %
Wind Speed: XX km/h
Feels Like: XX °C
Pressure: XXXX mb
Visibility: XX km

🌦️ Weather Condition:
Clear

🤖 Machine Learning Prediction

Predicted Temperature: XX.XX °C

Difference between actual and predicted:
X.XX °C
```

---

## 🚀 Future Improvements

Some possible improvements for the project are:

* 🌧️ Predict rainfall
* ☁️ Predict weather conditions
* 📅 Add multi-day weather forecasting
* 📊 Add more interactive charts
* 🗺️ Add weather maps
* 📱 Improve mobile responsiveness
* 🔐 Store API keys securely
* ☁️ Deploy the application online
* 📈 Add model performance visualizations
* 🤖 Compare Random Forest with other ML algorithms

---

## 🎓 Project Objective

The main objective of this project is to demonstrate how **Machine Learning can be combined with real-time API data and a web interface** to create a practical weather prediction application.

The project covers the complete Machine Learning workflow:

```text
📥 Data Collection
      ↓
🧹 Data Cleaning
      ↓
📊 EDA
      ↓
⚙️ Feature Engineering
      ↓
🤖 Model Training
      ↓
📈 Model Evaluation
      ↓
💾 Model Saving
      ↓
🌐 API Integration
      ↓
🖥️ Streamlit Application
      ↓
🌡️ Real-Time Prediction
```

---

## 👩‍💻 BY:-

**Roshni Kumari**

🎓 AI & Data Science----Student

⭐ If you find this project useful, consider giving the repository a **star**!
