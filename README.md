Appliance Energy Prediction Using Machine Learning & LSTM
📌 Project Overview

This project focuses on predicting appliance energy consumption using environmental and sensor-based time-series data. It implements a complete end-to-end machine learning pipeline including:

Data preprocessing
Exploratory Data Analysis (EDA)
Feature engineering
Baseline machine learning models
Deep learning model (LSTM)
Model evaluation and comparison

The goal is to build an intelligent forecasting system that helps in energy optimization and smart building management.

📊 Dataset Description

The dataset contains 10-minute interval sensor readings from a residential building, including:

Temperature readings (T1–T9, T_out)
Humidity readings (RH_1–RH_9, RH_out)
Weather features (Windspeed, Visibility, Pressure, Dewpoint)
Target variable: Appliances energy consumption

🏗️ Project Structure

Appliance_Energy_Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── plots/
│   ├── metrics/
│   ├── models/
│   └── reports/
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── baseline_models.py
│   ├── sequence_generator.py
│   ├── models.py
│   ├── hyperparameter_tuning.py
│   ├── evaluation.py
│   ├── plots.py
│   ├── timeseries_validation.py
│   └── utils.py
│
├── main.py
├── requirements.txt
└── README.md

⚙️ Workflow Pipeline
1️⃣ Data Preprocessing
Load raw dataset
Handle missing values
Remove duplicates
Detect and treat outliers (IQR method)
Feature scaling using MinMaxScaler
Save cleaned dataset
2️⃣ Exploratory Data Analysis (EDA)
Data distribution analysis
Correlation heatmaps
Missing value visualization
Statistical summary of features
Trend analysis of energy consumption
3️⃣ Feature Engineering
Time-based features (hour, day, month, weekday)
Lag features (previous time steps)
Rolling statistics (moving averages)
Interaction features (temperature × humidity)
Feature selection and refinement
4️⃣ Baseline Models

Implemented models:

Linear Regression
Ridge Regression
Random Forest Regressor

Evaluation Metrics:

MAE
RMSE
R² Score
5️⃣ Deep Learning Model (LSTM)
Sequential LSTM architecture
Dropout regularization
Dense output layer
Optimized using Adam optimizer
Loss function: Mean Squared Error (MSE)
6️⃣ Hyperparameter Tuning

Tuned parameters:

LSTM units
Batch size
Learning rate
Dropout rate
Number of epochs
7️⃣ Model Evaluation

Models evaluated using:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score

📈 Results Summary
🔹 Baseline Models
Model	MAE	RMSE	R² Score
Linear Regression	13.93	21.67	0.744
Ridge Regression	13.92	21.67	0.744
Random Forest	14.05	22.29	0.729
🔹 LSTM Model
Model	MAE	RMSE	R² Score
LSTM	13.38	21.27	0.702


📌 Key Insights
Energy consumption is strongly influenced by environmental conditions.
Feature engineering significantly improves model performance.
Baseline models performed competitively due to strong linear patterns.
LSTM captures sequential patterns but requires tuning for better performance.


⚠️ Challenges Faced
Missing date column issues in feature engineering
Keras model loading compatibility errors
Sequence generation for LSTM input
Handling NaN values from lag/rolling features
Model performance variation between baseline and LSTM


🚀 Future Improvements
Implement GRU and Attention-based models
Use advanced hyperparameter tuning (Bayesian optimization)
Deploy model using FastAPI + React
Add real-time energy forecasting dashboard
Multi-step forecasting for future prediction


🛠️ Installation & Setup
1. Clone repository
git clone https://github.com/your-username/Appliance_Energy_Prediction.git
cd Appliance_Energy_Prediction


2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows


3. Install dependencies
pip install -r requirements.txt

▶️ Run Project
python main.py


📦 Requirements
Python 3.9+
NumPy
Pandas
Scikit-learn
TensorFlow / Keras
Matplotlib
Seaborn


👨‍💻 Author

Jithara Siriwardana
Machine Learning & AI Enthusiast
Focused on Deep Learning, Time-Series Forecasting, and Full-Stack AI Systems
