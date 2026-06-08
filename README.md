📘 README.md
Appliance Energy Prediction Using Deep Learning (LSTM)

📌 Project Overview

This project focuses on predicting appliance energy consumption using a multivariate time-series dataset collected from a residential building. The dataset includes environmental conditions, indoor/outdoor sensor readings, and time-based features recorded at 10-minute intervals.

The goal is to build a predictive model using Machine Learning and Deep Learning (LSTM) to accurately forecast energy usage.

🎯 Objectives
Perform data preprocessing and cleaning
Conduct exploratory data analysis (EDA)
Engineer meaningful time-series features
Build baseline machine learning models
Develop a deep learning LSTM model
Evaluate performance using regression metrics
Optimize model for better accuracy
📂 Project Structure

Appliance-Energy-Prediction/
│
├── data/
│   ├── raw/
│   │   └── energy_data_set.csv
│   └── processed/
│       └── processed\_energy\_data.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── data\_preprocessing.py
│   ├── feature\_engineering.py
│   ├── baseline\_models.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│   └── trained\_model.keras
│
├── outputs/
│   ├── plots/
│   └── metrics/
│
├── reports/
│   └── report.pdf
│
├── requirements.txt
├── main.py
└── README.md

📊 Dataset Information
Target Variable: Appliances (Energy Consumption in Wh)
Sampling Rate: 10-minute intervals
Size: ~20,000 records
Features:
Indoor temperature (T1–T6)
Indoor humidity (RH_1–RH_6)
Outdoor weather (T_out, RH_out, windspeed, visibility)
Time features (hour, day, month, weekday)
Energy usage (lights, appliances)

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/YourUsername/Appliance-Energy-Prediction.git
cd Appliance-Energy-Prediction
2️⃣ Create virtual environment (recommended)
python -m venv venv

Activate:

Windows:

venv\Scripts\activate

Mac/Linux:

source venv/bin/activate
3️⃣ Install dependencies
pip install -r requirements.txt
▶️ How to Run the Project
🔹 Step 1: Data Preprocessing
python src/data_preprocessing.py

✔ This will:

Clean dataset
Handle missing values
Create time features
Save processed dataset in:
data/processed/processed_energy_data.csv
🔹 Step 2: Feature Engineering
python src/feature_engineering.py
🔹 Step 3: Train Models
python src/train.py

✔ Trains:

Baseline models (Linear Regression, Random Forest)
LSTM deep learning model
🔹 Step 4: Evaluate Model
python src/evaluate.py

✔ Generates:

MAE, RMSE, R²
Actual vs Predicted plots
Residual plots
🔹 Step 5: Run Full Pipeline
python main.py

✔ Executes full workflow end-to-end

📈 Evaluation Metrics

The model is evaluated using:

MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
MAPE (Mean Absolute Percentage Error)
R² Score

🧠 Model Architecture (LSTM)
LSTM Layer (64 units)
Dropout (0.2)
LSTM Layer (32 units)
Dense Output Layer
Optimizer: Adam
Loss Function: MSE

📊 Visualizations

The project includes the following plots:

Energy consumption trend
Correlation heatmap
Feature importance plot
Training vs validation loss
Actual vs predicted values
Residual distribution plot

All outputs are saved in:

outputs/plots/
🧪 Baseline Models
Linear Regression
Random Forest Regressor

These are used for performance comparison with LSTM.

🚀 Results Summary
Model	Performance
Linear Regression	Low accuracy
Random Forest	Medium accuracy
LSTM Model	Best performance

⚠️ Challenges Faced
Missing values in time-series data
Overfitting in deep learning model
Feature scaling complexity
Sequential dependency handling

💡 Future Improvements
Use Transformer models for better forecasting
Add external weather API data
Deploy model using Streamlit or Flask
Improve hyperparameter tuning

👨‍💻 Author

Name: Jithara Siriwardana
Project: Appliance Energy Prediction
Domain: Deep Learning / Time-Series Forecasting
