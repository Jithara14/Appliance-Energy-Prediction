
📘 Appliance Energy Prediction Using Deep Learning (LSTM)
📌 Project Overview

This project focuses on predicting appliance energy consumption using a multivariate time-series dataset collected from a residential building. The dataset includes environmental conditions, indoor/outdoor sensor readings, and time-based features recorded at 10-minute intervals.

The main objective is to build a Machine Learning and Deep Learning (LSTM) model to accurately forecast energy usage and analyze consumption patterns.

🎯 Objectives

Perform data preprocessing and cleaning
Conduct exploratory data analysis (EDA)
Engineer meaningful time-series features
Build baseline machine learning models
Develop a deep learning LSTM model
Apply hyperparameter tuning for optimization
Evaluate performance using regression metrics
Optimize model for better accuracy

📂 Project Structure
Appliance-Energy-Prediction/
│
├── data/
│   ├── raw/
│   │   └── energy_data_set.csv
│   └── processed/
│       └── processed_energy_data.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── baseline_models.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── hyperparameter_tuning.py   # NEW (optional file)
│   └── utils.py
│
├── models/
│   └── trained_model.keras
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
⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/YourUsername/Appliance-Energy-Prediction.git
cd Appliance-Energy-Prediction
2️⃣ Create virtual environment

Windows

python -m venv venv
venv\Scripts\activate

Mac/Linux

python -m venv venv
source venv/bin/activate
3️⃣ Install dependencies
pip install -r requirements.txt
▶️ How to Run the Project
Step 1: Preprocessing
python src/data_preprocessing.py
Step 2: Feature Engineering
python src/feature_engineering.py
Step 3: Train Model
python src/train.py
Step 4: Evaluate Model
python src/evaluate.py
Step 5: Run Full Pipeline
python main.py

🧠 Model Architecture (LSTM)
LSTM Layer (64 units)
Dropout (0.2)
LSTM Layer (32 units)
Dense Layer (32 units, ReLU)
Output Layer (1 unit)
Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
🚀 Hyperparameter Tuning
📌 Overview

Hyperparameter tuning is performed to improve the performance of the LSTM model by optimizing key training and architecture parameters. Since deep learning models are sensitive to configuration, tuning helps achieve better generalization and reduce overfitting.

🔧 Hyperparameters Tuned

The following hyperparameters were analyzed and optimized:

1. Learning Rate

Controls the speed of model learning.

Tested values:
0.001 (default)
0.0005
0.0001

✔ Final choice: 0.001

2. Batch Size

Number of samples processed before updating weights.

Tested values:
16
32
64

✔ Final choice: 32

3. LSTM Units

Defines model capacity.

Tested:
64
128
256

✔ Final choice: 128 → 64 (2-layer LSTM)

4. Dropout Rate

Used to prevent overfitting.

Tested:
0.1
0.2
0.3

✔ Final choice: 0.2

5. Number of Layers
1 layer → underfitting
2 layers → best performance
3 layers → overfitting

✔ Final choice: 2 LSTM layers

6. Epochs
Range tested: 20–50
Early stopping applied to avoid overtraining

✔ Final behavior: Early stopping at optimal epoch

⚙️ Optimization Techniques Used
✔ Early Stopping

Stops training when validation loss stops improving:

patience = 5
restore_best_weights = True
✔ Validation Split
20% of training data used for validation
Helps monitor generalization performance
✔ Model Selection Strategy
Manual tuning approach
Compared multiple configurations
Selected model based on lowest validation loss + best RMSE
📊 Final Tuned Configuration
Hyperparameter	Value
Learning Rate	0.001
Batch Size	32
LSTM Layers	2
Units	128 + 64
Dropout	0.2
Optimizer	Adam
Loss Function	MSE
📈 Impact of Hyperparameter Tuning

After tuning:

Reduced overfitting
Improved validation stability
Increased R² score
Lowered RMSE and MAE
Better generalization on unseen data
🚀 Future Improvements
Use Keras Tuner / Optuna for automated search
Try Transformer-based models
Bayesian Optimization for faster tuning
Grid Search for structured comparison
👨‍💻 Author
Name: Jithara Siriwardana
Project: Appliance Energy Prediction
Domain: Deep Learning / Time-Series Forecasting
