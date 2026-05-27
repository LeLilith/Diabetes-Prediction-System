# AI-Based Diabetes Prediction System

This repository contains a diabetes prediction project that uses machine learning to classify whether a person is diabetic based on medical features. It includes data preprocessing, model training, evaluation, and a Streamlit web app for predictions.

## Contents

- `app.py` - Streamlit application for real-time diabetes risk prediction.
- `model_training.ipynb` - Notebook for data cleaning, preprocessing, training, hyperparameter tuning, and model evaluation.
- `diabetes.csv` - Raw dataset used for training and testing.
- `requirements.txt` - Python package dependencies.
- `project-info.md` - Project overview and goals.
- `diabetes_model.pkl` - Saved Random Forest model (generated after training).
- `diabetes_scaler.pkl` - Saved scaler used for input normalization.

## Features

- Handles missing values for clinical measurements.
- Scales numeric features using `StandardScaler`.
- Builds and evaluates a Random Forest classifier.
- Applies SMOTE oversampling to address class imbalance.
- Provides a Streamlit UI for entering patient data and viewing predictions.

## Installation

1. Create a Python virtual environment:

```bash
python -m venv .venv
```

2. Activate the environment:

- Windows (PowerShell):
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- Windows (cmd):
  ```cmd
  .venv\Scripts\activate.bat
  ```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Streamlit App

Start the web app from the repository root:

```bash
streamlit run app.py
```

Then open the provided local URL in your browser.

## Training and Evaluation

Open `model_training.ipynb` in Jupyter or VS Code Notebook to run the model training pipeline.

The notebook includes steps for:

- Loading and inspecting `diabetes.csv`
- Replacing invalid zeros with missing values
- Imputing missing values with the median
- Scaling features
- Training a Random Forest classifier
- Searching hyperparameters with `GridSearchCV`
- Evaluating model metrics: accuracy, precision, recall, F1-score, and ROC-AUC
- Using SMOTE to balance classes
- Saving the final model and scaler to `diabetes_model.pkl` and `diabetes_scaler.pkl`

## Dataset

The dataset is expected to be stored in `diabetes.csv` and must include the following columns:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

## Notes

- The Streamlit app depends on `diabetes_model.pkl` and `diabetes_scaler.pkl` being present in the root folder.
- If you retrain the model, rerun the notebook and regenerate the saved model/scaler files.
- Use the notebook for experimentation and performance comparison.

## License

This project does not include a license. Add one if you intend to share or publish the code."# -Diabetes-Prediction-System" 
"# Diabetes-Prediction-System" 
"# Diabetes-Prediction-System" 
"# Diabetes-Prediction-System" 
