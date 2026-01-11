## Job Suitability Predictor
# Project Overview

The Job Suitability Predictor is a machine learning web application that predicts whether a person is suitable for a job based on their age, gender, education, skill level, and years of experience. The project is part of a Capstone initiative to demonstrate skills in Python, machine learning, data preprocessing, and web deployment with Streamlit.

Features

Predict job suitability based on user input.

Interactive UI using Streamlit.

Human-readable inputs: dropdowns for Gender, Education, and Skill Level.

Robust ML model with data scaling and preprocessing.

Easily extendable with real datasets.

Project Structure
capstone123/
├─ app.py                     ← Streamlit application
├─ model.joblib               ← Saved ML model with scaler and encoders
├─ train_and_save_model.py     ← Script to create model.joblib
└─ src/
    ├─ __init__.py             ← Package initializer
    ├─ utility.py              ← Save/load model & prepare input
    └─ model.py                ← Model training function

Data and Modeling
1. Dataset

Columns: Age, Gender, Education, Skill_Level, Experience, Target

Target variable: 1 = Suitable, 0 = Not Suitable.

Note: For demonstration purposes, a small dummy dataset was used. In a real scenario, replace this with actual data collected from job applicants.

2. Data Cleaning & Preprocessing

Label Encoding for categorical variables:

Gender: Male=0, Female=1

Education: High School=0, Bachelors=1, Masters=2

Skill_Level: Beginner=0, Intermediate=1, Advanced=2

StandardScaler applied to features for consistent scaling.

3. Model Training

Model used: RandomForestClassifier from scikit-learn.

Train/test split: 80% training, 20% testing.

Model evaluated using accuracy score.

### How It Works

User opens the app in a browser.

User enters:

Age

Gender

Education

Skill Level

Years of Experience

Inputs are converted using LabelEncoders.

Inputs are scaled using the StandardScaler.

The model predicts Suitable or Not Suitable.

Result is displayed on the app.