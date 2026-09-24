# Smart Manufacturing Equipment Failure Prediction

## 📌 Project Overview

Smart Manufacturing Equipment Failure Prediction is a supervised machine learning project that predicts whether manufacturing equipment is likely to fail based on sensor and operational data.

The project focuses on applying an end-to-end machine learning workflow, starting from data preprocessing and exploratory data analysis to model training, evaluation, model comparison, and deployment.

The final Random Forest model is deployed using Streamlit to provide an interactive interface for equipment failure prediction.

---

## 🎯 Problem Statement

Unexpected equipment failures can cause significant production downtime, maintenance costs, operational delays, and loss of productivity in manufacturing environments.

The objective of this project is to develop a machine learning solution that can analyze equipment sensor data and predict potential equipment failures, supporting predictive maintenance and data-driven decision-making.

---

## 💡 Objectives

- Analyze manufacturing equipment sensor data.
- Perform data cleaning and preprocessing.
- Conduct exploratory data analysis (EDA).
- Identify important patterns and features associated with equipment failure.
- Train and compare multiple machine learning models.
- Select an appropriate classification model.
- Evaluate model performance using classification metrics.
- Deploy the trained model using Streamlit.
- Provide an interactive interface for equipment failure prediction.

---

## 🔄 Project Workflow

```text
Raw Manufacturing Data
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Analysis
        ↓
Train-Test Split
        ↓
Machine Learning Models
        ↓
Model Comparison
        ↓
Model Evaluation
        ↓
Random Forest Classifier
        ↓
Model Deployment
        ↓
Streamlit Application
        ↓
Equipment Failure Prediction
🛠️ Technologies Used
Programming Language
Python
Data Analysis
Pandas
NumPy
Data Visualization
Matplotlib
Seaborn
Machine Learning
Scikit-learn
Random Forest Classifier
Deployment
Streamlit
Development Environment
Jupyter Notebook
VS Code
📊 Dataset

The project uses manufacturing equipment sensor and operational data containing features that describe the operating conditions of industrial equipment.

The data is used to identify patterns associated with equipment failure and train a supervised classification model.

Dataset Processing

The dataset undergoes:

Data inspection
Missing value analysis
Data cleaning
Feature preparation
Target variable identification
Exploratory data analysis
Train-test splitting
Model-ready preprocessing
🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure and characteristics of the manufacturing data.

The analysis includes:

Distribution analysis
Feature relationships
Correlation analysis
Failure class distribution
Outlier analysis
Sensor behavior analysis
Feature importance analysis

EDA helps identify patterns and relationships between equipment operating conditions and failure outcomes.

🤖 Machine Learning

This project follows a supervised classification approach.

Different machine learning models were trained and compared to determine an appropriate model for equipment failure prediction.

The final model selected for deployment is:

Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve predictive performance and handle complex relationships between input features and the target variable.

It is particularly useful for this project because manufacturing sensor data can contain non-linear relationships between operating conditions and equipment failures.

📈 Model Evaluation

The classification model is evaluated using commonly used classification metrics:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix

These metrics help evaluate the model's ability to correctly identify equipment failure cases while minimizing incorrect predictions.

🚀 Model Deployment

The trained Random Forest model is deployed using Streamlit.

The Streamlit application provides an interactive interface where users can enter equipment-related sensor values.

The application then:

User Input
    ↓
Input Validation
    ↓
Feature Preprocessing
    ↓
Trained Random Forest Model
    ↓
Prediction
    ↓
Equipment Failure Result

This allows the machine learning model to be used interactively rather than only through a development notebook.

🖥️ Application Features
Interactive user interface
Equipment sensor input
Real-time prediction
Machine learning model integration
Simple and user-friendly interface
Failure prediction output
📂 Project Structure
Smart_Manufacturing_Project/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── model_development.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── app.py
│
├── requirements.txt
│
└── README.md

The exact file names and folders may vary depending on the final project structure.

⚙️ Installation

Clone the repository:

git clone https://github.com/priyyaahh/Smart_manufacturing_equipment_failure_prediction.git

Navigate to the project directory:

cd Smart_manufacturing_equipment_failure_prediction

Create a virtual environment:

python -m venv venv

Activate the virtual environment on Windows:

venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt
▶️ Run the Application

Run the Streamlit application using:

streamlit run app.py

The application will open in your browser.

📌 Key Learnings

Through this project, I gained practical experience in:

Python for machine learning
Data preprocessing
Exploratory data analysis
Feature analysis
Supervised machine learning
Classification algorithms
Random Forest
Model evaluation
Model comparison
Machine learning model serialization
Streamlit application development
Basic ML model deployment
🔮 Future Enhancements

The project can be further enhanced by:

Integrating real-time IoT sensor data.
Developing a REST API using FastAPI.
Deploying the model on AWS, Azure, or GCP.
Implementing model monitoring.
Adding data and model drift detection.
Implementing automated model retraining.
Adding SHAP-based explainable AI.
Implementing real-time maintenance alerts.
Building an end-to-end MLOps pipeline.
Integrating the solution with manufacturing systems.
🏭 Real-World Applications

This type of predictive maintenance solution can be applied across industries such as:

Manufacturing
Automotive
Industrial equipment
Energy
Electronics
Process industries
Smart factories

The solution can help organizations move from reactive maintenance toward predictive maintenance by using machine learning to identify potential equipment failures before they occur.


Priya

Data Science | Machine Learning | AI | Data Analytics
