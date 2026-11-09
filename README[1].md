# Student Performance Prediction System

## Project Description
A classroom prototype for identifying students who may be
academically at risk using historical student information.

## Dataset
UCI Student Performance Dataset - Mathematics.

https://archive.ics.uci.edu/dataset/320/student+performance

## Prediction Target
- G3 < 10 = AT RISK
- G3 >= 10 = NOT AT RISK

G3 is the target variable and is excluded from model input.

## Users
- Faculty Member
- Academic Analyst

## Technologies
- Python
- Pandas
- Scikit-learn
- Random Forest
- Streamlit
- Google Colab
- GitHub

## Project Files
- SAD_mini_analysis.txt
- train_model.py
- app.py
- requirements.txt
- README.md
- student_data.csv
- student_model.pkl
- model_metrics.json

## How to Run

Install requirements:

pip install -r requirements.txt

Train the model:

python train_model.py

Run Streamlit:

streamlit run app.py

## Model Evaluation
Accuracy, precision, recall and F1-score are generated
by train_model.py and stored in model_metrics.json.

## Limitation
The model uses historical data from Portuguese secondary
schools and may not generalize to other institutions.
Predictions are estimates and should not be the sole basis
for real academic decisions.

## References
UCI Student Performance Dataset:
https://archive.ics.uci.edu/dataset/320/student+performance

Scikit-learn:
https://scikit-learn.org/stable/

Streamlit:
https://docs.streamlit.io/

GitHub:
https://docs.github.com/en/get-started
