# 🩺 Diabetes Health Indicator

A full-stack machine learning pipeline for predicting diabetes, built with clarity, reproducibility, and responsible deployment in mind. This project includes data cleaning, visualization, model benchmarking, and a user-friendly Streamlit app.

## 📌 Project Highlights

- ✅ **Data Cleaning**: Imputation of missing values, outlier handling using MAD, and column validation.
- 📊 **Visualization**: Exploratory plots to understand feature distributions and relationships.
- 🤖 **Model Training**: Benchmarking four classifiers:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine
- 🌐 **Deployment**: A Streamlit app for real-time diabetes prediction with celebratory feedback and disclaimers.

## 🚀 Live App

Try the deployed app here 👉 [Diabetes Indicator Streamlit App](https://diabetes-indicator.streamlit.app/)



## 🧠 How It Works

1. **Preprocessing**: Missing values are imputed using median/mode strategies. Outliers are handled using MAD.
2. **Modeling**: Each classifier is trained and evaluated. Logistic Regression is selected for deployment based on performance.
3. **Deployment**: The app allows users to input health metrics and receive predictions, with celebratory feedback and disclaimers for responsible use.

## 🌟 Streamlit App Features

- Clean UI with input validation
- Celebration effects for positive predictions
- Medical disclaimer for ethical deployment
- Robust error handling and environment compatibility

## 👥 Contributors

This project is a collaborative effort by:

| Name         | GitHub Handle                             |
|--------------|--------------------------------------------|
| Om Kumar     | [Om-Kumar-Ace](https://github.com/Om-Kumar-Ace) |
| Anshu Anand  | [anshuanand22](https://github.com/anshuanand22) |
| Nayan Mahato | [Nayanbatman](https://github.com/Nayanbatman) |
| Saket Kumar  | [saketkumar28](https://github.com/saketkumar28) |

## 📦 Installation

To run locally:

```bash
git clone https://github.com/Om-Kumar-Ace/Diabetes-Health-Indicator.git
cd Diabetes-Health-Indicator
pip install -r requirements.txt
streamlit run app.py
