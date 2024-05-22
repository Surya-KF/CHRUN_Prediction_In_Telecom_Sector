import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle

# Load the data
df_1 = pd.read_csv("first_telc.csv")

# Load the model
model = pickle.load(open("model.sav", "rb"))

# Get the columns used during model training
expected_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                     'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                     'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']


def main():
    st.markdown("<h1 style='text-align: center; color:#9fd3c7 ;'>Customer Churn Prediction in Telecom Sector</h1>", unsafe_allow_html=True)

    # Define the input fields
    SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    MonthlyCharges = st.text_input("Monthly Charges")
    TotalCharges = st.text_input("Total Charges")
    gender = st.selectbox("Gender", ["Male", "Female"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                                    "Credit card (automatic)"])
    tenure = st.text_input("Tenure")

    if st.button("Predict"):
        data = [[SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents, PhoneService,
                 MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                 StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure]]

        new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
                                             'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                             'InternetService',
                                             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                             'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                             'PaymentMethod', 'tenure'])

        # Preprocess new data in the same way as training data
        # Group the tenure in bins of 12 months
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        new_df['tenure_group'] = pd.cut(new_df.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
        new_df.drop(columns=['tenure'], axis=1, inplace=True)

        # Convert categorical variables to the same format as during training
        new_df_dummies = pd.get_dummies(new_df[expected_features])

        # Ensure all expected features are present, even if missing in new_df_dummies
        for feature in model.feature_names_in_:
            if feature not in new_df_dummies.columns:
                new_df_dummies[feature] = 0

        # Reorder the columns to match the model's training data
        new_df_dummies = new_df_dummies[model.feature_names_in_]

        single = model.predict(new_df_dummies)
        probability = model.predict_proba(new_df_dummies)[:, 1]

        if single == 1:
            st.markdown("<h2 style='text-align: center; color: red;'>This customer is likely to churn!</h2>",
                        unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Confidence: {probability[0] * 100:.2f}%</h3>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: green;'>This customer is likely to continue!</h2>",
                        unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Confidence: {probability[0] * 100:.2f}%</h3>",
                        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
