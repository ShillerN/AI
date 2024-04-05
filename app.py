import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


# Page
st.title('Machine Learning Model Selection')

page = st.sidebar.selectbox("Select Page", ["Model Selection"])

if page == "Model Selection":
    st.header('Regression Model Selection')
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "txt"])
    
    if uploaded_file is not None:
        # Read dataset
        df = pd.read_csv(uploaded_file)

        # Show dataset
        st.write("Preview of the dataset:")
        st.write(df.head())

        # Sidebar for selecting features and target variable
        selected_features = st.sidebar.multiselect("Select features for regression", df.columns)
        selected_target = st.sidebar.selectbox("Select the target variable", df.columns)

        if st.sidebar.button("Train Model"):
            # Split dataset into training and testing sets
            X = df[selected_features]
            y = df[selected_target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a simple linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Show model performance
            st.write("Model trained successfully!")
            st.write("Mean Absolute Error:", np.mean(np.abs(y_pred - y_test)))

            # Download predictions
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Download Predictions")
            st.sidebar.write("Download the predictions made by the model.")
            if st.sidebar.button("Download Predictions"):
                # Create a DataFrame with predictions
                predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                # Convert DataFrame to CSV and download
                predictions_csv = predictions_df.to_csv(index=False)
                st.sidebar.download_button(label="Download CSV", data=predictions_csv, file_name="predictions.csv", mime="text/csv")