import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Page title
st.title("🤖 No-Code ML — Train Your Own Model")
st.write("Upload any CSV dataset, pick a target column, and train a machine learning model — no coding required.")

# Step 1 - Upload CSV
st.header("Step 1 — Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Step 2 - Pick target column
    st.header("Step 2 — Select Target Column")
    target = st.selectbox("Which column do you want to predict?", df.columns)

    # Step 3 - Pick model
    st.header("Step 3 — Choose a Model")
    model_choice = st.selectbox("Select ML Model", [
        "Random Forest",
        "Logistic Regression"
    ])

    # Step 4 - Train
    st.header("Step 4 — Train Your Model")
    if st.button("🚀 Train Model"):
        # Prepare data
        X = df.drop(columns=[target])
        y = df[target]

        # Encode text columns
        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Encode target if text
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Results
        st.success(f"✅ Model trained successfully!")
        st.metric("Accuracy", f"{accuracy*100:.2f}%")

        # Feature importance (Random Forest only)
        if model_choice == "Random Forest":
            st.subheader("Feature Importance")
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig, ax = plt.subplots()
            sns.barplot(data=importance, x='Importance', y='Feature', ax=ax)
            st.pyplot(fig)

        # Classification report
        st.subheader("Detailed Report")
        report = classification_report(y_test, predictions, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())