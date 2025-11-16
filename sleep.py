import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Page configuration
st.set_page_config(page_title="Sleep Quality Predictor", page_icon="😴", layout="wide")

# Title
st.title("😴 Sleep Quality Predictor")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Analysis", "Model Training", "Make Predictions"])

# Load or train model flag
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    df.columns = df.columns.str.replace(' ', '_')
    df = df.drop_duplicates()
    return df

# PAGE 1: DATA ANALYSIS
if page == "Data Analysis":
    st.header("📊 Data Analysis & Exploration")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.replace(' ', '_')
        df = df.drop_duplicates()
    else:
        try:
            df = load_data()
        except:
            st.warning("Please upload the Sleep_health_and_lifestyle_dataset.csv file")
            st.stop()
    
    # Display basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Duplicate Rows", df.duplicated().sum())
    
    # Show data
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    
    # Data info
    with st.expander("View Dataset Info"):
        st.text(f"Shape: {df.shape}")
        st.text("\nColumn Types:")
        # Fix: Convert dtype to string to avoid Arrow conversion issues
        info_df = pd.DataFrame({
            'Column': df.columns, 
            'Type': df.dtypes.astype(str), 
            'Non-Null Count': df.count()
        })
        st.dataframe(info_df)
        st.text(f"\nMissing Values:\n{df.isnull().sum()}")
    
    # Visualizations
    st.subheader("📈 Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Sleep Quality Distribution", "Stress vs Sleep", "Correlation Heatmap", "Feature Stats"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='Quality_of_Sleep', data=df, ax=ax, palette='viridis')
        ax.set_title("Distribution of Sleep Quality")
        st.pyplot(fig)
    
    with tab2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='Quality_of_Sleep', y='Stress_Level', data=df, ax=ax, palette='coolwarm')
        ax.set_title("Stress vs Sleep Quality")
        st.pyplot(fig)
    
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    
    with tab4:
        st.write("Correlation with Quality_of_Sleep:")
        corr = df.corr(numeric_only=True)['Quality_of_Sleep'].sort_values(ascending=False)
        st.dataframe(corr)

# PAGE 2: MODEL TRAINING
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    # Load data
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.replace(' ', '_')
        df = df.drop_duplicates()
    else:
        try:
            df = load_data()
        except:
            st.warning("Please upload the Sleep_health_and_lifestyle_dataset.csv file")
            st.stop()
    
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 25) / 100
        n_estimators = st.slider("Number of Trees", 50, 200, 100)
    with col2:
        random_state = st.number_input("Random State", 0, 100, 42)
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Prepare data
            X = df.drop('Quality_of_Sleep', axis=1)
            y = df['Quality_of_Sleep']
            
            # Encode categorical variables
            categorical_cols = X.select_dtypes(include=['object']).columns
            label_encoders = {}
            
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state), stratify=y
            )
            
            # Scale data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=int(random_state))
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save to session state
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.label_encoders = label_encoders
            st.session_state.X_columns = X.columns.tolist()
            st.session_state.model_trained = True
            
            # Save models to disk
            joblib.dump(model, 'sleep_quality_predictor.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            joblib.dump(label_encoders, 'label_encoders.pkl')
            
            st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
            
            with col2:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 4))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            feat_imp = pd.Series(model.feature_importances_, index=X.columns)
            fig, ax = plt.subplots(figsize=(8, 5))
            feat_imp.sort_values().plot(kind='barh', ax=ax, color='teal')
            ax.set_title("Feature Importance")
            st.pyplot(fig)
    
    # Load existing model
    if os.path.exists('sleep_quality_predictor.pkl') and not st.session_state.model_trained:
        if st.button("Load Existing Model"):
            st.session_state.model = joblib.load('sleep_quality_predictor.pkl')
            st.session_state.scaler = joblib.load('scaler.pkl')
            st.session_state.label_encoders = joblib.load('label_encoders.pkl')
            st.session_state.model_trained = True
            st.success("✅ Model loaded successfully!")

# PAGE 3: MAKE PREDICTIONS
elif page == "Make Predictions":
    st.header("🔮 Make Predictions")
    
    # Check if model is trained
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train or load a model first from the 'Model Training' page.")
        st.stop()
    
    st.write("Enter the details below to predict sleep quality:")
    
    # Load data to get categorical options
    try:
        df = load_data()
    except:
        st.error("Unable to load dataset for reference")
        st.stop()
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        person_id = st.number_input("Person ID", min_value=1, value=1)
        gender = st.selectbox("Gender", df['Gender'].unique() if 'Gender' in df.columns else ['Male', 'Female'])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        occupation = st.selectbox("Occupation", df['Occupation'].unique() if 'Occupation' in df.columns else ['Engineer'])
        sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=12.0, value=7.0, step=0.1)
        physical_activity = st.number_input("Physical Activity Level (0-100)", min_value=0, max_value=100, value=50)
    
    with col2:
        stress_level = st.number_input("Stress Level (1-10)", min_value=1, max_value=10, value=5)
        bmi_category = st.selectbox("BMI Category", df['BMI_Category'].unique() if 'BMI_Category' in df.columns else ['Normal'])
        blood_pressure = st.text_input("Blood Pressure", value="120/80")
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=120, value=70)
        daily_steps = st.number_input("Daily Steps", min_value=0, max_value=20000, value=7000)
        sleep_disorder = st.selectbox("Sleep Disorder", df['Sleep_Disorder'].unique() if 'Sleep_Disorder' in df.columns else ['None'])
    
    if st.button("Predict Sleep Quality", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Person_ID': [person_id],
            'Gender': [gender],
            'Age': [age],
            'Occupation': [occupation],
            'Sleep_Duration': [sleep_duration],
            'Physical_Activity_Level': [physical_activity],
            'Stress_Level': [stress_level],
            'BMI_Category': [bmi_category],
            'Blood_Pressure': [blood_pressure],
            'Heart_Rate': [heart_rate],
            'Daily_Steps': [daily_steps],
            'Sleep_Disorder': [sleep_disorder]
        })
        
        # Encode categorical variables
        for col in st.session_state.label_encoders.keys():
            if col in input_data.columns:
                try:
                    input_data[col] = st.session_state.label_encoders[col].transform(input_data[col])
                except:
                    st.error(f"Invalid value for {col}")
                    st.stop()
        
        # Make prediction
        prediction = st.session_state.model.predict(input_data)
        
        # Display result
        st.success(f"### Predicted Sleep Quality: {prediction[0]}")
        
        # Show probability distribution if available
        if hasattr(st.session_state.model, 'predict_proba'):
            proba = st.session_state.model.predict_proba(input_data)[0]
            classes = st.session_state.model.classes_
            
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Sleep Quality': classes,
                'Probability': proba
            })
            st.bar_chart(prob_df.set_index('Sleep Quality'))

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")