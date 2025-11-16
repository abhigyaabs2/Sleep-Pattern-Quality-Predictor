# 😴 Sleep Quality Predictor

A machine learning-powered web application built with Streamlit that predicts sleep quality based on health and lifestyle factors.


## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

The Sleep Quality Predictor is an interactive web application that analyzes the relationship between various health metrics and sleep quality. It uses a Random Forest Classifier to predict sleep quality ratings based on factors such as:

- Age, Gender, and Occupation
- Sleep Duration
- Physical Activity Level
- Stress Level
- BMI Category
- Heart Rate and Blood Pressure
- Daily Steps
- Sleep Disorders

## ✨ Features

### 📊 Data Analysis
- **Interactive Data Exploration**: View and analyze the dataset with detailed statistics
- **Visualizations**: 
  - Sleep quality distribution charts
  - Stress vs. sleep quality box plots
  - Correlation heatmaps
  - Feature statistics
- **Data Insights**: Understand relationships between different health factors

### 🤖 Model Training
- **Configurable Parameters**: Customize test size, number of trees, and random state
- **Real-time Training**: Train the model directly in the browser
- **Performance Metrics**: 
  - Accuracy score
  - Classification report
  - Confusion matrix
  - Feature importance visualization
- **Model Persistence**: Save and load trained models

### 🔮 Predictions
- **User-friendly Input Form**: Enter health metrics through an intuitive interface
- **Instant Predictions**: Get sleep quality predictions in real-time
- **Probability Distribution**: View confidence levels for each prediction
- **Input Validation**: Ensures data quality and prevents errors

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/abhigyaabs2/Sleep-Pattern-Quality-Predictor.git
cd Sleep-Pattern-Quality-Predictor
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run sleep.py
```

5. **Open your browser**
The app will automatically open at `http://localhost:8501`

## 📦 Requirements

Create a `requirements.txt` file with the following:

```
streamlit==1.28.0
pandas==2.1.0
numpy==1.24.3
seaborn==0.12.2
matplotlib==3.7.2
scikit-learn==1.3.0
joblib==1.3.2
```

## 💻 Usage

### 1. Data Analysis
- Navigate to the "Data Analysis" page
- Upload your CSV file or use the default dataset
- Explore visualizations and statistics
- Understand correlations between features

### 2. Model Training
- Go to the "Model Training" page
- Adjust training parameters:
  - Test size (10-40%)
  - Number of trees (50-200)
  - Random state (for reproducibility)
- Click "Train Model" to build your classifier
- View performance metrics and feature importance
- Model is automatically saved for later use

### 3. Making Predictions
- Navigate to the "Make Predictions" page
- Fill in the required health metrics:
  - Personal information (age, gender, occupation)
  - Sleep and activity data
  - Health indicators
- Click "Predict Sleep Quality"
- View your predicted sleep quality score and probability distribution

## 📊 Dataset

The application uses the **Sleep Health and Lifestyle Dataset** which contains:

- **374 records** of individuals
- **13 features** including demographics, health metrics, and lifestyle factors
- **Target variable**: Quality of Sleep (rated 1-10)

### Features Description:

| Feature | Description |
|---------|-------------|
| Person_ID | Unique identifier |
| Gender | Male/Female |
| Age | Age in years |
| Occupation | Job category |
| Sleep_Duration | Hours of sleep per day |
| Quality_of_Sleep | Rating from 1-10 (Target) |
| Physical_Activity_Level | Minutes of activity per day |
| Stress_Level | Rating from 1-10 |
| BMI_Category | Underweight/Normal/Overweight/Obese |
| Blood_Pressure | Systolic/Diastolic |
| Heart_Rate | Beats per minute |
| Daily_Steps | Number of steps per day |
| Sleep_Disorder | None/Insomnia/Sleep Apnea |

## 🤖 Model

### Algorithm
**Random Forest Classifier** - An ensemble learning method that constructs multiple decision trees and outputs the mode of their predictions.

### Why Random Forest?
- ✅ Handles both numerical and categorical features
- ✅ Resistant to overfitting
- ✅ Provides feature importance rankings
- ✅ Works well with small to medium datasets
- ✅ No need for feature scaling (though we apply it for consistency)

### Model Performance
The model achieves approximately **85-90% accuracy** on test data, depending on the random split and parameters used.

### Feature Importance
The model identifies the most influential factors affecting sleep quality, helping users understand what impacts their sleep most.

## 📁 Project Structure

```
sleep-quality-predictor/
│
├── app.py                                    # Main Streamlit application
├── Sleep_health_and_lifestyle_dataset.csv   # Dataset
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
│
├── models/                                   # Saved models (generated)
│   ├── sleep_quality_predictor.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
│
└── screenshots/                              # App screenshots (optional)
    ├── data_analysis.png
    ├── model_training.png
    └── predictions.png
```

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)**: Web application framework
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis
- **[NumPy](https://numpy.org/)**: Numerical computing
- **[Scikit-learn](https://scikit-learn.org/)**: Machine learning library
- **[Seaborn](https://seaborn.pydata.org/)**: Statistical data visualization
- **[Matplotlib](https://matplotlib.org/)**: Plotting library
- **[Joblib](https://joblib.readthedocs.io/)**: Model serialization

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contribution:
- Add more machine learning models (XGBoost, SVM, Neural Networks)
- Implement model comparison feature
- Add data augmentation techniques
- Create a mobile-responsive design
- Add user authentication and history tracking
- Implement A/B testing for different models

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- Dataset source: [Kaggle - Sleep Health and Lifestyle Dataset](https://www.kaggle.com/)
- Inspired by health and wellness research
- Thanks to the Streamlit community for excellent documentation
- Special thanks to all contributors

## 📈 Future Enhancements

- [ ] Add more visualization options
- [ ] Implement hyperparameter tuning
- [ ] Add model explainability (SHAP values)
- [ ] Create API endpoint for predictions
- [ ] Add batch prediction functionality
- [ ] Implement user feedback mechanism
- [ ] Add sleep recommendations based on predictions
- [ ] Multi-language support

---

**⭐ If you found this project helpful, please consider giving it a star!**
