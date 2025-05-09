
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# Cache the data loading and figure creation for performance
@st.cache_data
def load_data_and_create_figure():
    # Load the Titanic dataset
    df = pd.read_csv('2nd_try_cleaned_Data.csv')

    # Create a 3x2 subplot grid
    fig = make_subplots(
        rows=3, cols=2, 
        subplot_titles=(
            'Survival Distribution', 
            'Age Distribution', 
            'Smoking Count', 
            'Correlation Matrix',
            'Smokers and Sick'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    # ✅ 1. Survival count plot
    survived_counts = df['HeartDisease'].value_counts()

    fig.add_trace(
        go.Bar(
            x=['sick'], 
            y=[survived_counts.get(1, 0)], 
            name='sick',
            marker=dict(color='red'),
            text=[survived_counts.get(1, 0)], 
            textposition='auto'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=['well'], 
            y=[survived_counts.get(0, 0)], 
            name='well',
            marker=dict(color='blue'),
            text=[survived_counts.get(0, 0)], 
            textposition='auto'
        ),
        row=1, col=1
    )

    # ✅ 2. Age distribution
    age = pd.to_numeric(df['AgeCategory'], errors='coerce').dropna()
    kde = gaussian_kde(age)
    x = np.linspace(age.min(), age.max(), 100)
    kde_values = kde(x)

    fig.add_trace(
        go.Histogram(
            x=age,
            nbinsx=20,
            histnorm='probability density',
            name='AgeCategory',
            marker=dict(color='orange'),
            opacity=0.7
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=kde_values,
            mode='lines',
            name='KDE',
            line=dict(color='red')
        ),
        row=1, col=2
    )

    # ✅ 3. Smoking count
    smoking_count = df['Smoking'].value_counts()

    fig.add_trace(
        go.Bar(
            x=['YES'], 
            y=[smoking_count.get('Yes', 0)], 
            name='Smokers',
            marker=dict(color='red'),
            text=[smoking_count.get('Yes', 0)], 
            textposition='auto'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=['NO'], 
            y=[smoking_count.get('No', 0)], 
            name='Non-Smokers',
            marker=dict(color='blue'),
            text=[smoking_count.get('No', 0)], 
            textposition='auto'
        ),
        row=2, col=1
    )

    # ✅ 4. Correlation Matrix
    corr_matrix = df.corr(numeric_only=True)[['Stroke','AlcoholDrinking','Smoking']]

    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(
                len=0.45,
                y=0.21,
                yanchor='middle'
            )
        ),
        row=2, col=2
    )

    # ✅ 5. Smokers and Sick
    grouped = df.groupby(['Smoking', 'HeartDisease']).size().unstack(fill_value=0)

    fig.add_trace(
        go.Bar(
            name='Sick = 1',
            x=grouped.index,
            y=grouped[1],
            marker_color='red'
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Bar(
            name='Sick = 0',
            x=grouped.index,
            y=grouped[0],
            marker_color='blue'
        ),
        row=3, col=1
    )

    # ✅ Layout
    fig.update_layout(
        height=1200,
        width=1200,
        title_text="Univariate and Multivariate Analysis",
        title_x=0.5,
        showlegend=True,
        barmode='group',
        template='plotly_white'
    )

    fig.update_xaxes(title_text="Age", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=1, col=2)
    fig.update_xaxes(title_text="Smoking Count", row=2, col=1)

    return df, fig

df, fig = load_data_and_create_figure()

# App interface
st.title('Heart Disease Predictor')

# Display the interactive Plotly figure
st.subheader('EDA')
st.plotly_chart(fig, use_container_width=False)

# Load pre-trained model and preprocessor
model = joblib.load("project_model.pkl")
preprocessor = joblib.load("preprocessor11.pkl")

# Predictive interface
st.subheader("Model Prediction")
# BMI	Smoking	AlcoholDrinking	Stroke	PhysicalHealth	MentalHealth	DiffWalking	Sex	AgeCategory	Race	Diabetic	PhysicalActivity	GenHealth	SleepTime	Asthma	KidneyDisease	SkinCancer
# Input widgets
col1, col2 = st.columns(2)
with col1:
    BMI = st.number_input("BMI", min_value=12, max_value=87, value=30,step=0.5)
    Smoking = st.selectbox("Smoking", [0,1])
    AlcoholDrinking = st.selectbox("AlcoholDrinking", [0,1])
    Stroke = st.selectbox("Stroke", [0,1])
    PhysicalHealth = st.number_input("PhysicalHealth", min_value=0, max_value=30, value=15)
    MentalHealth = st.number_input("MentalHealth", min_value=0, max_value=30, value=15)
    DiffWalking = st.selectbox("DiffWalking", [0,1])
    Sex = st.selectbox("Sex", [0,1])
   




with col2:
    AgeCategory = st.number_input("AgeCategory", min_value=0, max_value=12, value=10)
    Race = st.selectbox("Race", [0,1])
    Diabetic = st.selectbox("Diabetic", [0,1])
    PhysicalActivity = st.selectbox("PhysicalActivity", [0,1])
    GenHealth = st.number_input("GenHealth", min_value=0, max_value=4, value=2)
    SleepTime = st.number_input("SleepTime", min_value=0, max_value=24, value=10)
    Asthma = st.selectbox("Asthma", [0,1])
    KidneyDisease = st.selectbox("KidneyDisease", [0,1])
    SkinCancer = st.selectbox("SkinCancer", [0,1])

    # Prediction logic
if st.button("Predict case"):
    input_data = pd.DataFrame([[BMI, Smoking, AlcoholDrinking,Stroke,PhysicalHealth,MentalHealth,DiffWalking,Sex,AgeCategory,
                                   Race,Diabetic,	PhysicalActivity,	GenHealth,	SleepTime,	Asthma,	KidneyDisease,	SkinCancer]],
                                columns=['BMI', 'Smoking', 'AlcoholDrinking','Stroke','PhysicalHealth','MentalHealth','DiffWalking','Sex','AgeCategory',
                                   'Race',	'Diabetic',	'PhysicalActivity',	'GenHealth',	'SleepTime',	'Asthma',	'KidneyDisease',	'SkinCancer'])

    processed_data = preprocessor.transform(input_data)
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    st.subheader("Result")
    st.metric("Survival Probability", f"{probability:.1%}")
    st.write(f"Prediction {'Survived' if prediction == 1 else 'Did not survive'}")
