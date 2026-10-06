import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="Iris Classifier // Dark Mode",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for Cyber Dark Theme
st.markdown("""
    <style>
    /* Main Background & Base Font Color */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #111622;
        border-right: 1px solid #1E293B;
    }
    
    /* Headers & Glowing Text */
    h1 {
        color: #F8FAFC !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #38BDF8 !important;
        font-weight: 600 !important;
    }
    .stCaption, p {
        color: #94A3B8 !important;
    }
    
    /* Button with Neon Gradient & Glow */
    .stButton>button {
        background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%);
        color: #FFFFFF !important;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.6rem 1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.25);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.5);
        transform: translateY(-2px);
    }
    
    /* Result Card - Cyber Glassmorphism */
    .result-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);
        backdrop-filter: blur(12px);
        padding: 28px;
        border-radius: 16px;
        border: 1px solid #00F5D4;
        text-align: center;
        box-shadow: 0 0 25px rgba(0, 245, 212, 0.2), inset 0 0 15px rgba(0, 245, 212, 0.05);
    }
    .result-title {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .result-value {
        font-size: 36px;
        font-weight: 900;
        color: #00F5D4;
        text-shadow: 0 0 20px rgba(0, 245, 212, 0.6);
        margin-bottom: 8px;
        letter-spacing: 1px;
    }
    .result-confidence {
        font-size: 14px;
        color: #38BDF8;
        font-family: monospace;
    }
    
    /* Divider */
    hr {
        border-color: #1E293B !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load Dataset & Train Model
@st.cache_data
def load_data_and_model():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    avg_df = X.mean().reset_index()
    avg_df.columns = ['Feature', 'Dataset Average']
    
    return iris, model, avg_df

iris, model, avg_df = load_data_and_model()

# Header Section
st.markdown("<h1>⚡ Iris Species Neural Classifier</h1>", unsafe_allow_html=True)
st.caption("Real-time morphological classification powered by Random Forest")

st.markdown("---")

# Sidebar - Input Features
st.sidebar.markdown("### 🎛️ Input Parameters")
st.sidebar.caption("Fine-tune floral morphological measurements:")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.4, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

predict_btn = st.sidebar.button("Execute Inference")

# Layout: Split into 2 Columns
col1, col2 = st.columns([1.2, 1])

# Plotly Shared Dark Config
chart_font = dict(family="sans-serif", size=12, color="#94A3B8")

with col1:
    st.markdown("### 📊 Morphological Vector vs Baseline")
    st.caption("Comparison between input values and global averages")
    
    user_inputs = [sepal_length, sepal_width, petal_length, petal_width]
    features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    dataset_averages = avg_df['Dataset Average'].values
    
    # Dual Neon Bars
    fig = go.Figure(data=[
        go.Bar(
            name='Input Value',
            x=features,
            y=user_inputs,
            marker=dict(color='#00F5D4', line=dict(color='#5EEAD4', width=1)),
            text=user_inputs,
            textposition='auto',
            textfont=dict(color='#0B0E14', family='monospace')
        ),
        go.Bar(
            name='Dataset Average',
            x=features,
            y=dataset_averages,
            marker=dict(color='#334155', line=dict(color='#64748B', width=1)),
            text=np.round(dataset_averages, 2),
            textposition='auto',
            textfont=dict(color='#E2E8F0', family='monospace')
        )
    ])
    
    fig.update_layout(
        barmode='group',
        height=330,
        margin=dict(l=20, r=20, t=30, b=20),
        font=chart_font,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#E2E8F0')),
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, linecolor='#334155'),
        yaxis=dict(showgrid=True, gridcolor='#1E293B', linecolor='#334155')
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🎯 Inference State")
    
    # Model Prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction_idx = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    predicted_species = iris.target_names[prediction_idx].upper()
    confidence = probabilities[prediction_idx] * 100
    
    # Dark Neon Prediction Card
    st.markdown(f"""
        <div class="result-card">
            <div class="result-title">Predicted Taxonomy</div>
            <div class="result-value">{predicted_species}</div>
            <div class="result-confidence">CONFIDENCE: {confidence:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Class Probabilities")
    
    # Probability Distribution Bar Chart
    species_names = [name.upper() for name in iris.target_names]
    bar_colors = ['#6366F1' if i == prediction_idx else '#1E293B' for i in range(len(species_names))]
    border_colors = ['#818CF8' if i == prediction_idx else '#334155' for i in range(len(species_names))]
    
    fig_prob = go.Figure(data=[
        go.Bar(
            x=species_names,
            y=probabilities * 100,
            marker=dict(color=bar_colors, line=dict(color=border_colors, width=1.5)),
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition='auto',
            textfont=dict(color='#F8FAFC', family='monospace')
        )
    ])
    
    fig_prob.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=10, b=20),
        font=chart_font,
        yaxis=dict(title='Confidence (%)', range=[0, 105], showgrid=True, gridcolor='#1E293B', linecolor='#334155'),
        xaxis=dict(showgrid=False, linecolor='#334155'),
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_prob, use_container_width=True)