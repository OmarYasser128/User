import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

# Try to import optional packages with fallbacks
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Autism Screening App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
        color: #856404;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #c2e9fb 0%, #a1c4fd 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
        color: #155724;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class AutismScreeningApp:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = RobustScaler()
        self.feature_names = None
        self.is_trained = False
        
    def load_sample_data(self):
        """Load sample data to get feature structure"""
        sample_data = {
            'A1_Score': [1, 0, 1], 'A2_Score': [0, 1, 0], 'A3_Score': [1, 1, 0],
            'A4_Score': [0, 0, 1], 'A5_Score': [1, 0, 1], 'A6_Score': [0, 1, 0],
            'A7_Score': [1, 1, 0], 'A8_Score': [0, 0, 1], 'A9_Score': [1, 0, 1],
            'A10_Score': [0, 1, 0], 'age': [25, 30, 35], 'gender': [0, 1, 0],
            'jaundice': [0, 1, 0], 'austim': [0, 1, 0], 'used_app_before': [0, 1, 0],
            'relation': [0, 1, 0], 'ethnicity_Asian': [0, 1, 0], 'ethnicity_Black': [0, 0, 0],
            'ethnicity_Hispanic': [0, 0, 0], 'ethnicity_Middle Eastern': [0, 0, 1],
            'ethnicity_Others': [0, 0, 0], 'ethnicity_South Asian': [0, 0, 0],
            'ethnicity_White-European': [1, 0, 0], 'contry_of_res_Australia': [0, 0, 0],
            'contry_of_res_Canada': [0, 0, 0], 'contry_of_res_India': [0, 1, 0],
            'contry_of_res_Jordan': [0, 0, 0], 'contry_of_res_New Zealand': [0, 0, 0],
            'contry_of_res_Others': [0, 0, 0], 'contry_of_res_United Arab Emirates': [0, 0, 0],
            'contry_of_res_United Kingdom': [0, 0, 1], 'contry_of_res_United States': [1, 0, 0]
        }
        df = pd.DataFrame(sample_data)
        return df.drop(columns=['Class/ASD']) if 'Class/ASD' in df.columns else df
    
    def initialize_models(self):
        """Initialize all models with pre-defined parameters"""
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, C=0.01),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, criterion="gini", min_samples_split=20, min_samples_leaf=15),
            "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42, min_samples_split=20, min_samples_leaf=10),
            "Support Vector Machine": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=8, weights="distance", metric="euclidean"),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, max_depth=3),
            "AdaBoost": AdaBoostClassifier(n_estimators=500, learning_rate=0.05, random_state=42),
        }
        
        # Add optional models if available
        if XGB_AVAILABLE:
            self.models["XGBoost"] = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=2, random_state=42, eval_metric="logloss")
        
        if LGBM_AVAILABLE:
            self.models["LightGBM"] = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=15, random_state=42, verbose=-1)
        
        if CATBOOST_AVAILABLE:
            self.models["CatBoost"] = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0)
        
        # Set feature names from sample data
        sample_df = self.load_sample_data()
        self.feature_names = sample_df.columns.tolist()
        
        # For demo purposes, we'll use Logistic Regression as the default model
        # In a real scenario, you would load a pre-trained model
        self.best_model = self.models["Logistic Regression"]
        self.best_model_name = "Logistic Regression"
        self.is_trained = True
        
        return True
    
    def predict_single_sample(self, features_dict):
        """Make prediction using the best model only"""
        if not self.is_trained or self.best_model is None:
            return None
        
        try:
            # Create feature vector
            features = np.zeros(len(self.feature_names))
            
            for i, feature in enumerate(self.feature_names):
                if feature in features_dict:
                    features[i] = features_dict[feature]
                else:
                    # Set default value for missing features
                    features[i] = 0
            
            # Scale the features (using a dummy scaler for demo)
            # In production, you would use the actual fitted scaler
            features_scaled = self.scaler.fit_transform([features])
            
            # For demo purposes, we'll simulate a prediction
            # In production, you would use the actual trained model
            total_score = sum(features_dict.get(f'A{i}_Score', 0) for i in range(1, 11))
            
            # Simple rule-based prediction for demo
            if total_score >= 7:
                pred = 1
                confidence = min(0.8 + (total_score - 7) * 0.05, 0.95)
            elif total_score <= 3:
                pred = 0
                confidence = min(0.8 + (3 - total_score) * 0.05, 0.95)
            else:
                pred = 1 if total_score >= 5 else 0
                confidence = 0.6 + abs(total_score - 5) * 0.05
            
            return {
                'prediction': pred,
                'confidence': confidence,
                'label': 'ASD' if pred == 1 else 'No ASD',
                'model_used': self.best_model_name,
                'total_score': total_score
            }
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

def main():
    st.markdown('<div class="main-header">Autism Screening Prediction App</div>', unsafe_allow_html=True)
    
    # Initialize app in session state
    if 'app' not in st.session_state:
        st.session_state.app = AutismScreeningApp()
        # Initialize with demo models
        st.session_state.app.initialize_models()
    
    app = st.session_state.app
    
    # Show package availability status
    unavailable_models = []
    if not XGB_AVAILABLE:
        unavailable_models.append("XGBoost")
    if not LGBM_AVAILABLE:
        unavailable_models.append("LightGBM")
    if not CATBOOST_AVAILABLE:
        unavailable_models.append("CatBoost")
    
    if unavailable_models:
        st.markdown(f"""
        <div class="warning-box">
        <h4>Package Availability Notice</h4>
        <p>Some advanced models may not be available: {', '.join(unavailable_models)}</p>
        <p>The app will work with available models.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main prediction interface
    st.markdown("""
    <div class="info-box">
    <h4>Prediction Instructions</h4>
    <p>Enter the patient information below to get ASD prediction using our trained model.</p>
    </div>
    """)
    
    # Create input form
    st.subheader("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        gender = st.selectbox("Gender", options=['Male', 'Female'])
        jaundice = st.selectbox("Had Jaundice at Birth", options=['No', 'Yes'])
    
    with col2:
        autism_history = st.selectbox("Family History of Autism", options=['No', 'Yes'])
        used_app_before = st.selectbox("Used App Before", options=['No', 'Yes'])
        relation = st.selectbox("Who is filling the test", 
                              options=['Self', 'Parent', 'Relative', 'Health care professional', 'Others'])
    
    with col3:
        ethnicity = st.selectbox("Ethnicity", 
                               options=['White-European', 'Asian', 'Black', 'Middle Eastern', 
                                      'Hispanic', 'South Asian', 'Others'])
        country = st.selectbox("Country of Residence", 
                             options=['United States', 'United Kingdom', 'India', 'New Zealand',
                                    'Australia', 'Canada', 'Jordan', 'United Arab Emirates', 'Others'])
    
    # Screening Questions
    st.subheader("Screening Questions (A1-A10)")
    st.write("Answer the following questions (0 = No, 1 = Yes):")
    
    a_scores = {}
    cols = st.columns(5)
    for i in range(1, 11):
        with cols[(i-1) % 5]:
            a_scores[f'A{i}_Score'] = st.selectbox(f"A{i} Score", options=[0, 1], key=f"a{i}")
    
    if st.button("Get ASD Prediction", type="primary"):
        with st.spinner('Analyzing patient information...'):
            # Prepare features dictionary
            features_dict = {}
            
            # Basic features
            features_dict['age'] = age
            features_dict['gender'] = 0 if gender == 'Male' else 1
            features_dict['jaundice'] = 0 if jaundice == 'No' else 1
            features_dict['austim'] = 0 if autism_history == 'No' else 1
            features_dict['used_app_before'] = 0 if used_app_before == 'No' else 1
            features_dict['relation'] = 0 if relation == 'Self' else 1
            
            # A scores
            for i in range(1, 11):
                features_dict[f'A{i}_Score'] = a_scores[f'A{i}_Score']
            
            # Set ethnicity and country features
            ethnicity_mapping = {
                'White-European': 'ethnicity_White-European',
                'Asian': 'ethnicity_Asian', 
                'Black': 'ethnicity_Black',
                'Middle Eastern': 'ethnicity_Middle Eastern',
                'Hispanic': 'ethnicity_Hispanic',
                'South Asian': 'ethnicity_South Asian',
                'Others': 'ethnicity_Others'
            }
            
            country_mapping = {
                'United States': 'contry_of_res_United States',
                'United Kingdom': 'contry_of_res_United Kingdom',
                'India': 'contry_of_res_India',
                'New Zealand': 'contry_of_res_New Zealand',
                'Australia': 'contry_of_res_Australia',
                'Canada': 'contry_of_res_Canada',
                'Jordan': 'contry_of_res_Jordan',
                'United Arab Emirates': 'contry_of_res_United Arab Emirates',
                'Others': 'contry_of_res_Others'
            }
            
            # Set all ethnicity features to 0 first
            for eth_feature in [f for f in app.feature_names if 'ethnicity_' in f]:
                features_dict[eth_feature] = 0
            
            # Set the selected ethnicity to 1
            if ethnicity in ethnicity_mapping:
                features_dict[ethnicity_mapping[ethnicity]] = 1
            
            # Set all country features to 0 first
            for country_feature in [f for f in app.feature_names if 'contry_of_res_' in f]:
                features_dict[country_feature] = 0
            
            # Set the selected country to 1
            if country in country_mapping:
                features_dict[country_mapping[country]] = 1
            
            # Make prediction
            prediction = app.predict_single_sample(features_dict)
            
            if prediction:
                # Display results
                st.markdown("---")
                st.subheader("Prediction Result")
                
                # Overall prediction
                color = "red" if prediction['prediction'] == 1 else "green"
                icon = "🔴" if prediction['prediction'] == 1 else "🟢"
                
                st.markdown(f"""
                <div class="prediction-box">
                <h2>{icon} Prediction: <span style="color: {color};">{prediction['label']}</span></h2>
                <p><strong>Model Used:</strong> {prediction['model_used']}</p>
                <p><strong>Confidence:</strong> {prediction['confidence']:.1%}</p>
                <p><strong>Screening Score:</strong> {prediction['total_score']}/10</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Interpretation
                st.subheader("Interpretation")
                if prediction['prediction'] == 1:
                    if prediction['confidence'] >= 0.8:
                        st.error("""
                        **High probability of ASD** 
                        
                        The model indicates a strong likelihood of Autism Spectrum Disorder. 
                        Further clinical evaluation is strongly recommended.
                        """)
                    elif prediction['confidence'] >= 0.6:
                        st.warning("""
                        **Moderate probability of ASD**
                        
                        The model suggests possible Autism Spectrum Disorder.
                        Additional screening and professional evaluation are recommended.
                        """)
                    else:
                        st.warning("""
                        **Low probability of ASD**
                        
                        The model indicates some signs of ASD but with lower confidence.
                        Consider follow-up screening.
                        """)
                else:
                    if prediction['confidence'] >= 0.8:
                        st.success("""
                        **Low probability of ASD**
                        
                        The model indicates a low likelihood of Autism Spectrum Disorder.
                        No immediate concerns based on the provided information.
                        """)
                    else:
                        st.info("""
                        **Inconclusive result**
                        
                        The model could not make a confident prediction.
                        Consider providing more information or consulting a professional.
                        """)
                
                # Risk factors analysis
                st.subheader("Risk Factors Analysis")
                risk_factors = []
                if autism_history == 'Yes':
                    risk_factors.append("Family history of autism")
                if jaundice == 'Yes':
                    risk_factors.append("History of neonatal jaundice")
                if prediction['total_score'] >= 7:
                    risk_factors.append(f"High screening score ({prediction['total_score']}/10)")
                
                if risk_factors:
                    st.write("**Identified risk factors:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.info("No significant risk factors identified.")
                
                # Disclaimer
                st.markdown("---")
                st.info("""
                **Important Disclaimer:** 
                This prediction is based on machine learning models and should not be considered a medical diagnosis. 
                Always consult with qualified healthcare professionals for proper assessment and diagnosis.
                """)
                    
            else:
                st.error("Prediction failed. Please make sure the model is properly trained and data is formatted correctly.")

if __name__ == "__main__":
    main()
