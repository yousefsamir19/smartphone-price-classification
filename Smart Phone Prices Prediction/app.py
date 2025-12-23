import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set Page Config
st.set_page_config(
    page_title="Smartphone Price Classifier",
    page_icon="📱",
    layout="wide"
)

#abd alkarem was here ;
st.markdown("""
    <style>
        .stApp {
            background-color: #1a0b2e;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)
#abd alkarem was here ;



# --- 1. Load Models, Encoders, and Scaler ---
@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:

        artifacts['oe'] = joblib.load('S:\Projects\Smart Phone Prices Prediction\smartphone-price-classification\Smart Phone Prices Prediction\oe.pkl')
        artifacts['te'] = joblib.load('S:\Projects\Smart Phone Prices Prediction\smartphone-price-classification\Smart Phone Prices Prediction\\te.pkl')
        artifacts['ohn'] = joblib.load('S:\Projects\Smart Phone Prices Prediction\smartphone-price-classification\Smart Phone Prices Prediction\ohn.pkl')
        artifacts['scaler'] = joblib.load('S:\Projects\Smart Phone Prices Prediction\smartphone-price-classification\Smart Phone Prices Prediction\scaler.pkl')

        artifacts['models'] = {
            'K-Nearest Neighbors': joblib.load('S:\Projects\Smart Phone Prices Prediction\smartphone-price-classification\Smart Phone Prices Prediction\knn.pkl'),
            'Decision Tree': joblib.load('S:\Projects\Smart Phone Prices Prediction\smartphone-price-classification\Smart Phone Prices Prediction\DT.pkl'),
            'Random Forest': joblib.load('S:\Projects\Smart Phone Prices Prediction\smartphone-price-classification\Smart Phone Prices Prediction\RF.pkl'),
            'Support Vector Machine': joblib.load('S:\Projects\Smart Phone Prices Prediction\smartphone-price-classification\Smart Phone Prices Prediction\SVM.pkl'),
            'XGBOOST': joblib.load('S:\Projects\Smart Phone Prices Prediction\smartphone-price-classification\Smart Phone Prices Prediction\XGB.pkl')
        }
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        return None
    return artifacts

artifacts = load_artifacts()

# --- 2. Preprocessing Functions ---
def clean_processor_series(val):
    val_str = str(val).strip()
    val_str = val_str.replace(" Gen1", ".1").replace(" Gen2", ".2").replace("Unknown", "35")
    try:
        return float(val_str)
    except:
        return 0.0

def preprocess_input(data, artifacts):
    df = pd.DataFrame([data])

    # Cleaning
    df['Processor_Series'] = df['Processor_Series'].apply(clean_processor_series)

    # Log Transformation
    log_cols = [
        'rating','Processor_Series','Core_Count','Clock_Speed_GHz',
        'RAM_Size_GB','Storage_Size_GB','fast_charging_power',
        'Screen_Size','Resolution_Width','Resolution_Height',
        'Refresh_Rate','primary_rear_camera_mp','primary_front_camera_mp',
        'num_front_cameras'
    ]
    for c in log_cols:
        if c in df.columns:
            df[c] = np.log1p(df[c].astype(float))

    # Binary Mapping
    bmap = {'Yes': 1, 'No': 0}
    df['5G'] = df['5G'].map(bmap)
    df['NFC'] = df['NFC'].map(bmap)

    # Ordinal Encoding
    temp_oe_df = pd.DataFrame({
        'Performance_Tier': ['unknown'],
        'RAM_Tier': df['RAM_Tier']
    })
    try:
        oe_out = artifacts['oe'].transform(temp_oe_df)
        df['RAM_Tier'] = oe_out[:, 1]
    except Exception:
        ram_map = {'unknown':0,'budget':1,'low-end':2,'mid-range':3,'high-end':4,'flagship':5}
        df['RAM_Tier'] = df['RAM_Tier'].map(ram_map).fillna(0)

    # Target Encoding
    df['brand'] = artifacts['te'].transform(df['brand'])

    # One-Hot Encoding
    one_hot_cols = ['Processor_Brand','Notch_Type','os_name']
    try:
        ohe_out = artifacts['ohn'].transform(df[one_hot_cols])
        ohe_cols = artifacts['ohn'].get_feature_names_out(one_hot_cols)
        ohe_df = pd.DataFrame(ohe_out, columns=ohe_cols, index=df.index)
        df = pd.concat([df, ohe_df], axis=1)
    except Exception as e:
        st.error(f"Encoding Error: {e}")
        return None

    # Feature Selection
    selected_features = [
        'Core_Count','RAM_Size_GB','primary_rear_camera_mp','primary_front_camera_mp','Processor_Series',
        'performance_score','Clock_Speed_GHz','Resolution_Height','camera_quality_score','NFC',
        'Resolution_Width','Storage_Size_GB','RAM_Size_GB','rating','primary_front_camera_mp',
        'fast_charging_power','Screen_Size','RAM_Tier','brand','Notch_Type_water drop notch',
        'Processor_Brand_helio','primary_rear_camera_mp','Refresh_Rate','5G'
    ]

    final_df = pd.DataFrame(np.zeros((1,len(selected_features))), columns=selected_features)
    for col in selected_features:
        if col in df.columns:
            final_df[col] = df[col].values[0]

    # Scaling
    scaler = artifacts['scaler']
    scaled_array = scaler.transform(final_df)
    #final_df_scaled = pd.DataFrame(scaled_array, columns=final_df.columns, index=final_df.index)

    return scaled_array

# --- 3. Streamlit UI ---
st.title("📱 Smartphone Price Classifier")
st.markdown("Predict if a smartphone is **Expensive** or **Budget-Friendly** based on its specs!")
st.write("---")

if artifacts:
    # Sidebar Model Selection
    st.sidebar.header("⚙️ Configuration")
    selected_model_name = st.sidebar.selectbox("Select Model", list(artifacts['models'].keys()))
    model = artifacts['models'][selected_model_name]

    # Input Form
    st.subheader("Enter Smartphone Specifications")
    col1, col2, col3 = st.columns(3)
    with col1:
        
        rating = st.number_input("Rating (0-5)")
        dual_sim = st.radio("Dual SIM?", ["Yes", "No"], horizontal=True)
        g_4 = st.radio("4G Support?", ["Yes", "No"], horizontal=True)
        has_5g = st.radio("5G Support?", ["Yes","No"], horizontal=True)
        vo5g = st.radio("Vo5G Support?", ["Yes", "No"], horizontal=True)
        nfc = st.radio("Has NFC?", ["Yes","No"], horizontal=True)
        ir_blaster = st.radio("IR Blaster?", ["Yes", "No"], horizontal=True)
        processor_brand = st.selectbox("Processor Brand", ["snapdragon","dimensity","helio","exynos","bionic","unisoc","other"])
        processor_series = st.text_input("Processor Series (e.g., 8 Gen 1)", "680")
        core_count = st.number_input("Core Count", 1, 16, 8)
        clock_speed = st.number_input("Clock Speed (GHz)", 0.1, 5.0, 2.4)
        ram_gb = st.number_input("RAM (GB)", 1, 64, 8)
        storage_gb = st.number_input("Storage (GB)", 16, 1024, 128)
    with col2:
        ram_tier = st.selectbox("RAM Tier", ["unknown","budget","low-end","mid-range","high-end","flagship"])
        battery = st.number_input("Battery Capacity (mAh)", 1000, 10000, 5000)
        fast_charge = st.number_input("Fast Charging (W)", value=18)
        screen_size = st.number_input("Screen Size (inches)", value=6.5)
        resolution_w = st.number_input("Resolution Width (px)", value=1080)
        resolution_h = st.number_input("Resolution Height (px)", value=2400)
        refresh_rate = st.number_input("Refresh Rate (Hz)", value=90)
    with col3:
        notch_type = st.selectbox("Notch Type", ["punch hole","water drop notch","small notch","no notch","unknown"])
        rear_cam = st.number_input("Primary Rear Camera (MP)", value=48)
        front_cam = st.number_input("Primary Front Camera (MP)", value=16)
        num_rear = st.number_input("Number of Rear Cameras", 1, 5, 3)
        num_front = st.number_input("Number of Front Cameras", 1, 5, 1)
    st.write("---")
    st.subheader("Additional Specifications")

    col7, col8 = st.columns(2)

    
    with col7:
        memory_card_support = st.number_input("Memory Card Support (Slots)", min_value=0, max_value=4, value=0)
        memory_card_size = st.number_input("Memory Card Size (GB)", min_value=0, max_value=1024, value=0)
        os_name = st.selectbox("OS Name", ["android","ios","Other"])
        os_version = st.text_input("OS Version", "Android 13")
        brand = st.selectbox("Brand", ["xiaomi","samsung","apple","vivo","oppo","realme","oneplus","motorola","Other"])

    st.write("---")
    st.subheader("Advanced Specs")
    col4, col5, col6 = st.columns(3)
    with col6:
        perf_score = st.number_input("Performance Score (Benchmark)", value=0.0)
        cam_score = st.number_input("Camera Quality Score", value=0.0)

    # Predict Button
    if st.button("Predict Price Category", type="primary"):
        user_data = {
            'rating': rating,
            'Core_Count': core_count,
            'Clock_Speed_GHz': clock_speed,
            'RAM_Size_GB': ram_gb,
            'Storage_Size_GB': storage_gb,
            'battery_capacity': battery,
            'fast_charging_power': fast_charge,
            'Screen_Size': screen_size,
            'Resolution_Width': resolution_w,
            'Resolution_Height': resolution_h,
            'Refresh_Rate': refresh_rate,
            'primary_rear_camera_mp': rear_cam,
            'num_rear_cameras': num_rear,
            'primary_front_camera_mp': front_cam,
            'num_front_cameras': num_front,
            'brand': brand,
            'Processor_Series': processor_series,
            'Processor_Brand': processor_brand,
            'Notch_Type': notch_type,
            'os_name': os_name,
            'RAM_Tier': ram_tier,
            '5G': has_5g,
            'NFC': nfc
            # 'performance_score': perf_score,
            # 'camera_quality_score': cam_score
        }
        user_data['performance_score'] = user_data['Core_Count'] * user_data['Clock_Speed_GHz'] * (user_data['RAM_Size_GB'] / 4)
        user_data['camera_quality_score']=(user_data['primary_rear_camera_mp'] * 0.7 + user_data['primary_front_camera_mp'] * 0.3)
        processed_data = preprocess_input(user_data, artifacts)
        if processed_data is not None:
            prediction = model.predict(processed_data)[0]
            st.write("---")
            if prediction == 1:
                st.markdown("""
                                               <div style="
                                                   background-color: #ff4b4b; 
                                                   color: white; 
                                                   padding: 15px; 
                                                   border-radius: 10px; 
                                                   text-align: center; 
                                                   font-size: 20px; 
                                                   font-weight: bold; 
                                                   border: 2px solid #bd2130;
                                                   width: 60%;
                                                   margin: 0 auto 10px auto;
                                               ">
                                                   (Expensive)
                                               </div>
                                               """, unsafe_allow_html=True)
            else:

                st.markdown("""
                                <div style="
                                    background-color: #ff4b4b; 
                                    color: white; 
                                    padding: 15px; 
                                    border-radius: 10px; 
                                    text-align: center; 
                                    font-size: 20px; 
                                    font-weight: bold; 
                                    border: 2px solid #bd2130;
                                    width: 60%; 
                                    margin: 0 auto 10px auto;
                                ">
                                    (Non-Expensive)
                                </div>
                                """, unsafe_allow_html=True)
else:
    st.warning("Artifacts could not be loaded. Please check your files.")
