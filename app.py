"""
Waste Detection App using Streamlit and YOLO11
This app detects waste materials: Cardboard, Glass, Metal, Paper, and Plastic
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import cv2
import pandas as pd
from datetime import datetime
import random
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Smart Bin - AI Waste Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for analytics
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
    
if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now().strftime("%H:%M")

# Initialize session state for counters
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
    st.session_state.images_processed = 0

# Initialize session state for sample image button tracking
if 'use_sample_clicked' not in st.session_state:
    st.session_state.use_sample_clicked = False

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
    background: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), 
                url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e');
    background-size: cover;
    background-attachment: fixed;
    color: #1e293b;
    font-family: 'Inter', sans-serif;
}
    
    /* Make cards look like Frosted Glass */
    .stats-card, .detection-card, [data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 20px !important;
}
    /* Main container */
    .main-header {
    background: linear-gradient(135deg, rgba(19, 78, 74, 0.9) 0%, rgba(6, 95, 70, 0.8) 100%) !important;
        padding: 2.5rem 3.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
}
    
    .main-title {
        color: #f0fdf4 !important;
        font-size: 4rem !important;
        font-weight: 800 !important;
        letter-spacing: -2px !important;
        margin: 0;
        line-height: 1;
    }
    
    .main-subtitle {
        color: #34d399 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    
    /* The 'For a cleaner planet' Box */
    .eco-badge-container {
        margin-left: auto;
        padding: 1rem 2rem;
        border-left: 1px solid rgba(255, 255, 255, 0.2); /* Creates a nice separator */
        text-align: right;
    }

    .eco-badge-container:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(1.05);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.02);
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        padding: 2rem 1rem;
    }
    
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Cards */
    .stats-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 1rem;
    }
    
    .stats-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.05);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1.2;
    }
    
    .stats-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    
    /* Category badges */
    .category-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.25rem;
        border: 1px solid #e2e8f0;
        background: white;
        color: #1e293b;
    }
    
    .category-badge:hover {
        background: #f8fafc;
    }
    
    /* Detection results */
    .detection-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    .confidence-bar {
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s;
        border: 1px solid transparent;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        border: 2px dashed #e2e8f0;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #64748b !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Success/Info/Warning boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: white;
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
    
    /* Live indicator */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        background: #fee2e2;
        color: #dc2626;
        padding: 0.25rem 1rem;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #fecaca;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: #dc2626;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.1); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Camera selection */
    .camera-option {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .camera-option:hover {
        border-color: #667eea;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.1);
    }
    
    .camera-option.selected {
        border-color: #667eea;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: #0f172a;
        }
        
        [data-testid="stSidebar"] {
            background: #1e293b;
            border-right-color: #334155;
        }
        
        .sidebar-header {
            color: #f1f5f9;
            border-bottom-color: #334155;
        }
        
        .stats-card {
            background: #1e293b;
            border-color: #334155;
        }
        
        .stats-number {
            color: #f1f5f9;
        }
        
        .category-badge {
            background: #1e293b;
            border-color: #334155;
            color: #f1f5f9;
        }
        
        .detection-card {
            background: #1e293b;
            border-color: #334155;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #818cf8, #a78bfa);
        }
        
        [data-testid="stMetricValue"] {
            color: #f1f5f9 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Model path - use best trained model
MODEL_PATH = "my_model/train/weights/best.pt"

# Class names and their properties
CLASSES = {
    'Cardboard': {'emoji': '📦', 'color': '#F97316', 'bg': '#FFF3E6'},
    'Glass': {'emoji': '🫙', 'color': '#0EA5E9', 'bg': '#E6F3FF'},
    'Metal': {'emoji': '🥫', 'color': '#6B7280', 'bg': '#F3F4F6'},
    'Paper': {'emoji': '📄', 'color': '#10B981', 'bg': '#E6F9F0'},
    'Plastic': {'emoji': '🧴', 'color': '#8B5CF6', 'bg': '#F3E8FF'}
}

# Detection colors for OpenCV (BGR format)
CLASS_COLORS_BGR = {
    'Cardboard': (0, 165, 255),    # Orange
    'Glass': (255, 165, 0),        # Blue
    'Metal': (128, 128, 128),      # Gray
    'Paper': (0, 255, 0),          # Green
    'Plastic': (255, 0, 255),      # Purple
}
# Real-world impact data per material
IMPACT_DATA = {
    'Cardboard': {
        'decompose': '2 months',
        'energy_saved': 'Powers a 100W bulb for 24h',
        'co2_saved': 0.8,
        'fact': 'Recycling cardboard saves 24% of the energy needed for new cardboard.',
        'impact_math': '1 ton saves 17 trees'
    },
    'Glass': {
        'decompose': '1 million years',
        'energy_saved': 'Powers a laptop for 30 mins',
        'co2_saved': 0.3,
        'fact': 'Glass never wears out—it can be recycled an infinite number of times.',
        'impact_math': 'Melts at lower temp than raw sand'
    },
    'Metal': {
        'decompose': '200 years',
        'energy_saved': 'Powers a TV for 3 hours',
        'co2_saved': 9.0, 
        'fact': 'Recycling 1 aluminum can saves 95% of the energy to make a new one.',
        'impact_math': '1 can = 20h of a 100W lightbulb'
    },
    'Paper': {
        'decompose': '6 weeks',
        'energy_saved': 'Powers a microwave for 15 mins',
        'co2_saved': 1.5,
        'fact': 'Recycling paper uses 70% less energy than making it from trees.',
        'impact_math': 'Saves 7,000 gallons of water per ton'
    },
    'Plastic': {
        'decompose': '450 years',
        'energy_saved': 'Powers a 60W bulb for 3 hours',
        'co2_saved': 1.5,
        'fact': 'Plastic bottles in landfills can take 1,000 years to decompose without sunlight.',
        'impact_math': '1 ton saves 3.8 barrels of oil'
    }
}


# 7 Types of Plastic Wastes
PLASTIC_TYPES = {
    'PETE': {
        'name': 'PETE (Polyethylene Terephthalate)',
        'emoji': '🥤',
        'color': '#E11D48',
        'examples': 'Water bottles, food containers, polyester fabric',
        'recycling': 'Widely recycled'
    },
    'HDPE': {
        'name': 'HDPE (High-Density Polyethylene)',
        'emoji': '🧴',
        'color': '#2563EB',
        'examples': 'Milk jugs, detergent bottles, pipe materials',
        'recycling': 'Widely recycled'
    },
    'PVC': {
        'name': 'PVC (Polyvinyl Chloride)',
        'emoji': '🔧',
        'color': '#7C3AED',
        'examples': 'Pipes, window frames, flooring, cable insulation',
        'recycling': 'Limited recycling'
    },
    'LDPE': {
        'name': 'LDPE (Low-Density Polyethylene)',
        'emoji': '🛍️',
        'color': '#059669',
        'examples': 'Plastic bags, squeeze bottles, shrink wraps',
        'recycling': 'Partially recycled'
    },
    'PP': {
        'name': 'PP (Polypropylene)',
        'emoji': '🥣',
        'color': '#EA580C',
        'examples': 'Yogurt containers, bottle caps, food containers',
        'recycling': 'Widely recycled'
    },
    'PS': {
        'name': 'PS (Polystyrene)',
        'emoji': '📦',
        'color': '#475569',
        'examples': 'Styrofoam, disposable plates, egg cartons',
        'recycling': 'Difficult to recycle'
    },
    'OTHER': {
        'name': 'Other Plastics',
        'emoji': '🔄',
        'color': '#6B7280',
        'examples': 'Mixed plastics, polycarbonate, bioplastics',
        'recycling': 'Rarely recycled'
    }
}


@st.cache_resource
def load_model():
    """Load the YOLO model with caching"""
    try:
        with st.spinner("🚀 Loading AI model..."):
            model = YOLO(MODEL_PATH, task='detect')
            model.to('cpu')
            # Warm up
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(dummy_img, verbose=False)
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def draw_boxes(image, results):
    """Draw bounding boxes on image"""
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_copy = img_array.copy()
    boxes = results[0].boxes
    detection_data = []
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        class_name = list(CLASSES.keys())[cls_id]
        color = CLASS_COLORS_BGR.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label = f"{class_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Semi-transparent background
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (x1, y1 - 25), (x1 + label_size[0] + 10, y1), color, -1)
        cv2.addWeighted(overlay, 0.8, img_copy, 0.2, 0, img_copy)
        
        # Label text
        cv2.putText(img_copy, label, (x1 + 5, y1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        detection_data.append({
            'class': class_name,
            'confidence': conf,
            'bbox': (x1, y1, x2, y2)
        })
    
    return img_copy, detection_data

def process_frame(frame, model, confidence_threshold):
    """Process a single frame for detection"""
    pil_image = Image.fromarray(frame)
    results = model.predict(
        pil_image,
        conf=confidence_threshold,
        verbose=False,
        imgsz=640,
        iou=0.45,
        max_det=10
    )
    return draw_boxes(pil_image, results)

class VideoProcessor:
    def __init__(self, model, confidence_threshold):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        self.last_result = None
        self.last_detections = None
    
    def recv(self, frame):
        self.frame_count += 1
        import av
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result_img, detections = process_frame(img_rgb, self.model, self.confidence_threshold)
        
        self.last_result = result_img
        self.last_detections = detections
        
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
        # Add overlay with detection info
        if detections:
            h, w = result_bgr.shape[:2]
            
            # Semi-transparent background
            overlay = result_bgr.copy()
            cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, result_bgr, 0.7, 0, result_bgr)
            
            # Detection count
            cv2.putText(result_bgr, f"Detections: {len(detections)}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Class counts
            class_counts = {}
            for d in detections:
                class_counts[d['class']] = class_counts.get(d['class'], 0) + 1
            
            y_pos = 70
            for cls, count in class_counts.items():
                cv2.putText(result_bgr, f"{cls}: {count}", 
                           (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_pos += 25
        
        return av.VideoFrame.from_ndarray(result_bgr, format="bgr24")

# ==================== ANALYTICS FUNCTIONS ====================
def update_analytics(detections, source):
    """Update detection history with new data"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    for det in detections:
        st.session_state.detection_history.append({
            'timestamp': timestamp,
            'material': det['class'],
            'confidence': det['confidence'],
            'source': source  # 'upload' or 'webcam'
        })

def show_analytics_dashboard():
    """Display analytics dashboard in sidebar"""
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Analytics Dashboard")
    
    if not st.session_state.detection_history:
        st.sidebar.info("No data yet. Start detecting to see analytics!")
        return
    
    # Create DataFrame from history
    df = pd.DataFrame(st.session_state.detection_history)
    
    # Dashboard tabs in sidebar
    tab1, tab2, tab3 = st.sidebar.tabs(["📈 Stats", "🥧 Materials", "📅 Today"])
    
    with tab1:
        # Key metrics
        total_detections = len(df)
        unique_materials = df['material'].nunique()
        avg_confidence = df['confidence'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Detections", total_detections)
        with col2:
            st.metric("Unique Types", unique_materials)
        
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Detection trend (last 10 detections)
        recent = df.tail(10)
        if len(recent) > 1:
            fig = px.line(
                recent, 
                x=range(len(recent)), 
                y='confidence',
                title='Recent Confidence Trend',
                labels={'x': 'Detection #', 'y': 'Confidence'}
            )
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Material distribution pie chart
        material_counts = df['material'].value_counts().reset_index()
        material_counts.columns = ['material', 'count']
        
        # Add emojis to labels
        emoji_map = {
            'Cardboard': '📦',
            'Glass': '🫙',
            'Metal': '🥫',
            'Paper': '📄',
            'Plastic': '🧴'
        }
        material_counts['label'] = material_counts['material'].map(
            lambda x: f"{emoji_map.get(x, '♻️')} {x}"
        )
        
        fig = px.pie(
            material_counts,
            values='count',
            names='label',
            title='Materials Detected',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Most common material
        top_material = material_counts.iloc[0]['material']
        top_count = material_counts.iloc[0]['count']
        st.success(f"🏆 Most detected: **{top_material}** ({top_count} times)")
    
    with tab3:
        # Today's activity
        st.markdown(f"**Session started:** {st.session_state.session_start}")
        st.markdown(f"**Current time:** {datetime.now().strftime('%H:%M')}")
        
        # Detection rate
        session_minutes = max(1, (datetime.now() - datetime.strptime(st.session_state.session_start, "%H:%M")).seconds / 60)
        rate = total_detections / session_minutes
        st.metric("Detection Rate", f"{rate:.1f}/min")
        
        # Source breakdown
        source_counts = df['source'].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            upload_count = source_counts.get('upload', 0)
            st.markdown(f"📤 Upload: **{upload_count}**")
        with col2:
            webcam_count = source_counts.get('webcam', 0)
            st.markdown(f"📹 Webcam: **{webcam_count}**")
        
        # Reset button
        if st.button("🔄 Reset Today's Data"):
            st.session_state.detection_history = []
            st.session_state.session_start = datetime.now().strftime("%H:%M")
            st.rerun()
    
    # Global Waste Ticker (keep this inside the function)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌏 Global Waste Ticker")
    
    # Global average: ~67 tons of waste produced per second
    start_dt = datetime.strptime(st.session_state.session_start, "%H:%M")
    now_dt = datetime.now()
    # Normalize start_dt to today's date for subtraction
    start_dt = now_dt.replace(hour=start_dt.hour, minute=start_dt.minute, second=0)
    
    seconds_active = (now_dt - start_dt).seconds
    global_tons = seconds_active * 67
    
    st.sidebar.metric("Global Waste (Since Login)", f"{global_tons:,} Tons", delta="67 t/s", delta_color="inverse")
    
    # Session Savings
    session_co2 = calculate_session_impact()
    st.sidebar.metric("Your CO2 Savings", f"{session_co2:.3f} kg", delta="Personal Impact")

# 👇👇👇 MOVE THIS FUNCTION OUTSIDE (after show_analytics_dashboard ends) 👇👇👇
def calculate_session_impact():
    """Calculates total environmental savings based on session history"""
    total_co2 = 0
    if not st.session_state.detection_history:
        return 0
    
    for entry in st.session_state.detection_history:
        material = entry['material']
        if material in IMPACT_DATA:
            # Assuming average weight of 0.1kg per detected item for simulation
            total_co2 += IMPACT_DATA[material]['co2_saved'] * 0.1
            
    return total_co2

def main():
    # Header
    
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div class="main-subtitle">AI Waste Management</div>
                <div class="main-title">Smart Bin</div>
                <div style="color: #d1fae5; font-size: 1.1rem; font-weight: 300; margin-top: 5px; opacity: 0.8;">
                    Detection & Classification Engine
                </div>
            </div>
            <div class="eco-badge-container" style="display: flex; align-items: center; justify-content: center;">
                <div style="color: white; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 2px;">
                    For a cleaner planet
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check the model path.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ Controls</div>', unsafe_allow_html=True)
        
        # Detection Mode
        st.markdown("### 📸 Detection Mode")
        detection_mode = st.radio(
            "Select mode:",
            ["📤 Upload Image", "📹 Live Camera"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Confidence Threshold
        st.markdown("### 🎯 Confidence Threshold")
        confidence_threshold = st.slider(
            "Adjust sensitivity",
            min_value=0.05,
            max_value=1.0,
            value=0.15,
            step=0.05,
            help="Lower values detect more objects but may include false positives"
        )
        
        # Confidence indicator
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("Low")
        with col2:
            st.markdown("Medium")
        with col3:
            st.markdown("High")
        
        st.markdown("---")
        
        # Waste Categories
        st.markdown("### 🏷️ Waste Categories")
        for cls, props in CLASSES.items():
            st.markdown(f"""
            <div class="category-badge">
                <span style="margin-right: 8px; font-size: 1.2rem;">{props['emoji']}</span>
                <span>{cls}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Stats
        st.markdown("### 📊 Session Stats")
        if 'total_detections' not in st.session_state:
            st.session_state.total_detections = 0
            st.session_state.images_processed = 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Images", st.session_state.images_processed)
        with col2:
            st.metric("Detections", st.session_state.total_detections)
        
        # Add analytics dashboard to sidebar
        show_analytics_dashboard()
    
    # Main content
    if "📤 Upload Image" in detection_mode:
        # Image Upload Mode
        st.markdown("## 📤 Upload Image for Detection")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Upload an image containing waste materials"
        )
        
                # Sample images
        with st.expander("📁 Or try a sample image"):
            sample_images = {
                "📦 Cardboard Box": "cardboard/cardboard1.jpg",
                "🧴 Plastic Bottle": "plastic/1139e19349ea0ddf2664805706506c9a.jpg",
                "🫙 Glass Bottle": "glass/glass1.jpg"
            }
            selected_sample = st.selectbox("Select sample", list(sample_images.keys()), key="sample_selector")
            
            # Create two columns for buttons to avoid double-click issues
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                sample_clicked = st.button("📋 Load Sample", key="load_sample_btn")
            with col_s2:
                if st.button("🔄 Clear Sample", key="clear_sample_btn"):
                    st.session_state.use_sample_clicked = False
                    st.session_state.current_detection = None
                    st.session_state.last_processed_file = None
                    st.rerun()
        
        # Check if we should load sample image using session state
        should_load_sample = st.session_state.get('use_sample_clicked', False) or sample_clicked
        
        # If sample button was just clicked, set the flag
        if sample_clicked:
            st.session_state.use_sample_clicked = True
        
        # Initialize session state for storing current detection results
        if 'current_detection' not in st.session_state:
            st.session_state.current_detection = None
        
        # Track the last processed file to avoid re-processing
        if 'last_processed_file' not in st.session_state:
            st.session_state.last_processed_file = None
        
        # Determine what file we're dealing with - prioritize sample over upload
        current_file = uploaded_file.name if uploaded_file else None
        
        # Check if we need to process a new image
        need_new_processing = False
        
        # Track what we just processed to avoid double processing
        just_processed_sample = should_load_sample and st.session_state.get('sample_just_loaded', False)
        
        # Case 1: Sample was just clicked - only process sample, ignore uploaded file
        if should_load_sample and not just_processed_sample:
            if st.session_state.last_processed_file != f"sample:{selected_sample}":
                need_new_processing = True
        
        # Case 2: File was uploaded and it's different from last processed (and not just processed a sample)
        elif uploaded_file is not None and current_file != st.session_state.last_processed_file and not st.session_state.get('sample_just_loaded', False):
            need_new_processing = True
        
        # Clear the sample_just_loaded flag after checking
        if st.session_state.get('sample_just_loaded', False):
            st.session_state.sample_just_loaded = False
        
        if need_new_processing:
            # Reset the sample click flag after using it
            if should_load_sample:
                st.session_state.use_sample_clicked = False
                st.session_state.sample_just_loaded = True
            
            # Load image - prioritize sample if clicked
            if should_load_sample:
                try:
                    image = Image.open(sample_images[selected_sample])
                    image_name = selected_sample
                    st.session_state.last_processed_file = f"sample:{selected_sample}"
                except FileNotFoundError:
                    st.error(f"Sample image not found: {sample_images[selected_sample]}")
                    image = None
            elif uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_name = uploaded_file.name
                st.session_state.last_processed_file = uploaded_file.name
            else:
                image = None
            
            if image is not None:
                # Process image
                with st.spinner("🔍 Analyzing image..."):
                    results = model.predict(image, conf=confidence_threshold, verbose=False)
                    result_image, detections = draw_boxes(image, results)
                    
                    # Update analytics only if there are detections
                    if detections:
                        update_analytics(detections, 'upload')
                        # Update session stats
                        st.session_state.total_detections += len(detections)
                    
                    # Update images processed count
                    st.session_state.images_processed += 1
                    
                    # Store current detection results for display after rerun
                    st.session_state.current_detection = {
                        'image': image,
                        'result_image': result_image,
                        'detections': detections,
                        'image_name': image_name
                    }
                    
                    # Force rerun to update sidebar stats
                    st.rerun()
        
        # Display detection results from session state if available
        if st.session_state.current_detection is not None:
            detection_data = st.session_state.current_detection
            image = detection_data['image']
            result_image = detection_data['result_image']
            detections = detection_data['detections']
            image_name = detection_data['image_name']
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📷 Original")
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("### 🔍 Detection Result")
                st.image(result_image, use_container_width=True)
            
            # Detection stats
            if detections:
                st.markdown("## 📊 Detection Results")
                
                # Summary cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Objects", len(detections))
                
                with col2:
                    unique_classes = len(set([d['class'] for d in detections]))
                    st.metric("Categories", unique_classes)
                
                with col3:
                    avg_conf = sum([d['confidence'] for d in detections]) / len(detections)
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")
                
                with col4:
                    main_material = max(set([d['class'] for d in detections]), 
                                       key=lambda x: [d['class'] for d in detections].count(x))
                    st.metric("Main Material", main_material)
                
                # Plastic Subtype Selector - Only show when Plastic is detected
                if main_material == 'Plastic':
                    st.markdown("### 🧴 Select Plastic Type")
                    st.markdown("*Please identify the specific type of plastic for proper recycling:*")
                    
                    # Initialize session state for plastic type if not exists
                    if 'selected_plastic_type' not in st.session_state:
                        st.session_state.selected_plastic_type = 'PETE'
                    
                    # Create plastic type options for dropdown
                    plastic_options = [f"{PLASTIC_TYPES[pt]['emoji']} {PLASTIC_TYPES[pt]['name']}" for pt in PLASTIC_TYPES.keys()]
                    plastic_keys = list(PLASTIC_TYPES.keys())
                    
                    # Show plastic type selector
                    selected_plastic = st.selectbox(
                        "Choose Plastic Type:",
                        plastic_options,
                        index=plastic_keys.index(st.session_state.selected_plastic_type) if st.session_state.selected_plastic_type in plastic_keys else 0,
                        key="plastic_type_selector"
                    )
                    
                    # Get selected plastic type key
                    selected_idx = plastic_options.index(selected_plastic)
                    selected_plastic_key = plastic_keys[selected_idx]
                    st.session_state.selected_plastic_type = selected_plastic_key
                    
                    # Display selected plastic type info
                    plastic_info = PLASTIC_TYPES[selected_plastic_key]
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {plastic_info['color']}20 0%, {plastic_info['color']}10 100%); 
                                border: 2px solid {plastic_info['color']}; 
                                border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                            <span style="font-size: 2rem;">{plastic_info['emoji']}</span>
                            <span style="font-size: 1.2rem; font-weight: 700; color: {plastic_info['color']};">{plastic_info['name']}</span>
                        </div>
                        <div style="font-size: 0.9rem; color: #374151; margin-bottom: 0.5rem;">
                            <strong>Common Items:</strong> {plastic_info['examples']}
                        </div>
                        <div style="font-size: 0.85rem; color: #6B7280;">
                            <strong>♻️ Recycling:</strong> {plastic_info['recycling']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed breakdown
                st.markdown("### 📋 Detailed Breakdown")
                
                # Create a DataFrame for display
                df_data = []
                class_counts = {}
                
                for det in detections:
                    cls = det['class']
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                    df_data.append({
                        'Material': f"{CLASSES[cls]['emoji']} {cls}",
                        'Confidence': f"{det['confidence']:.1%}",
                        'Position': f"({det['bbox'][0]}, {det['bbox'][1]})"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Distribution visualization
                st.markdown("### 📈 Distribution")
                
                for cls, count in class_counts.items():
                    percentage = (count / len(detections)) * 100
                    props = CLASSES[cls]
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"{props['emoji']} **{cls}**")
                    with col2:
                        st.markdown(f"""
                        <div style="background: {props['bg']}; border-radius: 20px; padding: 2px;">
                            <div style="background: {props['color']}; width: {percentage}%; height: 20px; 
                                      border-radius: 20px; display: flex; align-items: center; 
                                      justify-content: flex-end; padding-right: 10px; color: white; 
                                      font-size: 0.8rem; font-weight: 600;">
                                {count} ({percentage:.0f}%)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # --- NEW: ENVIRONMENTAL IMPACT SECTION ---
                st.markdown(f"## 🌍 Environmental Impact: {main_material}")
                
                impact = IMPACT_DATA.get(main_material, {})
                
                # Impact Cards
                i_col1, i_col2, i_col3 = st.columns(3)
                with i_col1:
                    st.markdown(f"""
                        <div style="background: rgba(220, 38, 38, 0.05); border: 1px solid rgba(220, 38, 38, 0.2); padding: 1.5rem; border-radius: 20px; text-align: center; backdrop-filter: blur(5px);">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">⌛</div>
                            <div style="font-size: 0.75rem; color: #991b1b; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Nature's Clock</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #dc2626; margin: 0.5rem 0;">{impact.get('decompose', 'N/A')}</div>
                            <div style="font-size: 0.8rem; color: #b91c1c;">to decompose</div>
                        </div>""", unsafe_allow_html=True)
                with i_col2:
                    st.markdown(f"""
                        <div style="background: rgba(5, 150, 105, 0.05); border: 1px solid rgba(5, 150, 105, 0.2); padding: 1.5rem; border-radius: 20px; text-align: center; backdrop-filter: blur(5px);">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                            <div style="font-size: 0.75rem; color: #064e3b; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Energy Reclaimed</div>
                            <div style="font-size: 1.2rem; font-weight: 800; color: #059669; margin: 0.5rem 0; min-height: 3.6rem; display: flex; align-items: center; justify-content: center;">{impact.get('energy_saved', 'N/A')}</div>
                            <div style="font-size: 0.8rem; color: #047857;">saved by recycling</div>
                        </div>""", unsafe_allow_html=True)
                with i_col3:
                    st.markdown(f"""
                        <div style="background: rgba(37, 99, 235, 0.05); border: 1px solid rgba(37, 99, 235, 0.2); padding: 1.5rem; border-radius: 20px; text-align: center; backdrop-filter: blur(5px);">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌳</div>
                            <div style="font-size: 0.75rem; color: #1e3a8a; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Carbon Offset</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #2563eb; margin: 0.5rem 0;">{impact.get('co2_saved', 0)} kg</div>
                            <div style="font-size: 0.8rem; color: #1d4ed8;">per unit saved</div>
                        </div>""", unsafe_allow_html=True)
                
                st.info(f"💡 **Did you know?** {impact.get('fact', '')}")
            else:
                st.warning("🤔 No waste materials detected. Try adjusting the confidence threshold or using a different image.")
    
    else:
        # Live Camera Mode
        st.markdown("## 📹 Live Camera Detection")
        
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            import av
            
            # Camera settings
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown("""
                <div class="stats-card">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <span style="font-size: 2rem;">🎥</span>
                        <div>
                            <div style="font-size: 0.9rem; color: #64748b;">Camera Status</div>
                            <div style="font-size: 1.2rem; font-weight: 600;">Ready</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="stats-card">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <span style="font-size: 2rem;">⚡</span>
                        <div>
                            <div style="font-size: 0.9rem; color: #64748b;">Processing</div>
                            <div style="font-size: 1.2rem; font-weight: 600;">Real-time</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="live-indicator">
                    <span class="live-dot"></span>
                    <span>LIVE</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Camera selection
            st.markdown("### 📱 Camera Selection")
            camera_facing = st.radio(
                "Choose camera:",
                ["Front", "Back", "Auto"],
                horizontal=True,
                help="Select which camera to use"
            )
            
            facing_mode = {
                "Front": "user",
                "Back": "environment",
                "Auto": ""
            }[camera_facing]
            
            st.markdown("---")
            
            # WebRTC streamer
            webrtc_ctx = webrtc_streamer(
                key="waste-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: VideoProcessor(model, confidence_threshold),
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 1280},
                        "height": {"ideal": 720},
                        "facingMode": facing_mode if facing_mode else {"ideal": "environment"}
                    },
                    "audio": False
                },
                async_processing=True,
            )
            
            # ===== CAPTURE IMAGE FUNCTIONALITY =====
            # Initialize session state for captured frame
            if 'captured_frame' not in st.session_state:
                st.session_state.captured_frame = None
            if 'captured_detections' not in st.session_state:
                st.session_state.captured_detections = None
            
            # Capture button and display section
            col_cap1, col_cap2 = st.columns([3, 1])
            
            with col_cap1:
                st.markdown("### 📸 Capture Image")
                st.markdown("Click the button below to capture the current frame and get detailed detection results!")
            
            with col_cap2:
                # Button to capture current frame
                if st.button("📸 Capture & Analyze", use_container_width=True):
                    if hasattr(webrtc_ctx, "video_processor") and webrtc_ctx.video_processor is not None:
                        # Get the last processed frame and detections
                        if webrtc_ctx.video_processor.last_result is not None:
                            st.session_state.captured_frame = webrtc_ctx.video_processor.last_result
                            st.session_state.captured_detections = webrtc_ctx.video_processor.last_detections
                            st.rerun()
                        else:
                            st.warning("No frame available yet. Please wait for the camera to start detecting.")
                    else:
                        st.warning("Camera not ready. Please start the camera first.")
            
            # Clear captured results button
            if st.session_state.captured_frame is not None:
                if st.button("🔄 Clear Capture", use_container_width=True):
                    st.session_state.captured_frame = None
                    st.session_state.captured_detections = None
                    st.rerun()
                
                # ===== DISPLAY CAPTURED IMAGE DETECTION RESULTS =====
                st.markdown("---")
                st.markdown("## 📊 Detection Results (Captured Image)")
                
                captured_detections = st.session_state.captured_detections
                captured_frame = st.session_state.captured_frame
                
                if captured_detections:
                    # Summary cards - TOTAL OBJECTS, CATEGORIES, AVG CONFIDENCE, MAIN MATERIAL
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("TOTAL OBJECTS", len(captured_detections))
                    
                    with col2:
                        unique_classes = len(set([d['class'] for d in captured_detections]))
                        st.metric("CATEGORIES", unique_classes)
                    
                    with col3:
                        avg_conf = sum([d['confidence'] for d in captured_detections]) / len(captured_detections)
                        st.metric("AVG CONFIDENCE", f"{avg_conf:.1%}")
                    
                    with col4:
                        main_material = max(set([d['class'] for d in captured_detections]), 
                                           key=lambda x: [d['class'] for d in captured_detections].count(x))
                        st.metric("MAIN MATERIAL", main_material)
                    
                    # Plastic Subtype Selector for Webcam Capture - Only show when Plastic is detected
                    if main_material == 'Plastic':
                        st.markdown("### 🧴 Select Plastic Type")
                        st.markdown("*Please identify the specific type of plastic for proper recycling:*")
                        
                        # Create plastic type options for dropdown
                        plastic_options = [f"{PLASTIC_TYPES[pt]['emoji']} {PLASTIC_TYPES[pt]['name']}" for pt in PLASTIC_TYPES.keys()]
                        plastic_keys = list(PLASTIC_TYPES.keys())
                        
                        # Show plastic type selector
                        selected_plastic = st.selectbox(
                            "Choose Plastic Type:",
                            plastic_options,
                            index=0,
                            key="webcam_plastic_type_selector"
                        )
                        
                        # Get selected plastic type key
                        selected_idx = plastic_options.index(selected_plastic)
                        selected_plastic_key = plastic_keys[selected_idx]
                        
                        # Display selected plastic type info
                        plastic_info = PLASTIC_TYPES[selected_plastic_key]
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {plastic_info['color']}20 0%, {plastic_info['color']}10 100%); 
                                    border: 2px solid {plastic_info['color']}; 
                                    border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
                            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                                <span style="font-size: 2rem;">{plastic_info['emoji']}</span>
                                <span style="font-size: 1.2rem; font-weight: 700; color: {plastic_info['color']};">{plastic_info['name']}</span>
                            </div>
                            <div style="font-size: 0.9rem; color: #374151; margin-bottom: 0.5rem;">
                                <strong>Common Items:</strong> {plastic_info['examples']}
                            </div>
                            <div style="font-size: 0.85rem; color: #6B7280;">
                                <strong>♻️ Recycling:</strong> {plastic_info['recycling']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display captured image
                    col_img1, col_img2 = st.columns(2)
                    
                    with col_img1:
                        st.markdown("### 📷 Captured Frame")
                        st.image(captured_frame, use_container_width=True, channels="RGB")
                    
                    with col_img2:
                        st.markdown("### 🔍 Detection Overlay")
                        st.image(captured_frame, use_container_width=True, channels="RGB")
                    
                    # ===== DETAILED BREAKDOWN =====
                    st.markdown("### 📋 Detailed Breakdown")
                    
                    # Create a DataFrame for display
                    df_data = []
                    class_counts = {}
                    
                    for det in captured_detections:
                        cls = det['class']
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                        df_data.append({
                            'Material': f"{CLASSES[cls]['emoji']} {cls}",
                            'Confidence': f"{det['confidence']:.1%}",
                            'Position': f"({det['bbox'][0]}, {det['bbox'][1]})"
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # ===== DISTRIBUTION =====
                    st.markdown("### 📈 Distribution")
                    
                    for cls, count in class_counts.items():
                        percentage = (count / len(captured_detections)) * 100
                        props = CLASSES[cls]
                        
                        col_d1, col_d2 = st.columns([1, 4])
                        with col_d1:
                            st.markdown(f"{props['emoji']} **{cls}**")
                        with col_d2:
                            st.markdown(f"""
                            <div style="background: {props['bg']}; border-radius: 20px; padding: 2px;">
                                <div style="background: {props['color']}; width: {percentage}%; height: 20px; 
                                          border-radius: 20px; display: flex; align-items: center; 
                                          justify-content: flex-end; padding-right: 10px; color: white; 
                                          font-size: 0.8rem; font-weight: 600;">
                                    {count} ({percentage:.0f}%)
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # ===== ENVIRONMENTAL IMPACT =====
                    st.markdown(f"## 🌍 Environmental Impact: {main_material}")
                    
                    impact = IMPACT_DATA.get(main_material, {})
                    
                    # Impact Cards
                    i_col1, i_col2, i_col3 = st.columns(3)
                    with i_col1:
                        st.markdown(f"""
                            <div style="background: rgba(220, 38, 38, 0.05); border: 1px solid rgba(220, 38, 38, 0.2); padding: 1.5rem; border-radius: 20px; text-align: center; backdrop-filter: blur(5px);">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⌛</div>
                                <div style="font-size: 0.75rem; color: #991b1b; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Nature's Clock</div>
                                <div style="font-size: 1.5rem; font-weight: 800; color: #dc2626; margin: 0.5rem 0;">{impact.get('decompose', 'N/A')}</div>
                                <div style="font-size: 0.8rem; color: #b91c1c;">to decompose</div>
                            </div>""", unsafe_allow_html=True)
                    with i_col2:
                        st.markdown(f"""
                            <div style="background: rgba(5, 150, 105, 0.05); border: 1px solid rgba(5, 150, 105, 0.2); padding: 1.5rem; border-radius: 20px; text-align: center; backdrop-filter: blur(5px);">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                                <div style="font-size: 0.75rem; color: #064e3b; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Energy Reclaimed</div>
                                <div style="font-size: 1.2rem; font-weight: 800; color: #059669; margin: 0.5rem 0; min-height: 3.6rem; display: flex; align-items: center; justify-content: center;">{impact.get('energy_saved', 'N/A')}</div>
                                <div style="font-size: 0.8rem; color: #047857;">saved by recycling</div>
                            </div>""", unsafe_allow_html=True)
                    with i_col3:
                        st.markdown(f"""
                            <div style="background: rgba(37, 99, 235, 0.05); border: 1px solid rgba(37, 99, 235, 0.2); padding: 1.5rem; border-radius: 20px; text-align: center; backdrop-filter: blur(5px);">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌳</div>
                                <div style="font-size: 0.75rem; color: #1e3a8a; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Carbon Offset</div>
                                <div style="font-size: 1.5rem; font-weight: 800; color: #2563eb; margin: 0.5rem 0;">{impact.get('co2_saved', 0)} kg</div>
                                <div style="font-size: 0.8rem; color: #1d4ed8;">per unit saved</div>
                            </div>""", unsafe_allow_html=True)
                    
                    st.info(f"💡 **Did you know?** {impact.get('fact', '')}")
                    
                    # Update analytics when capture is done
                    update_analytics(captured_detections, 'webcam')
                    
                else:
                    st.warning("No detections found in the captured frame.")
                    # Still show the captured image
                    if st.session_state.captured_frame is not None:
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.markdown("### 📷 Captured Frame")
                            st.image(captured_frame, use_container_width=True, channels="RGB")
            
            # Status and tips
            if webrtc_ctx.state.playing:
                st.success("✅ Camera is active! Point at waste materials for detection.")
                
                with st.expander("💡 Detection Tips"):
                    st.markdown("""
                    - **Good lighting** improves accuracy
                    - Hold **stable** for best results
                    - Show items **clearly** in frame
                    - Supported: Cardboard, Glass, Metal, Paper, Plastic
                    """)
                
                # Real-time stats
                if hasattr(webrtc_ctx, "video_processor") and webrtc_ctx.video_processor is not None:
                    last_detections = webrtc_ctx.video_processor.last_detections
                    
                    if last_detections:
                        # Update analytics occasionally (30% chance)
                        if random.random() < 0.3:
                            update_analytics(last_detections, 'webcam')
                        
                        st.markdown("### 📊 Live Stats")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Objects", len(last_detections))
                        
                        with col2:
                            unique = len(set([d['class'] for d in last_detections]))
                            st.metric("Types", unique)
                        
                        with col3:
                            avg_conf = sum([d['confidence'] for d in last_detections]) / len(last_detections)
                            st.metric("Confidence", f"{avg_conf:.1%}")
                            
                        # --- UPGRADED: LIVE IMPACT DASHBOARD ---
                        # Get the most frequent material in the live frame
                        live_main = max(set([d['class'] for d in last_detections]), 
                                       key=lambda x: [d['class'] for d in last_detections].count(x))
                        
                        live_impact = IMPACT_DATA.get(live_main, {})
                        
                        st.markdown(f"### ⚡ Live Environmental Scan: {live_main}")
                        
                        # Create visual impact cards that update with the camera
                        i_col1, i_col2, i_col3 = st.columns(3)
                        
                        with i_col1:
                            st.markdown(f"""
                                <div class="stats-card" style="border-top: 4px solid #dc2626;">
                                    <div class="stats-label">⌛ Decomposition</div>
                                    <div style="font-size: 1.4rem; font-weight: 700; color: #dc2626;">{live_impact.get('decompose')}</div>
                                </div>""", unsafe_allow_html=True)
                                
                        with i_col2:
                            st.markdown(f"""
                                <div class="stats-card" style="border-top: 4px solid #059669;">
                                    <div class="stats-label">⚡ Energy Saving</div>
                                    <div style="font-size: 1.4rem; font-weight: 700; color: #059669;">{live_impact.get('energy_saved')}</div>
                                </div>""", unsafe_allow_html=True)
                                
                        with i_col3:
                            st.markdown(f"""
                                <div class="stats-card" style="border-top: 4px solid #2563eb;">
                                    <div class="stats-label">🌳 Planet Fact</div>
                                    <div style="font-size: 0.85rem; font-weight: 500; color: #1e293b; line-height:1.2;">{live_impact.get('fact')}</div>
                                </div>""", unsafe_allow_html=True)
                        
                        # Add a progress-style 'Urgency' bar for decomposition
                        # (Plastic/Glass = Full bar, Paper = small bar)
                        severity = {"Glass": 100, "Plastic": 80, "Metal": 60, "Cardboard": 20, "Paper": 10}
                        sev_score = severity.get(live_main, 50)
                        
                        st.markdown(f"""
                            <div style="margin-top: -10px; margin-bottom: 20px;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748b; margin-bottom: 4px;">
                                    <span>Eco-Urgency Level</span>
                                    <span>{sev_score}%</span>
                                </div>
                                <div style="width: 100%; background: #e2e8f0; border-radius: 10px; height: 8px;">
                                    <div style="width: {sev_score}%; background: linear-gradient(90deg, #667eea, #dc2626); border-radius: 10px; height: 8px;"></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Plastic Subtype Selector for Live Webcam - Always show for selection
                        st.markdown("### 🧴 Select Plastic Type (Live)")
                        st.markdown("*Choose the specific type of plastic for proper recycling:*")
                        
                        # Create plastic type options for dropdown
                        plastic_options = [f"{PLASTIC_TYPES[pt]['emoji']} {PLASTIC_TYPES[pt]['name']}" for pt in PLASTIC_TYPES.keys()]
                        plastic_keys = list(PLASTIC_TYPES.keys())
                        
                        # Show plastic type selector
                        selected_plastic = st.selectbox(
                            "Choose Plastic Type:",
                            plastic_options,
                            index=0,
                            key="live_plastic_type_selector"
                        )
                        
                        # Get selected plastic type key
                        selected_idx = plastic_options.index(selected_plastic)
                        selected_plastic_key = plastic_keys[selected_idx]
                        
                        # Display selected plastic type info
                        plastic_info = PLASTIC_TYPES[selected_plastic_key]
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {plastic_info['color']}20 0%, {plastic_info['color']}10 100%); 
                                    border: 2px solid {plastic_info['color']}; 
                                    border-radius: 16px; padding: 1rem; margin: 0.5rem 0;">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <span style="font-size: 1.5rem;">{plastic_info['emoji']}</span>
                                <span style="font-weight: 700; color: {plastic_info['color']};">{plastic_info['name']}</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #374151; margin-top: 0.25rem;">
                                <strong>Common Items:</strong> {plastic_info['examples']}
                            </div>
                            <div style="font-size: 0.75rem; color: #6B7280; margin-top: 0.25rem;">
                                <strong>♻️ Recycling:</strong> {plastic_info['recycling']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        except ImportError as e:
            st.error(f"📹 Live camera requires additional packages: {e}")
            st.info("Please install: `pip install streamlit-webrtc av`")
            st.info("You can still use the Image Upload mode for detection.")
    
      # Footer
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem;">
            <span>♻️ Smart Bin</span>
            <span>•</span>
            <span>AI-Powered</span>
            <span>•</span>
            <span>Real-time Detection</span>
        </div>
        <div style="color: #94a3b8;">
            Made with ❤️ for a cleaner planet
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()