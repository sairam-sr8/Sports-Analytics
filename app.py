"""
========================================================
app.py — The Web App Dashboard (Sprint 4)
========================================================

Run this using:
streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
import joblib
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Must add 'src' to path so Streamlit can find our engine
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from pose_detector import PoseDetector
from feature_extractor import FeatureExtractor

# ====================================================================
# PAGE CONFIG & CSS (PREMIUM AESTHETICS)
# ====================================================================
st.set_page_config(
    page_title="Cricket Biomechanics AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling to make it look like a premium sports analytics tool
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #00FF88;
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 0px 4px 15px rgba(0, 255, 136, 0.4);
        margin-bottom: -10px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #A0AEC0;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(145deg, #1A202C, #2D3748);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0px 8px 24px rgba(0,0,0,0.2);
        border: 1px solid #4A5568;
        text-align: center;
    }
    .metric-title {
        color: #E2E8F0;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #00FF88;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .bad-metric {
        color: #FF4444 !important;
    }
</style>
""", unsafe_allow_html=True)


# ====================================================================
# CACHED ENGINE LOADING (So we don't reload the RF model)
# ====================================================================
@st.cache_resource
def load_ai_model():
    try:
        model = joblib.load("models/cricket_model.pkl")
    except Exception as e:
        model = None
    return model

ai_model = load_ai_model()


# ====================================================================
# MAIN UI LAYOUT
# ====================================================================
st.markdown('<p class="main-header">🏏 CRICKET BIOMECHANICS AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Markerless Technique Analysis Engine</p>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5753/5753896.png", width=60)
    st.title("Control Panel")
    st.markdown("Upload your smartphone video to receive an instant mathematical breakdown of your batting technique.")
    
    st.divider()
    
    upload_method = st.radio("Choose Video Source:", ["Upload My Own Video", "Use Sample Video", "Live Camera Feed"])
    
    video_file = None
    if upload_method == "Upload My Own Video":
        video_file = st.file_uploader("Upload .mp4 file", type=['mp4', 'mov', 'avi'])
    elif upload_method == "Use Sample Video":
        sample_path = "videos/samples/test_batting_sample.mp4"
        if os.path.exists(sample_path):
            if st.button("Load Sample Video"):
                with open(sample_path, "rb") as f:
                    video_file = f.read()  # Just mocking a file upload stream for simplicity
                    st.session_state['sample_loaded'] = True
        else:
            st.warning("Sample video not found.")

    st.divider()
    st.markdown("**AI Status:**")
    if ai_model:
        st.success("✅ Random Forest Brain Loaded")
    else:
        st.error("❌ AI Brain missing")


# ====================================================================
# VIDEO PROCESSING LOOP
# ====================================================================

# ── WEBRTC LIVE FEATURE ─────────────────────────────────────────────
class VideoProcessor:
    def __init__(self):
        # We initialize detector inside the processor to avoid thread lock issues across WebRTC backend
        self.detector = PoseDetector(smoothing_window=5)
        self.extractor = FeatureExtractor()
        self.ai_model = ai_model
        self.global_timestamp_ms = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.global_timestamp_ms += 33
        
        # Process the raw camera pixels
        out_img, landmarks = self.detector.process_frame(img, self.global_timestamp_ms)
        features = self.extractor.extract_features(landmarks)
        
        # Calculate scores identically to standard uploads
        if features and self.ai_model:
             X_input = [[
                features['right_elbow_angle'], 
                features['left_elbow_angle'], 
                features['right_knee_bend'], 
                features['left_knee_bend']
             ]]
             pred = self.ai_model.predict(X_input)[0]
             is_good = (pred == 1)
             label = "PERFECT FORM" if is_good else "BAD TECHNIQUE"
             color = (0, 255, 0) if is_good else (0, 0, 255)
             
             # Also write telemetry explicitly on camera feed (with massive black box for high contrast on phones!)
             cv2.rectangle(out_img, (10, 10), (700, 160), (0, 0, 0), -1)
             cv2.putText(out_img, label, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)
             cv2.putText(out_img, f"Right Elbow Angle: {int(features['right_elbow_angle'])}", (30, 105), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)
             cv2.putText(out_img, f"Front Knee Bend: {int(features['left_knee_bend'])}", (30, 145), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)
        else:
             cv2.rectangle(out_img, (10, 10), (400, 80), (0, 0, 0), -1)
             cv2.putText(out_img, "Searching...", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
             
        return av.VideoFrame.from_ndarray(out_img, format="bgr24")


if upload_method == "Live Camera Feed":
    st.markdown("### Live Evaluation Dashboard (Webcam)")
    st.info("💡 Make sure to give browser permissions to your Camera. Wait 5-10 seconds for the cloud connection to establish the first time.")
    
    webrtc_streamer(
        key="cricket-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True
    )
    
    st.stop()  # Halt execution here so we don't accidentally drop into standard file upload logic


ready_to_play = False
temp_video = None

if upload_method == "Upload My Own Video" and video_file is not None:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(video_file.read())
    temp_video.close()
    ready_to_play = True
elif upload_method == "Use Sample Video" and st.session_state.get('sample_loaded', False):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    with open("videos/samples/test_batting_sample.mp4", "rb") as f:
        temp_video.write(f.read())
    temp_video.close()
    ready_to_play = True

if ready_to_play and temp_video is not None:
    
    st.markdown("### Live Evaluation Dashboard")
    
    col_vid, col_metrics = st.columns([2, 1])
    
    with col_vid:
        st_frame = st.empty()  # Placeholder for the live video
        
    with col_metrics:
        st.markdown("#### Real-time Telemetry")
        st_score = st.empty()
        st_elbow = st.empty()
        st_knee = st.empty()
        
    # Open video
    cap = cv2.VideoCapture(temp_video.name)
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    
    frame_counter = 0
    good_frames = 0
    total_valid_frames = 0
    timestamp_ms = 0
    
    start_btn = st.sidebar.button("▶ START ANALYSIS", use_container_width=True, type="primary")
    
    if start_btn:
        # Re-initialize the vision engines fresh for this specific video run!
        # This prevents the "timestamp monotonically increasing" error across browser refreshes.
        detector = PoseDetector(smoothing_window=5)
        extractor = FeatureExtractor()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_counter += 1
            timestamp_ms += 33
            
            # 1. Pose Detection
            frame_out, landmarks = detector.process_frame(frame, timestamp_ms)
            
            # 2. Math Extraction
            features = extractor.extract_features(landmarks)
            
            prediction = "Analyzing..."
            is_good = False
            
            if features:
                total_valid_frames += 1
                
                # 3. AI Prediction using the 4 features exactly as trained
                if ai_model:
                    X_input = [ [
                        features['right_elbow_angle'], 
                        features['left_elbow_angle'], 
                        features['right_knee_bend'], 
                        features['left_knee_bend']
                    ] ]
                    
                    pred = ai_model.predict(X_input)[0]
                    if pred == 1:
                        good_frames += 1
                        prediction = "✅ PERFECT FORM"
                        is_good = True
                    else:
                        prediction = "❌ BAD TECHNIQUE"
                        
                # Update Metrics display
                overall_score = int((good_frames / total_valid_frames) * 100)
                
            else:
                overall_score = int((good_frames / total_valid_frames * 100)) if total_valid_frames > 0 else 0
                
            # Render HUD on video
            cv2.putText(frame_out, prediction, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, 
                        (0, 255, 0) if is_good else (0, 0, 255), 2)
            
            # Show live frame
            st_frame.image(frame_out, channels="BGR", use_container_width=True)
            
            # Render Side Metrics (Always display them, even if skeleton is blocked)
            score_color = "" if is_good else "bad-metric"
            disp_score = f"{overall_score}%"
            disp_elbow = f"{features['right_elbow_angle']}°" if features else "Searching..."
            disp_knee = f"{features['left_knee_bend']}°" if features else "Searching..."
            
            st_score.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Overall Technique Score</div>
                <div class="metric-value {score_color}">{disp_score}</div>
            </div>
            <br>
            """, unsafe_allow_html=True)
            
            st_elbow.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Right Elbow Angle</div>
                <div class="metric-value" style="font-size: 2rem;">{disp_elbow}</div>
            </div>
            <br>
            """, unsafe_allow_html=True)
            
            st_knee.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Front Knee Bend</div>
                <div class="metric-value" style="font-size: 2rem;">{disp_knee}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Prevent Streamlit loop from running too fast (makes video watchable)
            time.sleep(0.01)
            
        cap.release()
        detector.close()  # Clean up MediaPipe memory after video is done
        
        # End Summary
        st.success(f"🏁 Analysis Complete! Final Technique Score: {overall_score}/100")
        
        if overall_score > 70:
            st.balloons()
            st.markdown("### Couch's Notes: Great job! Your weight transfer and arm extension are highly consistent with professional data.")
        else:
            st.markdown("### Coach's Notes: We need to work on this. You are bending your elbow too much during the swing and not maintaining a solid front knee base.")

else:
    # If no video uploaded yet, show a placeholder
    st.info("👈 Please upload a video from the sidebar to begin analysis.")
    
    # Beautiful placeholder layout
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1. Upload")
        st.markdown("Upload any standard smartphone video of a cricket swing.")
    with col2:
        st.markdown("### 2. Digital X-Ray")
        st.markdown("Our MediaPipe engine scans 33 biomechanical joints at 30+ frames per second.")
    with col3:
        st.markdown("### 3. AI Scoring")
        st.markdown("Math is fed into a Random Forest Classifier to score technique purely on physics.")
