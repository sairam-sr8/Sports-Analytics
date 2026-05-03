"""
app.py — Cricket Biomechanics AI Dashboard (V2)
Run: streamlit run app.py
"""

import streamlit as st
import cv2, numpy as np, tempfile, time, os, sys, json, joblib, av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["turn:openrelay.metered.ca:80"],  "username": "openrelayproject", "credential": "openrelayproject"},
    {"urls": ["turn:openrelay.metered.ca:443"], "username": "openrelayproject", "credential": "openrelayproject"},
]})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from pose_detector    import PoseDetector
from feature_extractor import FeatureExtractor
from scorer           import BiomechanicsScorer, PRO_PROFILES, calculate_similarity_score
from phase_detector   import PhaseDetector, PHASE_NAMES, PHASE_COLORS
from shot_classifier  import ShotClassifier, SHOT_NAMES, SHOT_EMOJIS
from shap_explainer   import SHAPExplainer, FEATURE_DESCRIPTIONS
from player_db        import PlayerDatabase
from report_generator import generate_pdf_report

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(page_title="Cricket Biomechanics AI", page_icon="🏏",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: #0a0e1a; }
.main-header { color: #00FF88; font-size: 2.8rem; font-weight: 800;
               text-shadow: 0 0 30px rgba(0,255,136,0.5); margin-bottom: -5px; }
.sub-header  { color: #718096; font-size: 1.1rem; margin-bottom: 20px; }
.metric-card { background: linear-gradient(145deg,#111827,#1f2937);
               border-radius: 14px; padding: 18px; text-align: center;
               border: 1px solid #374151; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
.metric-title { color: #9CA3AF; font-size: .85rem; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 6px; }
.metric-value { color: #00FF88; font-size: 2.2rem; font-weight: 800; }
.metric-value.bad  { color: #F87171; }
.metric-value.warn { color: #FBBF24; }
.phase-pill { display:inline-block; padding: 4px 14px; border-radius: 20px;
              font-size: .85rem; font-weight: 600; margin: 4px; }
.injury-box { background: rgba(239,68,68,0.15); border: 1px solid #EF4444;
              border-radius: 10px; padding: 12px; margin-top: 8px; }
.tip-box    { background: rgba(59,130,246,0.12); border: 1px solid #3B82F6;
              border-radius: 10px; padding: 12px; margin-top: 8px; }
.pro-card   { background: linear-gradient(135deg,#1a1f35,#252b45);
              border-radius: 14px; padding: 16px; border: 1px solid #4B5563; }
.section-title { color:#E2E8F0; font-size:1.1rem; font-weight:700;
                 border-left: 3px solid #00FF88; padding-left: 10px; margin: 16px 0 10px; }
</style>
""", unsafe_allow_html=True)


# ── Cached Loaders ───────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        meta_path = "models/feature_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        feat_cols = meta.get('feature_columns', [])
    except Exception:
        feat_cols = ['right_elbow_angle','left_elbow_angle','right_knee_bend','left_knee_bend']

    rf, gb = None, None
    try: rf = joblib.load("models/cricket_model.pkl")
    except: pass
    try: gb = joblib.load("models/cricket_shap_model.pkl")
    except: pass
    return rf, gb, feat_cols

@st.cache_resource
def get_db():
    return PlayerDatabase()

rf_model, gb_model, feat_cols = load_model()
db = get_db()
scorer  = BiomechanicsScorer()
explainer = SHAPExplainer(gb_model or rf_model, feat_cols)


# ── Header ───────────────────────────────────────────────────
st.markdown('<p class="main-header">🏏 CRICKET BIOMECHANICS AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enterprise Sports Analytics · 22-Feature Markerless Analysis Engine</p>', unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab_analyze, tab_progress, tab_compare = st.tabs(["🎬 Analysis", "📈 Player Progress", "🏆 Pro Comparison"])

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5753/5753896.png", width=55)
    st.title("Control Panel")
    st.divider()

    player_name = st.text_input("👤 Player Name", value="Player 1")

    upload_method = st.radio("Video Source:", ["Upload Video", "Sample Video", "Live Camera"])
    video_file = None

    if upload_method == "Upload Video":
        video_file = st.file_uploader("Upload .mp4", type=['mp4','mov','avi'])
    elif upload_method == "Sample Video":
        sample_path = "videos/samples/test_batting_sample.mp4"
        if os.path.exists(sample_path):
            if st.button("Load Sample"): st.session_state['sample_loaded'] = True
        else:
            st.warning("Sample video not found.")

    selected_pro = st.selectbox("Compare with Pro:", list(PRO_PROFILES.keys()))
    st.divider()

    with st.expander("🤖 AI Engine Status", expanded=False):
        if rf_model:
            st.success("✅ RF Model Loaded")
        else:
            st.error("❌ RF Model missing")
        if gb_model:
            st.success("✅ SHAP Model Loaded")
        else:
            st.warning("⚠️ SHAP model missing — re-train")
        st.info(f"📊 Active Features: {len(feat_cols)}")
        st.caption("Run `python run_ml_trainer.py` to retrain after adding new videos.")


# ════════════════════════════════════════════════════════════
# TAB 1: ANALYSIS
# ════════════════════════════════════════════════════════════
with tab_analyze:

    def render_session_summary(all_scores, all_features, video_filename):
        if not all_scores: return
        st.markdown("---")
        st.markdown("## 🏁 Session Analysis Complete")

        session_scores = scorer.aggregate_session_scores(all_scores)
        ov_final = session_scores['overall_score']

        # Score cards row
        c1,c2,c3,c4,c5 = st.columns(5)
        def big_card(col, val, title, emoji):
            cls = "bad" if val<50 else "warn" if val<70 else ""
            col.markdown(f'<div class="metric-card"><div class="metric-title">{emoji} {title}</div>'
                         f'<div class="metric-value {cls}">{val}</div></div>', unsafe_allow_html=True)
        big_card(c1, ov_final, "Overall", "🎯")
        big_card(c2, session_scores['balance_score'],   "Balance",   "⚖️")
        big_card(c3, session_scores['stability_score'], "Stability", "🎯")
        big_card(c4, session_scores['power_score'],     "Power",     "💥")
        big_card(c5, session_scores['timing_score'],    "Timing",    "⏱️")

        # Most common shot
        from collections import Counter
        all_shots  = [shot_cls.classify_frame(f)[1] for f in all_features]
        top_shot   = Counter(all_shots).most_common(1)[0][0] if all_shots else "Unknown"
        all_phases = [phase_det.detect_phase(f)[1] for f in all_features]
        top_phase  = Counter(all_phases).most_common(1)[0][0] if all_phases else "Unknown"

        ic1, ic2 = st.columns(2)
        ic1.info(f"🏏 Primary Shot Detected: **{top_shot}**")
        ic2.info(f"🔄 Dominant Phase: **{top_phase}**")

        # SHAP Coaching Tips
        st.markdown('<p class="section-title">🧠 AI Coach Insights (SHAP Explainability)</p>',
                    unsafe_allow_html=True)
        if all_features:
            avg_feats = {k: float(np.mean([f.get(k,0) for f in all_features]))
                         for k in all_features[0]}
            explanation = explainer.explain_features(avg_feats, top_n=4)

            if explanation['coaching_tips']:
                st.markdown('<div class="tip-box">', unsafe_allow_html=True)
                for tip in explanation['coaching_tips']:
                    st.markdown(f"&nbsp;&nbsp;💡 {tip}")
                st.markdown('</div>', unsafe_allow_html=True)

            if explanation['top_negative']:
                st.markdown("**📉 Biggest areas dragging your score:**")
                for item in explanation['top_negative'][:3]:
                    st.markdown(f"- **{item['label']}** — value: `{item['actual_value']}`")

            if not explanation['shap_available']:
                st.caption("ℹ️ SHAP model not found — using rule-based explanation. Re-run `run_ml_trainer.py` to enable full SHAP.")

        # Injury Risk
        if session_scores.get('injury_flags'):
            st.markdown('<div class="injury-box">', unsafe_allow_html=True)
            for flag in session_scores['injury_flags']:
                st.markdown(f"&nbsp;&nbsp;{flag}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Feedback
        if session_scores.get('feedback'):
            for msg in session_scores['feedback']:
                st.markdown(f"- {msg}")

        # Pro Similarity
        if all_features:
            sim = calculate_similarity_score(avg_feats, selected_pro)
            st.markdown(f'<p class="section-title">🏆 vs {selected_pro}</p>', unsafe_allow_html=True)
            sim_score = sim['similarity_score']
            st.progress(sim_score / 100)
            st.markdown(f"**Similarity Score: {sim_score}/100**")
            st.caption(sim['summary'])

        # Save to DB
        db.save_session(
            player_name=player_name,
            scores=session_scores,
            shot_type=top_shot,
            swing_phase=top_phase,
            video_filename=video_filename,
            feedback=session_scores.get('feedback', []),
            injury_flags=session_scores.get('injury_flags', []),
        )
        st.success(f"✅ Session saved to {player_name}'s progress history!")

        # Generate and Download PDF Report
        ai_tips = explanation['coaching_tips'] if all_features and 'explanation' in locals() else []
        try:
            pdf_path = generate_pdf_report(player_name, session_scores, top_shot, top_phase, ai_tips)
            with open(pdf_path, "rb") as pdf_file:
                PDFbyte = pdf_file.read()
            st.download_button(
                label="📥 Download Full Coaching Report (PDF)",
                data=PDFbyte,
                file_name=os.path.basename(pdf_path),
                mime='application/octet-stream',
                type="primary"
            )
        except Exception as e:
            st.error(f"Could not generate PDF: {e}")

        if ov_final > 70: st.balloons()

    # ── LIVE WEBRTC ────────────────────────────────────────
    if upload_method == "Live Camera":
        st.markdown("### 📡 Live Biomechanics Feed")
        st.info("Allow camera access. Takes 5–10s to connect via cloud relay.")

        class LiveProcessor:
            def __init__(self):
                self.detector   = PoseDetector(smoothing_window=5)
                self.extractor  = FeatureExtractor()
                self.scorer_eng = BiomechanicsScorer()
                self.phase_det  = PhaseDetector()
                self.shot_cls   = ShotClassifier()
                self.ts = 0
                self.all_scores = []
                self.all_features = []

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                self.ts += 33
                out, lm = self.detector.process_frame(img, self.ts)
                feats = self.extractor.extract_features(lm)

                if feats:
                    scores = self.scorer_eng.score_features(feats)
                    self.all_scores.append(scores)
                    self.all_features.append(feats)
                    
                    phase_id, phase_name, _ = self.phase_det.detect_phase(feats)
                    shot_id, shot_name, _   = self.shot_cls.classify_frame(feats)
                    overall = scores['overall_score']
                    # Minimal HUD: just a small score badge, no cluttered text
                    color_bgr = (0,255,136) if overall > 70 else (0,165,255) if overall > 50 else (0,0,255)
                    cv2.rectangle(out, (8, 8), (200, 48), (0, 0, 0), -1)
                    cv2.putText(out, f"{overall}/100", (16, 38),
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, color_bgr, 2)
                else:
                    cv2.rectangle(out, (8, 8), (260, 48), (0, 0, 0), -1)
                    cv2.putText(out, "Detecting...", (16, 38),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 200, 255), 2)

                return av.VideoFrame.from_ndarray(out, format="bgr24")

        ctx = webrtc_streamer(key="live", mode=WebRtcMode.SENDRECV,
                        rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=LiveProcessor,
                        media_stream_constraints={"video":{"facingMode":"environment","width":{"ideal":640}},"audio":False},
                        async_processing=True)
                        
        if ctx.state.playing:
            st.info("🔴 Live recording in progress. Stop the stream to save your session.")
        elif not ctx.state.playing and ctx.video_processor:
            if hasattr(ctx.video_processor, 'all_scores') and len(ctx.video_processor.all_scores) > 0:
                st.success(f"Captured {len(ctx.video_processor.all_scores)} frames. Saving session...")
                render_session_summary(ctx.video_processor.all_scores, ctx.video_processor.all_features, "Live Camera Feed")
                
        st.stop()

    # ── PREPARE VIDEO FILE ─────────────────────────────────
    ready, temp_video = False, None
    if upload_method == "Upload Video" and video_file:
        t = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        t.write(video_file.read()); t.close()
        temp_video, ready = t, True
    elif upload_method == "Sample Video" and st.session_state.get('sample_loaded'):
        t = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        with open("videos/samples/test_batting_sample.mp4","rb") as f: t.write(f.read())
        t.close(); temp_video, ready = t, True

    if not ready:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 1️⃣ Upload")
            st.markdown("Any smartphone cricket video")
        with c2:
            st.markdown("### 2️⃣ 22-Feature X-Ray")
            st.markdown("MediaPipe scans 33 joints across Arms, Legs, Core, Balance & Head")
        with c3:
            st.markdown("### 3️⃣ AI Scoring")
            st.markdown("Get Balance, Stability, Power & Timing scores with SHAP coaching tips")
        st.info("👈 Upload a video or load the sample from the sidebar to begin.")
        st.stop()

    # ── VIDEO ANALYSIS LOOP ────────────────────────────────
    col_vid, col_metrics = st.columns([3, 2])
    with col_vid:
        st.markdown('<p class="section-title">📹 Video Feed</p>', unsafe_allow_html=True)
        vid_frame = st.empty()
    with col_metrics:
        st.markdown('<p class="section-title">📊 Live Telemetry</p>', unsafe_allow_html=True)
        # Row 1: Overall spans full width
        m_overall = st.empty()
        # Row 2: Balance | Stability
        row2_c1, row2_c2 = st.columns(2)
        with row2_c1: m_balance   = st.empty()
        with row2_c2: m_stability = st.empty()
        # Row 3: Power | Timing
        row3_c1, row3_c2 = st.columns(2)
        with row3_c1: m_power  = st.empty()
        with row3_c2: m_timing = st.empty()
        # Row 4: Phase | Shot
        row4_c1, row4_c2 = st.columns(2)
        with row4_c1: m_phase = st.empty()
        with row4_c2: m_shot  = st.empty()

    if st.sidebar.button("▶ START ANALYSIS", use_container_width=True, type="primary"):
        cap = cv2.VideoCapture(temp_video.name)
        detector    = PoseDetector(smoothing_window=5)
        extractor   = FeatureExtractor()
        phase_det   = PhaseDetector()
        shot_cls    = ShotClassifier()

        all_scores, all_features = [], []
        frame_counter, ts = 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_counter += 1; ts += 33

            out, lm = detector.process_frame(frame, ts)
            feats = extractor.extract_features(lm)

            if feats:
                scores = scorer.score_features(feats)
                phase_id, phase_name, _ = phase_det.detect_phase(feats)
                shot_id, shot_name, _   = shot_cls.classify_frame(feats)
                all_scores.append(scores); all_features.append(feats)
                ov = scores['overall_score']
                bal, stab, pwr, tim = (scores['balance_score'], scores['stability_score'],
                                       scores['power_score'], scores['timing_score'])
                # --- NO TEXT OVERLAID ON VIDEO ---
                # Scores shown exclusively in the right-side panel
                # Only the skeleton (drawn by PoseDetector) remains on the frame

                def card(val, title, color=None):
                    cls = "bad" if val < 50 else "warn" if val < 70 else ""
                    style = f'color:{color};' if color else ''
                    return (f'<div class="metric-card"><div class="metric-title">{title}</div>'
                            f'<div class="metric-value {cls}" style="{style}">{val}</div></div>')

                def text_card(text, title, color="#60A5FA"):
                    return (f'<div class="metric-card"><div class="metric-title">{title}</div>'
                            f'<div style="color:{color};font-size:.95rem;font-weight:700;padding-top:4px">{text}</div></div>')

                # Row 1: Overall (full width, larger)
                ov_cls = "bad" if ov < 50 else "warn" if ov < 70 else ""
                m_overall.markdown(
                    f'<div class="metric-card" style="background:linear-gradient(135deg,#111827,#1e2d45);border-color:#374151;">'
                    f'<div class="metric-title">🎯 Overall Technique Score</div>'
                    f'<div class="metric-value {ov_cls}" style="font-size:2.8rem">{ov}/100</div></div>',
                    unsafe_allow_html=True)
                # Row 2: Balance | Stability
                m_balance.markdown(card(bal,  "⚖️ Balance"),   unsafe_allow_html=True)
                m_stability.markdown(card(stab, "🎯 Stability"), unsafe_allow_html=True)
                # Row 3: Power | Timing
                m_power.markdown(card(pwr,  "💥 Power"),   unsafe_allow_html=True)
                m_timing.markdown(card(tim,  "⏱️ Timing"),  unsafe_allow_html=True)
                # Row 4: Phase | Shot
                m_phase.markdown(text_card(phase_name, "🔄 Phase", "#60A5FA"), unsafe_allow_html=True)
                m_shot.markdown(text_card(f"{SHOT_EMOJIS.get(shot_id,'')}{shot_name}", "🏏 Shot", "#A78BFA"), unsafe_allow_html=True)

            vid_frame.image(out, channels="BGR", use_container_width=True)
            time.sleep(0.01)

        cap.release()
        detector.close()

        # ── SESSION SUMMARY ────────────────────────────────
        if all_scores:
            render_session_summary(all_scores, all_features, getattr(video_file, 'name', 'sample'))


# ════════════════════════════════════════════════════════════
# TAB 2: PLAYER PROGRESS
# ════════════════════════════════════════════════════════════
with tab_progress:
    st.markdown("## 📈 Player Progress Dashboard")
    all_players = db.list_players()
    if not all_players:
        st.info("No sessions recorded yet. Complete an analysis to see your progress here.")
    else:
        selected_player = st.selectbox("Select Player:", all_players)
        trend  = db.get_player_trend(selected_player)
        radar  = db.get_radar_data(selected_player)
        bests  = db.get_personal_best(selected_player)
        shots  = db.get_shot_distribution(selected_player)

        if bests:
            b1,b2,b3,b4,b5 = st.columns(5)
            def pb_card(col, val, label):
                col.markdown(f'<div class="metric-card"><div class="metric-title">{label}</div>'
                             f'<div class="metric-value">{val or "—"}</div></div>', unsafe_allow_html=True)
            pb_card(b1, bests.get('best_overall'),   "🏆 Best Overall")
            pb_card(b2, bests.get('best_balance'),   "⚖️ Best Balance")
            pb_card(b3, bests.get('best_stability'), "🎯 Best Stability")
            pb_card(b4, bests.get('best_power'),     "💥 Best Power")
            pb_card(b5, bests.get('total_sessions'), "📊 Sessions")

        if trend and trend.get('timestamps'):
            import pandas as pd
            st.markdown('<p class="section-title">Score Trend Over Time</p>', unsafe_allow_html=True)
            df_trend = pd.DataFrame({
                'Date':      trend['timestamps'],
                'Overall':   trend['overall_score'],
                'Balance':   trend['balance_score'],
                'Stability': trend['stability_score'],
                'Power':     trend['power_score'],
                'Timing':    trend['timing_score'],
            })
            st.line_chart(df_trend.set_index('Date'))

        if radar and radar.get('scores'):
            st.markdown('<p class="section-title">Latest Session Radar</p>', unsafe_allow_html=True)
            import pandas as pd
            radar_df = pd.DataFrame({
                'Dimension': radar['categories'],
                'Score':     radar['scores'],
            })
            st.bar_chart(radar_df.set_index('Dimension'))

        if shots:
            st.markdown('<p class="section-title">Shot Distribution</p>', unsafe_allow_html=True)
            import pandas as pd
            shot_df = pd.DataFrame(list(shots.items()), columns=['Shot', 'Count'])
            st.bar_chart(shot_df.set_index('Shot'))


# ════════════════════════════════════════════════════════════
# TAB 3: PRO COMPARISON
# ════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("## 🏆 Professional Player Benchmarks")
    st.markdown("Compare your biomechanics against elite international cricketers.")

    for pro_name, profile in PRO_PROFILES.items():
        with st.expander(f"**{pro_name}** — Overall Benchmark: {profile['overall_score']}/100"):
            st.markdown(f"*{profile['description']}*")
            cols = st.columns(3)
            metrics = [
                ("Elbow Angle", profile.get('right_elbow_angle'), "°"),
                ("Shoulder Rot.", profile.get('shoulder_rotation_angle'), "°"),
                ("Hip Rotation",  profile.get('hip_rotation_angle'), "°"),
                ("Front Knee",    profile.get('left_knee_bend'), "°"),
                ("Backlift Ht.",  profile.get('backlift_height'), ""),
                ("Feet Width",    profile.get('feet_width_ratio'), "x"),
            ]
            for i, (label, val, unit) in enumerate(metrics):
                cols[i % 3].metric(label, f"{val}{unit}" if val else "N/A")

    st.markdown("---")
    st.markdown("### 🧮 How Similarity is Calculated")
    st.markdown("""
    For each of 10 key biomechanical features, we calculate the absolute deviation 
    between your averaged value and the professional's reference value. 
    We then normalize this by the scale of the feature and compute an overall 
    closeness percentage. **100% = identical technique.**
    """)
