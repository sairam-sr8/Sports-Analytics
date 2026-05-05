"""
========================================================
pose_detector.py — The Digital X-Ray Engine
========================================================

WHAT THIS FILE DOES (In Plain English):
----------------------------------------
This is the brain of Sprint 1. It takes any cricket .mp4 video
and uses Google MediaPipe to detect 33 skeleton body points
(called "landmarks") on every single frame — like a digital X-ray.

It then draws the skeleton over the player and prints the
coordinates of the Right Elbow to the terminal as proof.

TECHNICAL NOTE — WHY WE USE THE TASKS API:
-------------------------------------------
MediaPipe has two APIs (programming interfaces):
  1. OLD API (mp.solutions)  → Works on Python 3.9 - 3.12 only
  2. NEW API (mediapipe.tasks) → Works on Python 3.9 - 3.13+ (CURRENT)

Your system has Python 3.13, so we use the NEW Tasks API.
The underlying AI model is exactly the same — just accessed differently.

We need ONE model file: models/pose_landmarker_full.task
This is Google's pre-trained AI that understands human bodies.

HOW THE 33 LANDMARKS WORK:
----------------------------
MediaPipe assigns every body part a NUMBER (index).
The ones we care about for batting:

    INDEX → BODY PART
    0     → Nose
    11    → Left Shoulder       12 → Right Shoulder
    13    → Left Elbow          14 → Right Elbow  (KEY FOR BATTING)
    15    → Left Wrist          16 → Right Wrist
    23    → Left Hip            24 → Right Hip
    25    → Left Knee           26 → Right Knee
    27    → Left Ankle          28 → Right Ankle

Each landmark gives us:
  - x: position from LEFT of frame (0.0 = left edge, 1.0 = right edge)
  - y: position from TOP of frame  (0.0 = top edge, 1.0 = bottom edge)
  - z: depth (negative = in front of body center)
  - visibility: how clearly the AI can see this joint (0.0 to 1.0)

AUTHOR: Cricket Biomechanics Analyzer Project
SPRINT: 1 — Environment & The Digital Skeleton
"""

import cv2                    # For video reading and drawing on frames
import mediapipe as mp        # Google's pose detection AI
import numpy as np            # For math and array operations
import os                     # For file path handling
import time                   # For FPS calculation

# The new MediaPipe Tasks API modules
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ============================================================
#  SKELETON CONNECTIONS
#  This defines which landmarks to connect with lines to draw
#  a realistic skeleton. Each tuple = (start_point, end_point)
# ============================================================
POSE_CONNECTIONS = [
    # Torso
    (11, 12),   # Left Shoulder — Right Shoulder
    (11, 23),   # Left Shoulder — Left Hip
    (12, 24),   # Right Shoulder — Right Hip
    (23, 24),   # Left Hip — Right Hip

    # Left Arm
    (11, 13),   # Left Shoulder — Left Elbow
    (13, 15),   # Left Elbow — Left Wrist

    # Right Arm (KEY FOR BATTING ANALYSIS)
    (12, 14),   # Right Shoulder — Right Elbow
    (14, 16),   # Right Elbow — Right Wrist

    # Left Leg
    (23, 25),   # Left Hip — Left Knee
    (25, 27),   # Left Knee — Left Ankle

    # Right Leg
    (24, 26),   # Right Hip — Right Knee
    (26, 28),   # Right Knee — Right Ankle

    # Face connections
    (0, 11),    # Nose — Left Shoulder (neck approximate)
    (0, 12),    # Nose — Right Shoulder
]


# ============================================================
# SMOOTHED LANDMARKS WRAPPER
# ============================================================
# When we smooth landmark positions using numpy math,
# the result is a numpy array — not a MediaPipe landmark object.
# The rest of our code expects to read .x, .y, .z, .visibility
# from each landmark. This helper class makes numpy arrays
# behave exactly like MediaPipe landmark objects.
# ============================================================

class _SingleLandmark:
    """Mimics a MediaPipe NormalizedLandmark so our code works unchanged."""
    __slots__ = ('x', 'y', 'z', 'visibility')

    def __init__(self, x, y, z, visibility):
        self.x          = float(x)
        self.y          = float(y)
        self.z          = float(z)
        self.visibility = float(visibility)


class _SmoothedLandmarks:
    """
    A list-like container of _SingleLandmark objects.
    Created from a smoothed numpy array (shape 33x4).
    Lets code work with smoothed data the same way as raw MediaPipe data.
    """
    def __init__(self, smooth_array):
        # smooth_array shape: (33, 4) — 33 landmarks x (x, y, z, visibility)
        self._data = [
            _SingleLandmark(row[0], row[1], row[2], row[3])
            for row in smooth_array
        ]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


# ============================================================
# POSE DETECTOR CLASS
# ============================================================
# A CLASS is like a blueprint for a machine.
# We define what the machine CAN DO (methods), then build it.
#
# Usage:
#   detector = PoseDetector()
#   detector.process_video("my_video.mp4")
# ============================================================

class PoseDetector:
    """
    The Digital X-Ray Machine for Cricket Batting Analysis.

    Takes any cricket .mp4 video, runs Google MediaPipe Tasks API
    on every frame, draws a skeleton over the player, and extracts
    joint coordinates for biomechanical analysis.
    """

    # ── Landmark Index Constants ──────────────────────────
    # Store body part numbers as named constants.
    # This is safer than typing raw numbers everywhere.
    NOSE            = 0
    LEFT_SHOULDER   = 11
    RIGHT_SHOULDER  = 12
    LEFT_ELBOW      = 13
    RIGHT_ELBOW     = 14   # Primary Sprint 1 focus
    LEFT_WRIST      = 15
    RIGHT_WRIST     = 16
    LEFT_HIP        = 23
    RIGHT_HIP       = 24
    LEFT_KNEE       = 25
    RIGHT_KNEE      = 26
    LEFT_ANKLE      = 27
    RIGHT_ANKLE     = 28

    # Default path to the MediaPipe model file
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "pose_landmarker_full.task")

    def __init__(self, model_path=None, min_pose_detection_confidence=0.5,
                 min_tracking_confidence=0.5, smoothing_window=5):
        """
        Initialize the PoseDetector.

        Args:
            model_path (str): Path to the pose_landmarker_full.task model file.
            min_pose_detection_confidence (float): 0.0-1.0, how sure to detect a person.
            min_tracking_confidence (float): 0.0-1.0, how sure to keep tracking.
            smoothing_window (int): Number of frames to average for smoothing.
                                    Higher = smoother but more lag. Default=5.
        """
        print("\n" + "=" * 60)
        print("  CRICKET BIOMECHANICS ANALYZER — Initializing...")
        print("=" * 60)

        if model_path is None:
            model_path = self.DEFAULT_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"\n[ERROR] MediaPipe model not found at: {model_path}\n"
                "  Download it with:\n"
                "  python -c \"import urllib.request; "
                "urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/latest/pose_landmarker_full.task',"
                " 'models/pose_landmarker_full.task')\""
            )

        # ── Configure the Pose Landmarker ──────────────────
        # KEY FIX: num_poses=3 — detect up to 3 people in frame.
        # We then CHOOSE the primary subject (batter) ourselves
        # using _select_primary_person(). This fixes the random
        # jumping between coach and batter.
        base_options  = mp_python.BaseOptions(model_asset_path=model_path)
        self._options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=3,                                     # Detect up to 3 people
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_pose_presence_confidence=0.5,
            output_segmentation_masks=False,
        )

        self._landmarker = mp_vision.PoseLandmarker.create_from_options(self._options)

        # ── Smoothing Buffer ───────────────────────────────
        # FIX for jitter: We keep the last N frames of landmark positions
        # and average them. This makes the skeleton move smoothly even
        # when the AI loses a joint for 1-2 frames during fast swings.
        #
        # Think of it like this:
        #   Without smoothing: elbow jumps 50px between frames (looks glitchy)
        #   With smoothing:    elbow moves 10px between frames (looks natural)
        self._smooth_window = smoothing_window
        # Buffer stores the last N landmark sets (one per frame)
        # Each entry is a list of 33 (x, y, z, visibility) tuples
        self._landmark_buffer = []

        print(f"  [OK] MediaPipe Pose Landmarker loaded (Tasks API)")
        print(f"  [OK] Model: {model_path}")
        print(f"  [OK] Multi-person detection: UP TO 3 people")
        print(f"  [OK] Primary subject selection: AUTOMATIC (best visibility + center)")
        print(f"  [OK] Landmark smoothing: {smoothing_window}-frame window")
        print(f"  [OK] Detection confidence : {min_pose_detection_confidence}")
        print(f"  [OK] Tracking confidence  : {min_tracking_confidence}")
        print("=" * 60 + "\n")

    # ─────────────────────────────────────────────────────────
    # METHOD 1A: Select Primary Person (Anti-Glitch Logic)
    # ─────────────────────────────────────────────────────────
    def _select_primary_person(self, all_pose_landmarks):
        """
        When multiple people are detected (e.g., coach + batter),
        this method picks the PRIMARY SUBJECT — the batter.

        HOW IT WORKS:
        We score each detected person on two factors:
          1. Visibility Score: Count how many of their 33 landmarks
             are clearly visible. A full-body batter in frame will
             have more visible landmarks than a partially-visible coach.

          2. Center Score: How close is the person to the horizontal
             center of the frame? Coaching videos always center the batter.
             The coach stands to the side.

        The person with the highest combined score = the batter.

        Args:
            all_pose_landmarks (list): List of detected poses. Each pose
                                       is a list of 33 NormalizedLandmark.
        Returns:
            best_landmarks (list): The 33 landmarks of the primary subject.
                                   None if no poses detected.
        """
        if not all_pose_landmarks:
            return None

        if len(all_pose_landmarks) == 1:
            return all_pose_landmarks[0]  # Only one person, easy!

        # Multiple people detected — score each one
        best_score      = -1
        best_landmarks  = None

        for pose_landmarks in all_pose_landmarks:
            # ── Score 1: Visibility ──────────────────────
            # Count landmarks with visibility > 0.5
            visible_count = sum(
                1 for lm in pose_landmarks if lm.visibility > 0.5
            )
            visibility_score = visible_count / 33.0  # Normalize to 0-1

            # ── Score 2: Center Proximity ────────────────
            # Find the horizontal center of this person's body
            # We use the average X position of shoulders and hips
            key_indices = [self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
                           self.LEFT_HIP, self.RIGHT_HIP]
            key_x_positions = [
                pose_landmarks[i].x for i in key_indices
                if pose_landmarks[i].visibility > 0.3
            ]
            if key_x_positions:
                person_center_x = sum(key_x_positions) / len(key_x_positions)
            else:
                person_center_x = 0.5  # Default to frame center if no data

            # Distance from frame center (0.5 = center, 0.0/1.0 = edges)
            # Closer to center = higher center score
            center_score = 1.0 - abs(person_center_x - 0.5) * 2.0
            center_score = max(0.0, center_score)

            # ── Combined Score ───────────────────────────
            # Visibility matters more (70%) than position (30%)
            # because sometimes the batter IS near the edge
            combined_score = (visibility_score * 0.7) + (center_score * 0.3)

            if combined_score > best_score:
                best_score     = combined_score
                best_landmarks = pose_landmarks

        return best_landmarks

    # ─────────────────────────────────────────────────────────
    # METHOD 1B: Apply Smoothing to Landmarks
    # ─────────────────────────────────────────────────────────
    def _smooth_landmarks(self, landmarks_norm):
        """
        Reduce jitter by averaging landmark positions over the last N frames.

        WHAT THIS SOLVES:
        During a fast bat swing, the wrist might move 100px in one frame
        then MediaPipe loses it for a frame then finds it again at a
        slightly different position. This creates a "flickering" effect.

        WITH SMOOTHING:
        We remember the last 5 positions and return the average.
        So instead of: frame1=200px, frame2=LOST, frame3=210px
        We get:        frame1=200px, frame2=204px, frame3=208px (smooth!)

        Args:
            landmarks_norm (list or None): Current frame's 33 landmarks.
        Returns:
            smoothed (list or None): Smoothed landmark positions.
        """
        if landmarks_norm is None:
            # No detection this frame — keep buffer as is, skip smoothing
            return None

        # Convert landmarks to simple numpy array for easy math
        # Shape: (33, 4) — 33 landmarks, each with (x, y, z, visibility)
        current = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                             for lm in landmarks_norm], dtype=np.float32)

        # Add current frame to buffer
        self._landmark_buffer.append(current)

        # Keep only the last N frames (sliding window)
        if len(self._landmark_buffer) > self._smooth_window:
            self._landmark_buffer.pop(0)

        # Average all frames in buffer
        # np.mean across all frames = smooth position
        smoothed_array = np.mean(self._landmark_buffer, axis=0)  # Shape: (33, 4)

        # We return the smoothed array (as numpy)
        # Downstream code reads .x, .y etc so we create a simple wrapper
        return _SmoothedLandmarks(smoothed_array)

    # ─────────────────────────────────────────────────────────
    # METHOD 1C: Draw skeleton on a frame
    # ─────────────────────────────────────────────────────────
    def _draw_skeleton(self, frame_output, landmarks_norm):
        """Draw bones and joint dots on the frame."""
        h, w = frame_output.shape[:2]

        # Draw bone connections
        for (start_idx, end_idx) in POSE_CONNECTIONS:
            lm_start = landmarks_norm[start_idx]
            lm_end   = landmarks_norm[end_idx]
            if lm_start.visibility > 0.3 and lm_end.visibility > 0.3:
                pt1 = (int(lm_start.x * w), int(lm_start.y * h))
                pt2 = (int(lm_end.x * w),   int(lm_end.y * h))
                cv2.line(frame_output, pt1, pt2, (255, 220, 0), 2)  # Gold lines

        # Draw joint dots
        key_joints = [self.RIGHT_ELBOW, self.LEFT_ELBOW,
                      self.RIGHT_KNEE,  self.LEFT_KNEE,
                      self.RIGHT_SHOULDER, self.LEFT_SHOULDER,
                      self.RIGHT_HIP,   self.LEFT_HIP]

        for idx in range(len(landmarks_norm)):
            lm = landmarks_norm[idx]
            if lm.visibility > 0.3:
                px = int(lm.x * w)
                py = int(lm.y * h)
                if idx in key_joints:
                    cv2.circle(frame_output, (px, py), 7, (0, 255, 100), -1)  # Large green
                    cv2.circle(frame_output, (px, py), 7, (0, 180, 60), 1)
                else:
                    cv2.circle(frame_output, (px, py), 4, (0, 200, 70), -1)   # Small green

        return frame_output

    # ─────────────────────────────────────────────────────────
    # METHOD 1: Process a single frame (main entry point)
    # ─────────────────────────────────────────────────────────
    def process_frame(self, frame_bgr, timestamp_ms):
        """
        Run pose detection on ONE video frame.

        Args:
            frame_bgr: A single BGR video frame (from OpenCV).
            timestamp_ms (int): Frame timestamp in milliseconds.

        Returns:
            frame_output: The frame with skeleton drawn on it.
            landmarks: Smoothed landmarks of the primary person, or None.
        """
        # OPTIMIZATION: Resize large frames to max 720p to speed up processing
        # MediaPipe internal processing is fixed resolution anyway (256x256)
        # so sending 4K images just slows down the pipeline with no benefit.
        h, w = frame_bgr.shape[:2]
        max_dim = max(h, w)
        if max_dim > 640:
            scale = 640 / max_dim
            proc_frame = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        else:
            proc_frame = frame_bgr
            
        frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run detection — may find 0, 1, 2, or 3 people
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        frame_output = frame_bgr.copy()

        # Step 1: Select the batter (best primary person)
        primary = self._select_primary_person(result.pose_landmarks)

        # Step 2: Apply smoothing to remove jitter
        smoothed = self._smooth_landmarks(primary)

        # Step 3: Draw skeleton on frame if person was found
        if smoothed is not None:
            frame_output = self._draw_skeleton(frame_output, smoothed)

        return frame_output, smoothed

    # ─────────────────────────────────────────────────────────
    # METHOD 2: Get pixel coordinates of a specific landmark
    # ─────────────────────────────────────────────────────────
    def get_landmark_coords(self, landmarks_norm, landmark_index, frame_w, frame_h):
        """
        Convert a landmark's normalized position to actual pixel coordinates.

        MediaPipe gives positions as fractions (0.0 to 1.0 of frame size).
        For example: x=0.5, y=0.5 means the exact CENTER of the frame.
        This function converts those fractions to actual pixel numbers.

        Args:
            landmarks_norm (list): List of NormalizedLandmark objects.
            landmark_index (int): Which body part? (e.g., 14 = Right Elbow)
            frame_w (int): Video frame width in pixels.
            frame_h (int): Video frame height in pixels.

        Returns:
            tuple: (x_pixels, y_pixels, z_depth, visibility)
                   OR None if landmarks is None or landmark not visible.

            - x_pixels  : Distance from LEFT edge of frame (in pixels)
            - y_pixels  : Distance from TOP edge of frame (in pixels)
            - z_depth   : 3D depth estimate (negative = closer to camera)
            - visibility: 0.0-1.0, how clearly the AI sees this joint
        """
        if landmarks_norm is None:
            return None
        if landmark_index >= len(landmarks_norm):
            return None

        lm = landmarks_norm[landmark_index]

        x_pixels   = int(lm.x * frame_w)
        y_pixels   = int(lm.y * frame_h)
        z_depth    = round(lm.z, 4)
        visibility = round(lm.visibility, 3)

        return x_pixels, y_pixels, z_depth, visibility

    # ─────────────────────────────────────────────────────────
    # METHOD 3: Draw on-screen HUD (information panel)
    # ─────────────────────────────────────────────────────────
    def _draw_hud(self, frame, frame_count, total_frames, fps, elbow_data):
        """
        Draw a Heads-Up Display panel on the video frame.
        Shows frame number, processing speed, and Right Elbow coords.
        """
        h, w = frame.shape[:2]

        # Semi-transparent dark panel at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 95), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Title
        cv2.putText(frame, "CRICKET BIOMECHANICS ANALYZER  |  Sprint 1",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 1)

        # Frame counter & FPS
        progress_pct = int(frame_count / total_frames * 100) if total_frames > 0 else 0
        cv2.putText(frame,
                    f"Frame: {frame_count}/{total_frames} ({progress_pct}%)  |  Speed: {fps:.1f} fps",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        # Right Elbow coordinates
        if elbow_data:
            x, y, z, vis = elbow_data
            elbow_str = f"Right Elbow -> X:{x}px  Y:{y}px  Z:{z:.3f}  Visibility:{vis:.2f}"
            cv2.putText(frame, elbow_str,
                        (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 100), 1)
        else:
            cv2.putText(frame, "Right Elbow -> NOT DETECTED",
                        (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 255), 1)

        # Quit tip (top-right)
        cv2.putText(frame, "Press Q to quit",
                    (w - 160, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        return frame

    # ─────────────────────────────────────────────────────────
    # METHOD 4: Process Full Video  ← THE MAIN METHOD
    # ─────────────────────────────────────────────────────────
    def process_video(self, video_path, output_path=None, show_window=True, print_every_n=15):
        """
        Process an entire cricket video from start to finish.

        This ties everything together:
          1. Opens the .mp4 file
          2. Reads one frame at a time in a loop
          3. Runs pose detection on each frame
          4. Draws the skeleton + HUD on each frame
          5. Highlights the Right Elbow with a special marker
          6. Prints Right Elbow X,Y to the terminal
          7. Optionally saves the output video
          8. Returns all landmark data for Sprint 2 analysis

        Args:
            video_path (str):    Full path to the input cricket video.
            output_path (str):   Full path to save output video w/ skeleton.
                                 Set to None to skip saving.
            show_window (bool):  Show live video window during processing.
            print_every_n (int): Print elbow coordinates every N frames.

        Returns:
            list: All landmarks, one set per frame. Used in Sprint 2.
        """
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"\n[ERROR] Cannot open video!")
            print(f"  Path: {video_path}")
            print("  Check: Does the file exist? Is it a valid .mp4?")
            return None

        # Read video properties
        frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_orig     = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps_orig if fps_orig > 0 else 0

        print(f"  Resolution   : {frame_w} x {frame_h} pixels")
        print(f"  FPS          : {fps_orig:.1f}")
        print(f"  Total Frames : {total_frames}")
        print(f"  Duration     : {duration:.1f} seconds")

        # Set up output writer if saving
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps_orig, (frame_w, frame_h))
            print(f"  Saving output to: {output_path}")

        # ── Print table header for Right Elbow data ──────
        print(f"\n{'='*65}")
        print(f"  RIGHT ELBOW COORDINATES (printed every {print_every_n} frames)")
        print(f"{'='*65}")
        print(f"  {'Frame':>6} | {'X (px)':>8} | {'Y (px)':>8} | {'Visibility':>10}")
        print(f"  {'-'*6}---{'-'*8}---{'-'*8}---{'-'*10}")

        all_landmarks  = []
        frame_count    = 0
        detected_count = 0
        start_time     = time.time()
        fps_display    = 0.0

        # ── MAIN FRAME LOOP ───────────────────────────────
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"\n  {'='*63}")
                print("  VIDEO COMPLETE — Reached end of file.")
                break

            frame_count += 1

            # Calculate processing speed
            elapsed = time.time() - start_time
            if elapsed > 0.001:
                fps_display = frame_count / elapsed

            # MediaPipe Tasks API needs a timestamp in milliseconds
            # We calculate it from the frame number and original FPS
            timestamp_ms = int((frame_count / fps_orig) * 1000) if fps_orig > 0 else frame_count * 33

            # ── Core detection step ──────────────────────
            frame_out, landmarks_norm = self.process_frame(frame, timestamp_ms)
            all_landmarks.append(landmarks_norm)

            if landmarks_norm is not None:
                detected_count += 1

            # ── Get Right Elbow coordinates ──────────────
            elbow_data = self.get_landmark_coords(
                landmarks_norm, self.RIGHT_ELBOW, frame_w, frame_h
            )

            # ── Draw special Right Elbow marker ──────────
            if elbow_data:
                x, y, z, vis = elbow_data

                # Outer red ring (draws attention)
                cv2.circle(frame_out, (x, y), 16, (0, 0, 220), 3)
                # Inner white fill
                cv2.circle(frame_out, (x, y), 8, (255, 255, 255), -1)
                # Inner red center
                cv2.circle(frame_out, (x, y), 4, (0, 0, 200), -1)

                # Label
                label_x = x + 20
                label_y = y - 10 if y > 40 else y + 30
                cv2.putText(frame_out, "R.ELBOW",
                            (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

                # Print to terminal
                if frame_count % print_every_n == 0:
                    print(f"  {frame_count:>6} | {x:>8} | {y:>8} | {vis:>10.3f}")
            else:
                if frame_count % print_every_n == 0:
                    print(f"  {frame_count:>6} | {'---':>8} | {'---':>8} | {'N/A':>10}")

            # ── Draw HUD panel ───────────────────────────
            frame_out = self._draw_hud(frame_out, frame_count, total_frames,
                                       fps_display, elbow_data)

            # ── Save frame if writer is set ──────────────
            if writer:
                writer.write(frame_out)

            # ── Show live window ─────────────────────────
            if show_window:
                cv2.imshow("Cricket Digital Skeleton — Press Q to quit", frame_out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n  [STOPPED] User pressed Q.")
                    break

        # ── Cleanup ──────────────────────────────────────
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # ── Summary ──────────────────────────────────────
        total_time = time.time() - start_time
        detect_pct = (detected_count / frame_count * 100) if frame_count > 0 else 0

        print(f"\n  PROCESSING SUMMARY")
        print(f"  {'─'*45}")
        print(f"  Total frames processed  : {frame_count}")
        print(f"  Frames with skeleton    : {detected_count} ({detect_pct:.1f}%)")
        print(f"  Total time              : {total_time:.1f} seconds")
        print(f"  Average speed           : {frame_count/total_time:.1f} frames/sec")
        if output_path and writer:
            print(f"  Output video saved to   : {output_path}")
        print(f"  {'─'*45}")
        print("  [DONE] PoseDetector.process_video() finished!\n")

        return all_landmarks

    def close(self):
        """Explicitly close the MediaPipe landmarker to free memory."""
        if hasattr(self, '_landmarker') and self._landmarker:
            self._landmarker.close()
            print("  [OK] PoseDetector closed and memory freed.")

    def __del__(self):
        """Auto-cleanup when object is garbage collected."""
        try:
            self.close()
        except Exception:
            pass
