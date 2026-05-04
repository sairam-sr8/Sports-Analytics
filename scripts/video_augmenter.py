import os
import cv2
import glob

def augment_video(input_path, output_path, flip_horizontal=True, brightness_factor=1.0):
    print(f"Augmenting: {os.path.basename(input_path)}")
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {input_path}")
        return False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip horizontal (Simulate left-handed from right-handed)
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
            
        # Brightness adjustment
        if brightness_factor != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)
            
        out.write(frame)
        frame_count += 1
        
    cap.release()
    out.release()
    print(f"  [OK] Saved augmented video: {os.path.basename(output_path)} ({frame_count} frames)")
    return True

def main():
    print("="*60)
    print("  CRICKET BIOMECHANICS — VIDEO DATA AUGMENTER")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_dirs = [
        os.path.join(base_dir, "videos", "good"),
        os.path.join(base_dir, "videos", "bad")
    ]
    
    processed_count = 0
    for vdir in video_dirs:
        if not os.path.exists(vdir):
            continue
            
        # Find all MP4s that are NOT already augmented
        mp4_files = [f for f in glob.glob(os.path.join(vdir, "*.mp4")) if "_aug" not in os.path.basename(f)]
        
        for fpath in mp4_files:
            # Create a flipped version
            base_name = os.path.splitext(os.path.basename(fpath))[0]
            out_flip = os.path.join(vdir, f"{base_name}_aug_flip.mp4")
            if not os.path.exists(out_flip):
                augment_video(fpath, out_flip, flip_horizontal=True, brightness_factor=1.0)
                processed_count += 1
                
            # Create a darker version (simulates bad lighting)
            out_dark = os.path.join(vdir, f"{base_name}_aug_dark.mp4")
            if not os.path.exists(out_dark):
                augment_video(fpath, out_dark, flip_horizontal=False, brightness_factor=0.7)
                processed_count += 1
                
    print("="*60)
    print(f"  [DONE] Created {processed_count} augmented videos.")
    print("="*60)

if __name__ == "__main__":
    main()
