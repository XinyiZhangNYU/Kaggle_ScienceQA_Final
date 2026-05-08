import os
import pandas as pd
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def pad_to_square(img_path, output_path, fill_color=(255, 255, 255)):
    """Pad the image with a white background to make it square and save it, preventing key information from being cropped."""
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if w == h:
            img.save(output_path)
            return
        side = max(w, h)
        new_img = Image.new("RGB", (side, side), fill_color)
        new_img.paste(img, ((side - w) // 2, (side - h) // 2))
        new_img.save(output_path)
    except Exception as e:
        print(f"Failed to process image {img_path}: {e}")

def main():
    seed_everything(42)
    
    # ⚠️ Please modify according to your actual local path
    BASE_DIR = r"C:\Users\benxi\Downloads\images"
    TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
    VAL_CSV = os.path.join(BASE_DIR, "val.csv")
    
    # 1. Split the Unseen validation set
    print("🔍 Splitting the Unseen validation set...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    
    seen_questions = set(train_df['question'].dropna())
    val_unseen_df = val_df[~val_df['question'].isin(seen_questions)]
    
    unseen_path = os.path.join(BASE_DIR, "val_unseen.csv")
    val_unseen_df.to_csv(unseen_path, index=False)
    print(f"✅ Generated {unseen_path} (Total {len(val_unseen_df)} records)")

    # 2. (Optional) Offline Padding of images
    # If you don't want to process images offline, you can do it in __getitem__ in Colab
    print("\n🖼️ Preparing offline Padding for images (skipping processed ones)...")
    dirs_to_process = ["train", "val", "test"] # Assuming images are in these three subdirectories
    
    for d in dirs_to_process:
        in_dir = os.path.join(BASE_DIR, "images", d)
        out_dir = os.path.join(BASE_DIR, "images_padded", d)
        if not os.path.exists(in_dir):
            continue
            
        os.makedirs(out_dir, exist_ok=True)
        files = os.listdir(in_dir)
        
        for f in tqdm(files, desc=f"Padding {d} images"):
            in_path = os.path.join(in_dir, f)
            out_path = os.path.join(out_dir, f)
            if not os.path.exists(out_path): # Avoid duplicate processing
                pad_to_square(in_path, out_path)
                
    print("🎉 Offline data preparation complete! Please upload val_unseen.csv and the processed images to Drive.")

if __name__ == "__main__":
    main()
