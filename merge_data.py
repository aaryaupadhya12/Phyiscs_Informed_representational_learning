

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')



KAGGLE_PNEUMONIA_PATH = '/kaggle/input/chest-xray-pneumonia/chest_xray'

# For NIH, try these common paths:
NIH_PATH = '/kaggle/input/data'  # or '/kaggle/input/nih-chest-xrays/data'

# For CheXpert, the CSV might reference files with a prefix like "CheXpert-v1.0-small/"
CHEXPERT_PATH = '/kaggle/input/chexpert'

OUTPUT_DIR = '/kaggle/working/merged_dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)


NIH_MAX_SAMPLES = 20000
CHEXPERT_MAX_SAMPLES = 20000


def verify_path(path, name):
    """Verify if path exists and print status"""
    if os.path.exists(path):
        print(f"{name} found at: {path}")
        return True
    else:
        print(f"{name} NOT found at: {path}")
        return False

def find_file_recursive(root_dir, filename):
    """Recursively search for a file"""
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def explore_directory_structure(path, max_depth=3, current_depth=0):
    """Print directory structure for debugging"""
    if not os.path.exists(path) or current_depth >= max_depth:
        return
    
    items = []
    try:
        items = os.listdir(path)[:10]  # Show first 10 items
    except PermissionError:
        return
    
    indent = "  " * current_depth
    print(f"{indent}📁 {os.path.basename(path)}/")
    
    for item in items:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            print(f"{indent}  └─ 📂 {item}/")
            if current_depth < max_depth - 1:
                sub_items = os.listdir(item_path)[:3]
                for sub in sub_items:
                    print(f"{indent}      └─ {sub}")
        else:
            print(f"{indent}  └─ 📄 {item}")


def prepare_kaggle_pneumonia(source_path):
    """
    Structure: chest_xray/train|test|val/NORMAL|PNEUMONIA/*.jpeg
    """
    print("\n" + "="*60)
    print("Processing Kaggle Pneumonia Dataset")
    print("="*60)
    
    if not verify_path(source_path, "Kaggle Pneumonia"):
        return pd.DataFrame()
    
    data = []
    splits = ['train', 'test', 'val']
    
    for split in splits:
        split_path = os.path.join(source_path, split)
        if not os.path.exists(split_path):
            print(f"Split '{split}' not found, skipping...")
            continue
            
        for category in ['NORMAL', 'PNEUMONIA']:
            category_path = os.path.join(split_path, category)
            if not os.path.exists(category_path):
                print(f"Category '{category}' in '{split}' not found")
                continue
                
            images = [f for f in os.listdir(category_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"  Found {len(images)} images in {split}/{category}")
            
            for img_file in images:
                full_path = os.path.join(category_path, img_file)
                data.append({
                    'filepath': full_path,
                    'label': 1 if category == 'PNEUMONIA' else 0,
                    'source': 'kaggle_pneumonia',
                    'original_label': category,
                    'split': split
                })
    
    df = pd.DataFrame(data)
    print(f"\nKaggle Pneumonia: {len(df)} images loaded")
    print(f"   - NORMAL: {len(df[df['label']==0])}")
    print(f"   - PNEUMONIA: {len(df[df['label']==1])}")
    return df



def prepare_nih_chestxray14(source_path, max_samples=None):
    """
    Structure: Data_Entry_2017.csv + images/ folder
    """
    print("\n" + "="*60)
    print("Processing NIH ChestX-ray14 Dataset")
    print("="*60)
    
    if not verify_path(source_path, "NIH Dataset"):
        return pd.DataFrame()
    
    # Find metadata CSV
    metadata_path = os.path.join(source_path, 'Data_Entry_2017.csv')
    if not os.path.exists(metadata_path):
        print(f"Searching for Data_Entry_2017.csv recursively...")
        metadata_path = find_file_recursive(source_path, 'Data_Entry_2017.csv')
        if not metadata_path:
            print(f"Data_Entry_2017.csv not found. Skipping NIH dataset.")
            return pd.DataFrame()
    
    print(f"Found metadata: {metadata_path}")
    
    # Find images directory
    images_dir = os.path.join(source_path, 'images')
    if not os.path.exists(images_dir):
        # Try common alternatives
        for alt in ['images_001', 'data', 'Images']:
            alt_path = os.path.join(source_path, alt)
            if os.path.exists(alt_path):
                images_dir = alt_path
                break
    
    print(f"Images directory: {images_dir}")
    
    df = pd.read_csv(metadata_path)
    print(f" Found {len(df)} entries in metadata")
    
    data = []
    pneumonia_findings = ['Pneumonia', 'Infiltration', 'Consolidation', 'Edema']
    
    normal_count = 0
    pneumonia_count = 0
    missing_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Processing NIH'):
        # Try to find image in various subdirectories
        img_name = row['Image Index']
        img_path = None
        
        # Try direct path
        test_path = os.path.join(images_dir, img_name)
        if os.path.exists(test_path):
            img_path = test_path
        else:
            # Try subdirectories (images_001, images_002, etc.)
            for subdir in os.listdir(source_path):
                if subdir.startswith('images'):
                    test_path = os.path.join(source_path, subdir, 'images', img_name)
                    if os.path.exists(test_path):
                        img_path = test_path
                        break
        
        if not img_path:
            missing_count += 1
            continue
        
        finding = row['Finding Labels']
        
        if finding == 'No Finding':
            label = 0
            normal_count += 1
        elif any(pf in finding for pf in pneumonia_findings):
            label = 1
            pneumonia_count += 1
        else:
            continue  # Skip other findings
        
        data.append({
            'filepath': img_path,
            'label': label,
            'source': 'nih_chestxray14',
            'original_label': finding,
            'split': 'unknown'
        })
        
        if max_samples and len(data) >= max_samples:
            break
    
    df_result = pd.DataFrame(data)
    print(f"\n NIH ChestX-ray14: {len(df_result)} images loaded")
    print(f"   - NORMAL: {normal_count}")
    print(f"   - PNEUMONIA: {pneumonia_count}")
    print(f"   - Missing images: {missing_count}")
    return df_result



def prepare_chexpert(source_path, max_samples=None):
    """
    Structure: train.csv with Path column and finding columns
    """
    print("\n" + "="*60)
    print("Processing CheXpert Dataset")
    print("="*60)
    
    if not verify_path(source_path, "CheXpert Dataset"):
        return pd.DataFrame()
    
    # Show directory structure for debugging
    print(f"\n🔍 Exploring CheXpert directory structure:")
    explore_directory_structure(source_path, max_depth=2)
    
    # Find train.csv
    train_csv = os.path.join(source_path, 'train.csv')
    if not os.path.exists(train_csv):
        print(f"Searching for train.csv recursively...")
        train_csv = find_file_recursive(source_path, 'train.csv')
        if not train_csv:
            print(f"train.csv not found. Skipping CheXpert dataset.")
            return pd.DataFrame()
    
    print(f"Found metadata: {train_csv}")
    
    df = pd.read_csv(train_csv)
    print(f"📊 Found {len(df)} entries in metadata")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Sample first few paths to understand structure
    sample_paths = df['Path'].head(3).tolist()
    print(f"📝 Sample paths from CSV:")
    for p in sample_paths:
        print(f"   {p}")
    
    data = []
    normal_count = 0
    pneumonia_count = 0
    missing_count = 0
    checked_first = False
    
    for idx, row in tqdm(df.iterrows(), total=min(len(df), max_samples or len(df)), 
                         desc='Processing CheXpert'):
        # Handle relative paths - CheXpert paths usually start with "CheXpert-v1.0-small/"
        img_path = row['Path']
        
        # Try multiple path resolution strategies
        possible_paths = []
        
        # Strategy 1: Direct join
        possible_paths.append(os.path.join(source_path, img_path))
        
        # Strategy 2: Remove CheXpert prefix if it exists
        if 'CheXpert' in img_path:
            path_parts = img_path.split('/')
            # Find where actual patient folders start (usually after CheXpert-v1.0-small)
            for i, part in enumerate(path_parts):
                if part.startswith('patient'):
                    relative_path = '/'.join(path_parts[i:])
                    possible_paths.append(os.path.join(source_path, relative_path))
                    break
        
        # Strategy 3: Assume images are directly in source_path
        possible_paths.append(os.path.join(source_path, os.path.basename(img_path)))
        
        # Try each path
        found_path = None
        for test_path in possible_paths:
            if os.path.exists(test_path):
                found_path = test_path
                break
        
        if not found_path:
            missing_count += 1
            # Show first missing path for debugging
            if not checked_first and missing_count == 1:
                print(f"\nExample missing file:")
                print(f"   CSV path: {img_path}")
                print(f"   Tried paths:")
                for p in possible_paths[:2]:
                    print(f"     - {p}")
                checked_first = True
            continue
        
        img_path = found_path
        
        # Check findings (handling NaN values)
        lung_opacity = row.get('Lung Opacity', 0.0)
        consolidation = row.get('Consolidation', 0.0)
        edema = row.get('Edema', 0.0)
        no_finding = row.get('No Finding', 0.0)
        
        # Convert NaN to 0
        lung_opacity = 0.0 if pd.isna(lung_opacity) else lung_opacity
        consolidation = 0.0 if pd.isna(consolidation) else consolidation
        edema = 0.0 if pd.isna(edema) else edema
        no_finding = 0.0 if pd.isna(no_finding) else no_finding
        
        if no_finding == 1.0:
            label = 0
            normal_count += 1
        elif lung_opacity == 1.0 or consolidation == 1.0 or edema == 1.0:
            label = 1
            pneumonia_count += 1
        else:
            continue  # Uncertain or other findings
        
        data.append({
            'filepath': img_path,
            'label': label,
            'source': 'chexpert',
            'original_label': 'PNEUMONIA' if label == 1 else 'NORMAL',
            'split': 'unknown'
        })
        
        if max_samples and len(data) >= max_samples:
            break
    
    df_result = pd.DataFrame(data)
    print(f"\n CheXpert: {len(df_result)} images loaded")
    print(f"   - NORMAL: {normal_count}")
    print(f"   - PNEUMONIA: {pneumonia_count}")
    print(f"   - Missing images: {missing_count}")
    return df_result



print("\n" + ""*20)
print("STARTING DATASET MERGE PROCESS")
print(""*20)


kaggle_df = prepare_kaggle_pneumonia(KAGGLE_PNEUMONIA_PATH)
nih_df = prepare_nih_chestxray14(NIH_PATH, NIH_MAX_SAMPLES)
chexpert_df = prepare_chexpert(CHEXPERT_PATH, CHEXPERT_MAX_SAMPLES)


print("\n" + "="*60)
print("MERGING DATASETS")
print("="*60)

dfs_to_merge = []
if not kaggle_df.empty:
    dfs_to_merge.append(kaggle_df)
if not nih_df.empty:
    dfs_to_merge.append(nih_df)
if not chexpert_df.empty:
    dfs_to_merge.append(chexpert_df)

if not dfs_to_merge:
    print("ERROR: No datasets were loaded successfully!")
    print("Please check your paths and try again.")
else:
    combined_df = pd.concat(dfs_to_merge, ignore_index=True)
    
    print(f"\nCOMBINED DATASET STATISTICS")
    print("="*60)
    print(f"Total images: {len(combined_df)}")
    print(f"\nBy source:")
    print(combined_df['source'].value_counts())
    print(f"\nBy label:")
    print(f"  NORMAL (0):     {len(combined_df[combined_df['label']==0]):,}")
    print(f"  PNEUMONIA (1):  {len(combined_df[combined_df['label']==1]):,}")
    

    print("\n" + "="*60)
    print("✂️  CREATING TRAIN/VAL/TEST SPLITS")
    print("="*60)
    
    # 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        combined_df,
        test_size=0.3,
        stratify=combined_df['label'],
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=42
    )
    
    print(f"Train set: {len(train_df):,} images")
    print(f"  - NORMAL: {len(train_df[train_df['label']==0]):,}")
    print(f"  - PNEUMONIA: {len(train_df[train_df['label']==1]):,}")
    
    print(f"\nValidation set: {len(val_df):,} images")
    print(f"  - NORMAL: {len(val_df[val_df['label']==0]):,}")
    print(f"  - PNEUMONIA: {len(val_df[val_df['label']==1]):,}")
    
    print(f"\nTest set: {len(test_df):,} images")
    print(f"  - NORMAL: {len(test_df[test_df['label']==0]):,}")
    print(f"  - PNEUMONIA: {len(test_df[test_df['label']==1]):,}")
    
    # ----------------------------
    # 6️⃣ SAVE CSV FILES
    # ----------------------------
    print("\n" + "="*60)
    print("SAVING CSV FILES")
    print("="*60)
    
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    combined_df.to_csv(os.path.join(OUTPUT_DIR, 'combined_dataset.csv'), index=False)
    
    print(f"Saved: train.csv")
    print(f"Saved: val.csv")
    print(f"Saved: test.csv")
    print(f"Saved: combined_dataset.csv")
    
   
    print("\n" + "="*60)
    print("CREATING PYTORCH IMAGEFOLDER STRUCTURE")
    print("="*60)
    
    def create_imagefolder_structure(df, split_name):
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        normal_dir = os.path.join(split_dir, 'NORMAL')
        pneumonia_dir = os.path.join(split_dir, 'PNEUMONIA')
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(pneumonia_dir, exist_ok=True)
        
        copied = 0
        failed = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name}"):
            label_folder = pneumonia_dir if row['label'] == 1 else normal_dir
            
            # Create unique filename to avoid conflicts
            source_name = os.path.basename(row['filepath'])
            base_name, ext = os.path.splitext(source_name)
            dest_name = f"{row['source']}_{base_name}{ext}"A
            dest_path = os.path.join(label_folder, dest_name)
            
            try:
                if not os.path.exists(dest_path):
                    shutil.copy2(row['filepath'], dest_path)
                    copied += 1
            except Exception as e:
                print(f"Failed to copy {row['filepath']}: {e}")
                failed += 1
        
        print(f"{split_name}: {copied} images copied, {failed} failed")
    
    create_imagefolder_structure(train_df, 'train')
    create_imagefolder_structure(val_df, 'val')
    create_imagefolder_structure(test_df, 'test')
    
    print("\n" + "🎉 "*20)
    print("DATASET MERGE COMPLETE!")
    print("🎉 "*20)
    print(f"\nOutput location: {OUTPUT_DIR}")
    print("\nFolder structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── train/")
    print(f"  │   ├── NORMAL/")
    print(f"  │   └── PNEUMONIA/")
    print(f"  ├── val/")
    print(f"  │   ├── NORMAL/")
    print(f"  │   └── PNEUMONIA/")
    print(f"  ├── test/")
    print(f"  │   ├── NORMAL/")
    print(f"  │   └── PNEUMONIA/")
    print(f"  ├── train.csv")
    print(f"  ├── val.csv")
    print(f"  ├── test.csv")
    print(f"  └── combined_dataset.csv")