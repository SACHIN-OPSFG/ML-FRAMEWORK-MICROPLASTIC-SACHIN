"""
Download and convert the DeepParticle MICRO dataset (Frontiers 2024).
Source: Figshare, Apache 2.0 license.
Classes: hard (fragment), pellet, line (fiber), foam.
Published: July 2024.

This dataset provides:
- 4,795 annotated microplastic particles
- 68 high-resolution raw images split into 948 frames
- TSV bounding box annotations
- Individual cropped particle images sorted by class
"""

import os
import sys
import json
import tarfile
import hashlib
import shutil
import csv
from pathlib import Path
from collections import Counter

import requests
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / 'ml' / 'datasets' / 'raw' / 'frontiers_2024'
OUTPUT_DIR = ROOT / 'ml' / 'datasets' / 'raw' / 'frontiers_2024' / 'yolo_converted'

# DeepParticle MICRO dataset from Figshare
MICRO_URL = 'https://ndownloader.figshare.com/files/48254227'
MICRO_FILE = 'MICRO.tar.gz'

# Class mapping: DeepParticle -> Our taxonomy
# DeepParticle classes: hard, pellet, line, foam, noise, reference
# Our taxonomy: 0=fiber, 1=fragment, 2=film, 3=pellet, 4=foam, 5=bead
CLASS_MAP = {
    'hard': 1,      # -> fragment
    'pellet': 3,    # -> pellet
    'line': 0,      # -> fiber
    'foam': 4,      # -> foam
    'noise': None,  # skip
    'reference': None,  # skip
}

OUR_CLASSES = {
    0: 'fiber',
    1: 'fragment',
    2: 'film',
    3: 'pellet',
    4: 'foam',
    5: 'bead',
}


def download_micro():
    """Download the MICRO.tar.gz from Figshare."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / MICRO_FILE
    
    if target.exists():
        print(f"  Already downloaded: {target} ({target.stat().st_size / 1024 / 1024:.1f} MB)")
        return target
    
    print(f"  Downloading MICRO dataset from Figshare ({MICRO_URL})...")
    try:
        with requests.get(MICRO_URL, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            downloaded = 0
            tmp = target.with_suffix('.part')
            with open(tmp, 'wb') as f:
                for chunk in resp.iter_content(1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  Progress: {pct:.1f}% ({downloaded/1024/1024:.1f} MB)", end='', flush=True)
            print()
            tmp.replace(target)
        print(f"  Downloaded: {target.stat().st_size / 1024 / 1024:.1f} MB")
        return target
    except Exception as e:
        print(f"  ERROR: Download failed: {e}")
        return None


def extract_micro(tar_path):
    """Extract the MICRO.tar.gz archive."""
    extract_dir = RAW_DIR / 'MICRO_extracted'
    
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"  Already extracted: {extract_dir}")
        return extract_dir
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {tar_path.name}...")
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print(f"  Extracted to: {extract_dir}")
        return extract_dir
    except Exception as e:
        print(f"  ERROR: Extraction failed: {e}")
        return None


def find_annotations(extract_dir):
    """Find TSV annotation files in the extracted directory."""
    tsv_files = list(extract_dir.rglob('*.tsv'))
    csv_files = list(extract_dir.rglob('*.csv'))
    txt_files = [f for f in extract_dir.rglob('*.txt') if f.name != 'classes.txt']
    
    print(f"  Found {len(tsv_files)} TSV, {len(csv_files)} CSV, {len(txt_files)} TXT files")
    
    return tsv_files + csv_files


def find_class_folders(extract_dir):
    """Find class-sorted folders with cropped particle images."""
    class_folders = {}
    for class_name in CLASS_MAP.keys():
        folders = list(extract_dir.rglob(class_name))
        folders = [f for f in folders if f.is_dir()]
        if folders:
            class_folders[class_name] = folders
    
    return class_folders


def convert_cropped_particles_to_yolo(extract_dir):
    """
    Convert individually cropped particle images to YOLO-format training data.
    Each cropped image becomes a training sample with a single full-image bounding box.
    This is valid because each crop contains exactly one particle centered in the frame.
    """
    print("\n  Converting cropped particle images to YOLO format...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    stats = Counter()
    all_images = []
    
    # Find all class-sorted directories
    for class_name, mapped_id in CLASS_MAP.items():
        if mapped_id is None:
            continue
        
        # Search for directories named with the class
        class_dirs = list(extract_dir.rglob(class_name))
        class_dirs = [d for d in class_dirs if d.is_dir()]
        
        if not class_dirs:
            # Try looking for images with class prefix
            continue
        
        for class_dir in class_dirs:
            images = (
                list(class_dir.glob('*.jpg')) + 
                list(class_dir.glob('*.jpeg')) + 
                list(class_dir.glob('*.png')) +
                list(class_dir.glob('*.JPG')) +
                list(class_dir.glob('*.JPEG'))
            )
            
            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        if w < 10 or h < 10:
                            continue
                except Exception:
                    continue
                
                all_images.append({
                    'path': img_path,
                    'class_id': mapped_id,
                    'class_name': OUR_CLASSES[mapped_id],
                    'source_class': class_name,
                })
                stats[OUR_CLASSES[mapped_id]] += 1
    
    if not all_images:
        print("  WARNING: No cropped particle images found.")
        # Try to find images in any subdirectory
        all_found = list(extract_dir.rglob('*.jpg')) + list(extract_dir.rglob('*.jpeg')) + list(extract_dir.rglob('*.png'))
        print(f"  Total image files found in archive: {len(all_found)}")
        if all_found:
            # List directory structure
            dirs = set()
            for f in all_found[:100]:
                dirs.add(str(f.parent.relative_to(extract_dir)))
            print(f"  Sample directories: {sorted(dirs)[:20]}")
        return stats, all_images
    
    print(f"  Found {len(all_images)} cropped particle images:")
    for cls, count in sorted(stats.items()):
        print(f"    {cls}: {count}")
    
    # Split: 70/15/15 by class (stratified)
    import random
    random.seed(42)
    random.shuffle(all_images)
    
    # Group by class for stratified split
    by_class = {}
    for item in all_images:
        by_class.setdefault(item['class_name'], []).append(item)
    
    splits = {'train': [], 'val': [], 'test': []}
    for cls_images in by_class.values():
        n = len(cls_images)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        splits['train'].extend(cls_images[:n_train])
        splits['val'].extend(cls_images[n_train:n_train + n_val])
        splits['test'].extend(cls_images[n_train + n_val:])
    
    # Write YOLO format
    split_stats = {}
    for split, items in splits.items():
        img_dir = OUTPUT_DIR / split / 'images'
        lbl_dir = OUTPUT_DIR / split / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        n_boxes = 0
        for i, item in enumerate(items):
            stem = f"dp_{item['source_class']}_{i:05d}"
            dst_img = img_dir / f'{stem}.jpg'
            
            # Convert to JPG and resize if needed
            try:
                with Image.open(item['path']) as img:
                    img = img.convert('RGB')
                    # Resize if very large
                    if max(img.size) > 1280:
                        img.thumbnail((1280, 1280))
                    img.save(dst_img, quality=92)
                    w, h = img.size
            except Exception as e:
                print(f"    Skipped {item['path']}: {e}")
                continue
            
            # Create YOLO label: particle fills most of the cropped image
            # Estimate bbox as centered 80% of image (crop margins)
            cx, cy = 0.5, 0.5
            bw = min(0.85, max(0.5, 1.0))  # Most of the crop
            bh = min(0.85, max(0.5, 1.0))
            
            label_line = f"{item['class_id']} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"
            (lbl_dir / f'{stem}.txt').write_text(label_line, encoding='utf-8')
            n_boxes += 1
        
        split_stats[split] = {'images': len(items), 'annotations': n_boxes}
        print(f"  {split}: {len(items)} images, {n_boxes} annotations")
    
    return stats, all_images


def try_parse_tsv_annotations(extract_dir):
    """Try to parse TSV annotation files for frame-level detection."""
    tsv_files = list(extract_dir.rglob('*.tsv'))
    
    if not tsv_files:
        print("  No TSV annotation files found.")
        return []
    
    annotations = []
    for tsv_path in tsv_files:
        try:
            with open(tsv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 6:
                        try:
                            ann = {
                                'id': row[0],
                                'min_r': int(row[1]),
                                'min_c': int(row[2]),
                                'max_r': int(row[3]),
                                'max_c': int(row[4]),
                                'class': row[5].strip(),
                                'source_file': tsv_path.name,
                            }
                            annotations.append(ann)
                        except (ValueError, IndexError):
                            pass
        except Exception as e:
            print(f"    Error reading {tsv_path}: {e}")
    
    if annotations:
        class_counts = Counter(a['class'] for a in annotations)
        print(f"  Parsed {len(annotations)} annotations from TSV files:")
        for cls, cnt in class_counts.most_common():
            print(f"    {cls}: {cnt}")
    
    return annotations


def main():
    print("=" * 70)
    print("FRONTIERS 2024 DeepParticle MICRO Dataset Download & Conversion")
    print("License: Apache 2.0 (fully permissive)")
    print("=" * 70)
    
    # Step 1: Download
    tar_path = download_micro()
    if tar_path is None:
        print("FAILED: Could not download dataset.")
        return
    
    # Step 2: Extract
    extract_dir = extract_micro(tar_path)
    if extract_dir is None:
        print("FAILED: Could not extract dataset.")
        return
    
    # Step 3: Inspect structure
    print("\n  Inspecting extracted contents...")
    all_files = list(extract_dir.rglob('*'))
    print(f"  Total files: {len(all_files)}")
    
    # Count by extension
    ext_counts = Counter(f.suffix.lower() for f in all_files if f.is_file())
    print(f"  File types: {dict(ext_counts.most_common(10))}")
    
    # List top-level directories
    top_dirs = sorted(set(f.parts[len(extract_dir.parts)] for f in all_files if len(f.parts) > len(extract_dir.parts)))
    print(f"  Top-level entries: {top_dirs[:20]}")
    
    # Step 4: Try TSV annotations
    annotations = try_parse_tsv_annotations(extract_dir)
    
    # Step 5: Convert cropped particles to YOLO
    stats, images = convert_cropped_particles_to_yolo(extract_dir)
    
    # Step 6: Write metadata
    meta = {
        'source': 'Frontiers 2024 DeepParticle MICRO',
        'url': 'https://figshare.com/articles/dataset/DeepParticle_dataset_MICRO_MESO_MACRO_2022_/26511253',
        'license': 'Apache 2.0',
        'doi': '10.6084/m9.figshare.26511253.v1',
        'authors': 'Sarah-Jeanne Royer, Helen Wolter, et al. (The Ocean Cleanup)',
        'paper_doi': '10.3389/fenvs.2024.1386292',
        'class_map': {k: OUR_CLASSES.get(v) for k, v in CLASS_MAP.items() if v is not None},
        'particle_counts': dict(stats),
        'total_particles': sum(stats.values()),
        'total_tsv_annotations': len(annotations),
    }
    
    meta_path = RAW_DIR / 'frontiers_2024_metadata.json'
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f"\n  Metadata written: {meta_path}")
    
    print("\n  DONE. DeepParticle MICRO dataset processed.")


if __name__ == '__main__':
    main()
