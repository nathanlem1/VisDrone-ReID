"""
This code generates VisDrone-ReID dataset from VisDrone2019 multi-object tracking (VisDrone-MOT) dataset:
https://github.com/VisDrone/VisDrone-Dataset

Train and Val sequences create the ReID train set, whereas test sequences create the ReID test set.
Every Nth frame of each sequence (N = 5, 10, etc.) is used to crop the objects’ image patches. When evaluating the ReID
performance, each image crop of the test set is taken as query image once and is compared against all other test images
that serve as gallery.

"""

import os
import cv2
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from loguru import logger

from global_id_tracker import GlobalIDTracker


class_indices = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
class_names = ("pedestrian", "people", "bicycle", "car", "van", "truck", 
               "tricycle", "awning-tricycle", "bus", "motor")
CATEGORIES = dict(zip(class_indices, class_names))


# ======================================================
# Sequence Lists
# ======================================================

TRAIN_SEQS = ['uav0000013_01073_v', 'uav0000013_01392_v', 'uav0000020_00406_v', 
              'uav0000071_03240_v', 'uav0000072_04488_v', 'uav0000072_05448_v', 
              'uav0000072_06432_v', 'uav0000076_00720_v', 'uav0000079_00480_v', 
              'uav0000084_00000_v', 'uav0000099_02109_v', 'uav0000124_00944_v', 
              'uav0000126_00001_v', 'uav0000138_00000_v', 'uav0000140_01590_v',
              'uav0000143_02250_v', 'uav0000145_00000_v', 'uav0000150_02310_v', 
              'uav0000218_00001_v', 'uav0000222_03150_v', 'uav0000239_03720_v', 
              'uav0000239_12336_v', 'uav0000243_00001_v', 'uav0000244_01440_v', 
              'uav0000248_00001_v', 'uav0000263_03289_v', 'uav0000264_02760_v', 
              'uav0000266_03598_v', 'uav0000266_04830_v', 'uav0000270_00001_v',
              'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 
              'uav0000281_00460_v', 'uav0000288_00001_v', 'uav0000289_00001_v', 
              'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000300_00000_v', 
              'uav0000307_00000_v', 'uav0000308_00000_v', 'uav0000308_01380_v', 
              'uav0000309_00000_v', 'uav0000315_00000_v', 'uav0000316_01288_v',
              'uav0000323_01173_v', 'uav0000326_01035_v', 'uav0000329_04715_v', 
              'uav0000342_04692_v', 'uav0000352_05980_v', 'uav0000357_00920_v', 
              'uav0000360_00001_v', 'uav0000361_02323_v', 'uav0000363_00001_v', 
              'uav0000366_00001_v', 'uav0000013_00000_v']  # 56 sequences

VAL_SEQS = ['uav0000086_00000_v', 'uav0000117_02622_v', 'uav0000137_00458_v', 
            'uav0000182_00000_v', 'uav0000268_05773_v', 'uav0000305_00000_v', 
            'uav0000339_00001_v']  # 7 sequences

TEST_SEQS = ['uav0000009_03358_v', 'uav0000073_00600_v', 'uav0000073_04464_v', 
             'uav0000077_00720_v', 'uav0000088_00290_v', 'uav0000119_02301_v', 
             'uav0000120_04775_v', 'uav0000161_00000_v', 'uav0000188_00000_v', 
             'uav0000201_00000_v', 'uav0000249_00001_v', 'uav0000249_02688_v', 
             'uav0000297_00000_v', 'uav0000297_02761_v', 'uav0000306_00230_v',
             'uav0000355_00001_v', 'uav0000370_00001_v']  # 17 sequences


# ======================================================
# Dataset Statistics
# ======================================================

class DatasetStatistics:
    def __init__(self):
        self.total_images = 0
        self.total_identities = 0
        self.category_counts = defaultdict(int)
        self.images_per_identity = []
        self.sequences_processed = []
        self.identities_removed = 0
        
    def update(self, pid_images):
        self.total_identities = len(pid_images)
        for pid, img_data in pid_images.items():
            num_images = len(img_data)
            self.images_per_identity.append(num_images)
            self.total_images += num_images
            for _, class_name in img_data:  
                self.category_counts[class_name] += 1
    
    def get_summary(self):
        if not self.images_per_identity:
            return {
                'total_images': 0,
                'total_identities': 0,
                'avg_images_per_id': 0,
                'min_images_per_id': 0,
                'max_images_per_id': 0,
                'category_distribution': {},
                'identities_removed': self.identities_removed
            }
        
        return {
            'total_images': self.total_images,
            'total_identities': self.total_identities,
            'avg_images_per_id': np.mean(self.images_per_identity),
            'min_images_per_id': np.min(self.images_per_identity),
            'max_images_per_id': np.max(self.images_per_identity),
            'category_distribution': dict(self.category_counts),
            'identities_removed': self.identities_removed
        }


# ======================================================
# Utility Functions
# ======================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_annotations(ann_path):
    return np.loadtxt(ann_path, delimiter=',', dtype=float)


# ======================================================
# Core Processing Function
# ======================================================

def process_split(
    split_name,
    sequences_dir,
    annotations_dir,
    output_dir,
    tracker,
    sequence_list,
    sample_rate=10,
    min_box_area=500,
    min_images_per_identity_for_split=4,
    display=False
):
    logger.info(f"Processing {split_name} split with {len(sequence_list)} sequences")
    logger.info(f"Minimum images per identity: {min_images_per_identity_for_split}")

    pid_images = defaultdict(list)
    stats = DatasetStatistics()
    stats.sequences_processed = sequence_list.copy()

    for seq in tqdm(sequence_list, desc=f"{split_name} sequences"):
        seq_img_dir = os.path.join(sequences_dir, seq)
        ann_file = os.path.join(annotations_dir, f"{seq}.txt")

        if not os.path.exists(ann_file):
            logger.warning(f"Missing annotation: {ann_file}")
            continue

        if not os.path.exists(seq_img_dir):
            logger.warning(f"Missing sequence directory: {seq_img_dir}")
            continue

        annotations = load_annotations(ann_file)

        for frame_id in np.unique(annotations[:, 0]).astype(int):
            if frame_id % sample_rate != 0:
                continue

            frame_path = os.path.join(seq_img_dir, f"{frame_id:07d}.jpg")

            if not os.path.exists(frame_path):
                continue

            img = cv2.imread(frame_path)
            if img is None:
                continue

            # Safe copy
            ann_frame = annotations[annotations[:, 0] == frame_id].copy()

            # Convert (x, y, w, h) → (x1, y1, x2, y2)
            ann_frame[:, 4:6] += ann_frame[:, 2:4]

            for row in ann_frame:
                local_id = int(row[1])
                x1, y1, x2, y2 = map(int, row[2:6])
                class_idx = int(row[7])
                
                # Category filtering
                if class_idx not in CATEGORIES.keys():
                    continue
                
                class_name = CATEGORIES[class_idx]

                if x2 <= x1 or y2 <= y1:
                    continue

                box_area = (x2 - x1) * (y2 - y1)
                if box_area < min_box_area:
                    continue

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                global_id = tracker.get_global_id(seq, local_id)
                
                pid_images[global_id].append((crop, class_name))

                if display:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{local_id}: {class_name}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    cv2.imshow("VisDrone", img)
                    cv2.waitKey(1)

    
    # ======================================================
    # Filter identities with too few images
    # ======================================================
    logger.info(f"Filtering identities with < {min_images_per_identity_for_split} images...")
    original_count = len(pid_images)

    # Collect statistics before filtering
    images_per_id_before = [len(img_data) for img_data in pid_images.values()]
    if images_per_id_before:
        logger.info(f"Before filtering: {original_count} identities")
        logger.info(f"  Images per ID - Min: {min(images_per_id_before)}, "
                   f"Max: {max(images_per_id_before)}, "
                   f"Avg: {np.mean(images_per_id_before):.2f}")
    
    filtered_pid_images = {}
    for pid, img_data in pid_images.items():
        if len(img_data) >= min_images_per_identity_for_split:
            filtered_pid_images[pid] = img_data
    
    filtered_count = len(filtered_pid_images)
    removed_count = original_count - filtered_count
    
    logger.info(f"After filtering: {filtered_count} identities kept, {removed_count} removed")
    
    if filtered_count > 0:
        images_per_id_after = [len(img_data) for img_data in filtered_pid_images.values()]
        logger.info(f"  Images per ID - Min: {min(images_per_id_after)}, "
                   f"Max: {max(images_per_id_after)}, "
                   f"Avg: {np.mean(images_per_id_after):.2f}")
    
    pid_images = filtered_pid_images
    stats.identities_removed = removed_count

    # Update statistics
    stats.update(pid_images)

    # ======================================================
    # Save images with class name in filenames
    # ======================================================
    save_root = os.path.join(output_dir, split_name)
    ensure_dir(save_root)

    total_images = 0

    for pid, img_data in pid_images.items():
        pid_dir = os.path.join(save_root, f"{pid:06d}")
        ensure_dir(pid_dir)

        for idx, (crop, class_name) in enumerate(img_data):
            
            img_name = f"{pid:06d}_{class_name}_{idx:04d}.jpg"
            cv2.imwrite(os.path.join(pid_dir, img_name), crop)
            total_images += 1

    logger.success(
        f"{split_name}: {len(pid_images)} identities, {total_images} images saved"
    )

    if display:
        cv2.destroyAllWindows()
    
    return stats


# ======================================================
# README Generator
# ======================================================

def generate_readme(output_dir, train_stats, test_stats, args):
    """Generate comprehensive README file for the dataset"""
    
    readme_path = os.path.join(output_dir, "README.md")
    
    train_summary = train_stats.get_summary()
    test_summary = test_stats.get_summary()
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# VisDrone ReID Dataset (Unified ID Space)\n\n")

        # Overview
        f.write("## Overview\n\n")
        f.write("This dataset is derived from the VisDrone2019-MOT benchmark for object re-identification (ReID) tasks.\n")
        f.write("It contains crops of tracked objects from drone surveillance videos, organized by identity.\n\n")
        f.write("**Key Feature:** This dataset uses a **unified global ID space** across both training and test sets. ")
        f.write("Identities are tracked consistently across all sequences using a global ID mapping system.\n\n")
        
        # Dataset Statistics
        f.write("## Dataset Statistics\n\n")
        f.write("### Training Set\n")
        f.write(f"- **Total Identities:** {train_summary['total_identities']}\n")
        f.write(f"- **Total Images:** {train_summary['total_images']}\n")
        f.write(f"- **Images per Identity:** {train_summary['avg_images_per_id']:.2f} (avg), "
                f"{train_summary['min_images_per_id']} (min), {train_summary['max_images_per_id']} (max)\n")
        f.write(f"- **Source Sequences:** {len(TRAIN_SEQS)} (train) + {len(VAL_SEQS)} (val) = {len(TRAIN_SEQS) + len(VAL_SEQS)} total\n")
        f.write(f"- **Identities Removed (< {args.min_images_per_train_id} images):** {train_summary['identities_removed']}\n\n")
        
        f.write("### Test Set\n")
        f.write(f"- **Total Identities:** {test_summary['total_identities']}\n")
        f.write(f"- **Total Images:** {test_summary['total_images']}\n")
        f.write(f"- **Images per Identity:** {test_summary['avg_images_per_id']:.2f} (avg), "
                f"{test_summary['min_images_per_id']} (min), {test_summary['max_images_per_id']} (max)\n")
        f.write(f"- **Source Sequences:** {len(TEST_SEQS)}\n")
        f.write(f"- **Identities Removed (< {args.min_images_per_test_id} images):** {test_summary['identities_removed']}\n\n")

        # Category Distribution
        f.write("## Category Distribution\n\n")
        f.write("### Training Set\n")
        f.write("| Category | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        for cat in sorted(train_summary['category_distribution'].keys()):
            count = train_summary['category_distribution'][cat]
            pct = (count / train_summary['total_images']) * 100
            f.write(f"| {cat} | {count} | {pct:.2f}% |\n")
        
        f.write("\n### Test Set\n")
        f.write("| Category | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        for cat in sorted(test_summary['category_distribution'].keys()):
            count = test_summary['category_distribution'][cat]
            pct = (count / test_summary['total_images']) * 100
            f.write(f"| {cat} | {count} | {pct:.2f}% |\n")
        
        # Processing Parameters
        f.write("\n## Processing Parameters\n\n")
        f.write(f"- **Sample Rate:** Every {args.sample_rate} frames\n")
        f.write(f"- **Minimum Bounding Box Area:** {args.min_box_area} pixels\n")
        f.write(f"- **Minimum Images per Train Identity:** {args.min_images_per_train_id}\n")
        f.write(f"- **Minimum Images per Test Identity:** {args.min_images_per_test_id}\n")
        f.write(f"- **Valid Categories:** {', '.join(class_names)}\n")
        f.write(f"- **ID Space:** Unified (single global ID tracker across all sequences)\n\n")
        
        # Directory Structure
        f.write("## Directory Structure\n\n")
        f.write("```\n")
        f.write("VisDroneReID/\n")
        f.write("|-- README.md\n")
        f.write("|-- train/\n")
        f.write("|   |-- 000000/\n")
        f.write("|   |   |-- 000000_pedestrian_0000.jpg\n")
        f.write("|   |   |-- 000000_pedestrian_0001.jpg\n")
        f.write("|   |   +-- ...\n")
        f.write("|   |-- 000001/\n")
        f.write("|   |   |-- 000001_car_0000.jpg\n")
        f.write("|   |   +-- ...\n")
        f.write("|   +-- ...\n")
        f.write("|-- test/\n")
        f.write("|   |-- 005584/\n")
        f.write("|   |   |-- 005584_pedestrian_0000.jpg\n")
        f.write("|   |   |-- 005584_pedestrian_0001.jpg\n")
        f.write("|   |   +-- ...\n")
        f.write("|   |-- 005585/\n")
        f.write("|   |   |-- 005585_bus_0000.jpg\n")
        f.write("|   |   +-- ...\n")
        f.write("|   +-- ...\n")
        f.write("```\n\n")

        # File Naming Convention
        f.write("## File Naming Convention\n\n")
        f.write("**Format:** `{identity_id:06d}_{class_name}_{image_index:04d}.jpg`\n\n")
        f.write("**Components:**\n")
        f.write("- `identity_id`: Global unique person/vehicle ID (6 digits)\n")
        f.write("- `class_name`: Object category (pedestrian, car, etc.)\n")
        f.write("- `image_index`: Sequential image number for this identity (4 digits)\n\n")
        f.write("**Examples:**\n")
        f.write("- `000000_pedestrian_0000.jpg` - Identity 0, pedestrian, 1st image\n")
        f.write("- `000042_car_0123.jpg` - Identity 42, car, 124th image\n")
        f.write("- `001844_bus_0005.jpg` - Identity 1844, bus, 6th image\n\n")

        # Quality Filtering
        f.write("## Quality Filtering\n\n")
        f.write("Identities are filtered based on the number of available images:\n")
        f.write(f"- **Training:** Minimum {args.min_images_per_train_id} images per identity\n")
        f.write(f"- **Testing:** Minimum {args.min_images_per_test_id} images per identity\n\n")
        f.write("**Rationale:**\n")
        f.write("- Ensures sufficient samples for robust model training\n")
        f.write("- Enables reliable evaluation with multiple gallery matches per query\n")
        f.write("- Filters out short, potentially noisy tracks\n")
        f.write("- Improves dataset quality and evaluation reliability\n\n")

        # Evaluation Protocol
        f.write("## Evaluation Protocol\n\n")
        f.write("This dataset follows standard ReID evaluation for within-sequence matching:\n\n")
        f.write("1. **Query:** One random image per identity\n")
        f.write("2. **Gallery:** All remaining images for each identity\n")
        f.write("3. **Matching:** For each query, rank all gallery images by similarity\n")
        f.write("4. **Metrics:**\n")
        f.write("   - **CMC (Cumulative Matching Characteristics):** Rank-1, Rank-5, Rank-10 accuracy\n")
        f.write("   - **mAP (mean Average Precision):** Overall retrieval quality\n\n")
        f.write("since each identity appears in only one sequence.\n\n")
        f.write("Query and gallery can be generated on the fly from the test set during evaluation.\n\n")

        # Source Sequences
        f.write("## Source Sequences\n\n")
        f.write("### Training Sequences\n")
        f.write(f"Total: {len(TRAIN_SEQS)} sequences\n\n")
        f.write("```\n")
        for i in range(0, len(TRAIN_SEQS), 5):
            f.write(", ".join(TRAIN_SEQS[i:i+5]) + "\n")
        f.write("```\n\n")
        
        f.write("### Validation Sequences (included in training set)\n")
        f.write(f"Total: {len(VAL_SEQS)} sequences\n\n")
        f.write("```\n")
        f.write(", ".join(VAL_SEQS) + "\n")
        f.write("```\n\n")
        
        f.write("### Test Sequences\n")
        f.write(f"Total: {len(TEST_SEQS)} sequences\n\n")
        f.write("```\n")
        for i in range(0, len(TEST_SEQS), 5):
            f.write(", ".join(TEST_SEQS[i:i+5]) + "\n")
        f.write("```\n\n")
    
    logger.success(f"README generated at: {readme_path}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Prepare VisDrone ReID dataset with unified global ID space"
    )

    parser.add_argument(
        "--input_dir",
        default="../datasets/VisDrone2019",  # ./dataset
        type=str,
        help="Root directory containing VisDrone2019-MOT-* folders"
    )

    parser.add_argument(
        "--output_dir",
        default="./VisDroneReID_data",
        type=str,
        help="Output ReID dataset directory"
    )

    parser.add_argument("--sample_rate", type=int, default=10,
                       help="Sample every Nth frame (default: 10)")
    
    parser.add_argument("--min_box_area", type=int, default=500,
                       help="Minimum bounding box area in pixels (default: 500)")
    
    parser.add_argument("--min_images_per_train_id", type=int, default=4,
                       help="Minimum images per identity in training set (default: 4)")
    
    parser.add_argument("--min_images_per_test_id", type=int, default=2,
                       help="Minimum images per identity in test set (default: 2)")
    
    parser.add_argument("--display", action="store_true",
                       help="Display images during processing")

    args = parser.parse_args()

    # UNIFIED ID SPACE: Single tracker for both train and test
    unified_tracker = GlobalIDTracker()

    # Directories
    train_root = os.path.join(args.input_dir, "VisDrone2019-MOT-train")
    val_root = os.path.join(args.input_dir, "VisDrone2019-MOT-val")
    test_root = os.path.join(args.input_dir, "VisDrone2019-MOT-test-dev")

    ensure_dir(args.output_dir)

    logger.info("="*60)
    logger.info("VisDrone ReID Dataset Preparation")
    logger.info("(Unified Global ID Space)")
    logger.info("="*60)
    logger.info(f"Valid categories: {list(CATEGORIES.values())}")
    logger.info(f"Sample rate: every {args.sample_rate} frames")
    logger.info(f"Minimum box area: {args.min_box_area} pixels")
    logger.info(f"Minimum images per train identity: {args.min_images_per_train_id}")
    logger.info(f"Minimum images per test identity: {args.min_images_per_test_id}")
    logger.info(f"Train sequences: {len(TRAIN_SEQS)}, Val sequences: {len(VAL_SEQS)}, Test sequences: {len(TEST_SEQS)}")
    logger.info("="*60)

    # Train sequences → ReID train
    logger.info("\n[1/4] Processing training sequences...")
    train_stats_1 = process_split(
        split_name="train",
        sequences_dir=os.path.join(train_root, "sequences"),
        annotations_dir=os.path.join(train_root, "annotations"),
        output_dir=args.output_dir,
        tracker=unified_tracker,
        sequence_list=TRAIN_SEQS,
        sample_rate=args.sample_rate,
        min_box_area=args.min_box_area,
        min_images_per_identity_for_split=args.min_images_per_train_id,
        display=args.display
    )

    # Val sequences → ReID train (same tracker)
    logger.info("\n[2/4] Processing validation sequences...")
    train_stats_2 = process_split(
        split_name="train",
        sequences_dir=os.path.join(val_root, "sequences"),
        annotations_dir=os.path.join(val_root, "annotations"),
        output_dir=args.output_dir,
        tracker=unified_tracker,
        sequence_list=VAL_SEQS,
        sample_rate=args.sample_rate,
        min_box_area=args.min_box_area,
        min_images_per_identity_for_split=args.min_images_per_train_id,
        display=args.display
    )

    # Merge train statistics
    train_stats = DatasetStatistics()
    train_stats.total_images = train_stats_1.total_images + train_stats_2.total_images
    train_stats.total_identities = train_stats_1.total_identities + train_stats_2.total_identities
    train_stats.images_per_identity = train_stats_1.images_per_identity + train_stats_2.images_per_identity
    train_stats.identities_removed = train_stats_1.identities_removed + train_stats_2.identities_removed
    for cat, count in train_stats_1.category_counts.items():
        train_stats.category_counts[cat] += count
    for cat, count in train_stats_2.category_counts.items():
        train_stats.category_counts[cat] += count

    # Test sequences → ReID test (SAME unified tracker)
    logger.info("\n[3/4] Processing test sequences...")
    test_stats = process_split(
        split_name="test",
        sequences_dir=os.path.join(test_root, "sequences"),
        annotations_dir=os.path.join(test_root, "annotations"),
        output_dir=args.output_dir,
        tracker=unified_tracker,
        sequence_list=TEST_SEQS,
        sample_rate=args.sample_rate,
        min_box_area=args.min_box_area,
        min_images_per_identity_for_split=args.min_images_per_test_id,
        display=args.display
    )

    # Generate README
    logger.info("\nGenerating README...")
    generate_readme(args.output_dir, train_stats, test_stats, args)

    # Final summary
    logger.success("\n" + "="*60)
    logger.success("VisDrone ReID Dataset Preparation COMPLETE!")
    logger.success("="*60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"\nGlobal ID Tracker:")
    logger.info(f"   - Total global IDs created: {unified_tracker.global_id_counter}")
    logger.info(f"\nTraining Set:")
    logger.info(f"   - Identities: {train_stats.total_identities}")
    logger.info(f"   - Images: {train_stats.total_images}")
    logger.info(f"   - Removed: {train_stats.identities_removed} identities")
    logger.info(f"\nTest Set:")
    logger.info(f"   - Identities: {test_stats.total_identities}")
    logger.info(f"   - Images: {test_stats.total_images}")
    logger.info(f"   - Removed: {test_stats.identities_removed} identities")
    logger.success("="*60)


# Execute from the interpreter
if __name__ == "__main__":
    main()