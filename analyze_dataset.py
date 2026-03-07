"""
Analyze training dataset characteristics
Helps identify diversity issues
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict


def analyze_dataset(data_dir='input/train'):
    """Analyze dataset characteristics"""

    print("="*60)
    print("TRAINING DATASET ANALYSIS")
    print("="*60)

    stats = defaultdict(list)
    total_images = 0

    # Analyze all images
    for class_dir in sorted(Path(data_dir).iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg'))

        class_count = len(image_files)
        total_images += class_count

        # Sample analysis (first 10 images per class)
        sample_files = image_files[:min(10, len(image_files))]

        for img_path in sample_files:
            img = Image.open(img_path)
            img_array = np.array(img)

            # Collect stats
            stats['width'].append(img.size[0])
            stats['height'].append(img.size[1])
            stats['brightness'].append(img_array.mean())

            # Background detection (corners)
            h, w = img_array.shape[:2]
            corners = [
                img_array[0:10, 0:10].mean(),
                img_array[0:10, w-10:w].mean(),
                img_array[h-10:h, 0:10].mean(),
                img_array[h-10:h, w-10:w].mean()
            ]
            stats['background'].append(np.mean(corners))

    # Summary
    print(f"\nTotal Images: {total_images}")
    print(f"Classes: {len(list(Path(data_dir).iterdir()))}")
    print(f"Avg images per class: {total_images // len(list(Path(data_dir).iterdir()))}")

    print("\n" + "-"*60)
    print("IMAGE CHARACTERISTICS (Sample Analysis)")
    print("-"*60)

    print(f"\nImage Sizes:")
    print(f"  Width:  {int(np.min(stats['width']))} - {int(np.max(stats['width']))} "
          f"(avg: {int(np.mean(stats['width']))})")
    print(f"  Height: {int(np.min(stats['height']))} - {int(np.max(stats['height']))} "
          f"(avg: {int(np.mean(stats['height']))})")

    if np.std(stats['width']) < 1 and np.std(stats['height']) < 1:
        print("  ⚠️  WARNING: ALL images are the same size (low diversity)")

    print(f"\nBrightness:")
    print(f"  Range: {int(np.min(stats['brightness']))} - {int(np.max(stats['brightness']))} / 255")
    print(f"  Average: {int(np.mean(stats['brightness']))} / 255")
    print(f"  Std Dev: {int(np.std(stats['brightness']))}")

    if np.std(stats['brightness']) < 10:
        print("  ⚠️  WARNING: Very narrow brightness range (low diversity)")

    print(f"\nBackground:")
    print(f"  Average corner brightness: {int(np.mean(stats['background']))} / 255")

    if np.mean(stats['background']) < 20:
        print("  ⚠️  WARNING: Very dark backgrounds, likely black (low diversity)")
    elif np.mean(stats['background']) > 235:
        print("  ⚠️  WARNING: Very bright backgrounds, likely white (low diversity)")

    if np.std(stats['background']) < 10:
        print("  ⚠️  WARNING: Very uniform backgrounds (low diversity)")

    print("\n" + "="*60)
    print("DIVERSITY SCORE")
    print("="*60)

    # Calculate diversity score
    diversity_score = 0
    max_score = 6

    # Size variety
    if np.std(stats['width']) > 50:
        diversity_score += 1
        print("✓ Size variety: GOOD")
    else:
        print("✗ Size variety: POOR (all same size)")

    # Brightness variety
    if np.std(stats['brightness']) > 30:
        diversity_score += 1
        print("✓ Brightness variety: GOOD")
    else:
        print("✗ Brightness variety: POOR (narrow range)")

    # Background variety
    if np.std(stats['background']) > 50:
        diversity_score += 1
        print("✓ Background variety: GOOD")
    else:
        print("✗ Background variety: POOR (uniform)")

    # Background type
    avg_bg = np.mean(stats['background'])
    if 50 < avg_bg < 200:
        diversity_score += 1
        print("✓ Background type: VARIED")
    else:
        print("✗ Background type: EXTREME (all black or all white)")

    # Aspect ratio variety
    aspects = [w/h for w, h in zip(stats['width'], stats['height'])]
    if np.std(aspects) > 0.1:
        diversity_score += 1
        print("✓ Aspect ratio variety: GOOD")
    else:
        print("✗ Aspect ratio variety: POOR (all square)")

    # Overall brightness distribution
    brightness_range = np.max(stats['brightness']) - np.min(stats['brightness'])
    if brightness_range > 80:
        diversity_score += 1
        print("✓ Brightness distribution: WIDE")
    else:
        print("✗ Brightness distribution: NARROW")

    print(f"\nOverall Diversity Score: {diversity_score}/{max_score}")

    if diversity_score <= 2:
        print("🔴 CRITICAL: Very low diversity - expect poor real-world performance")
    elif diversity_score <= 4:
        print("🟡 WARNING: Low diversity - may struggle with real-world images")
    else:
        print("🟢 GOOD: Reasonable diversity - should generalize better")

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if diversity_score <= 2:
        print("\n⚠️  Your dataset has VERY low diversity!")
        print("\nUrgent recommendations:")
        print("1. Collect 50+ images per class with VARIED backgrounds")
        print("2. Use different lighting conditions (bright, dim, natural)")
        print("3. Take photos with different cameras/phones")
        print("4. Vary the distance and angle")
        print("\nExpected improvement: 30-60% → 85-95% on real-world images")

    elif diversity_score <= 4:
        print("\n⚠️  Your dataset has limited diversity")
        print("\nRecommendations:")
        print("1. Add 20-30 diverse images per class")
        print("2. Use extreme data augmentation")
        print("3. Apply preprocessing to real-world test images")
        print("\nExpected improvement: 60-70% → 80-90% on real-world images")

    else:
        print("\n✓ Your dataset has reasonable diversity!")
        print("\nOptional improvements:")
        print("1. Add edge cases (difficult lighting, partial occlusions)")
        print("2. Collect images from new environments")
        print("3. Test on diverse validation set")

    print("\n" + "="*60)


if __name__ == "__main__":
    analyze_dataset()
