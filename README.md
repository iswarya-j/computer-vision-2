import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def load_image(path):
    """Loads an image or generates a synthetic one if path is None."""
    if path is not None:
        # Load as Grayscale (0 flag is crucial for MRA)
        img = cv2.imread(path, 0)
        if img is None:
            raise ValueError(f"Could not load image at {path}")
    else:
        print("[INFO] No path provided. Generating synthetic image for demo...")
        img = np.zeros((512, 512), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (400, 400), 255, -1)
        cv2.circle(img, (256, 256), 100, 0, -1)

        noise = np.random.normal(0, 20, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img


def visualize_decomposition(coeffs):
    level_count = len(coeffs) - 1
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"Wavelet Decomposition (Levels 1 to {level_count})", fontsize=16)

    # Approximation
    cA = coeffs[0]
    ax = fig.add_subplot(level_count + 1, 4, 1)
    ax.imshow(cA, cmap='gray')
    ax.set_title(f"Approx (LL{level_count})")
    ax.axis('off')

    for i, details in enumerate(coeffs[1:]):
        level_index = level_count - i
        cH, cV, cD = details
        row = i + 1

        ax1 = fig.add_subplot(level_count + 1, 4, row * 4 - 2)
        ax1.imshow(cH, cmap='gray')
        ax1.set_title(f"LH{level_index}")
        ax1.axis('off')

        ax2 = fig.add_subplot(level_count + 1, 4, row * 4 - 1)
        ax2.imshow(cV, cmap='gray')
        ax2.set_title(f"HL{level_index}")
        ax2.axis('off')

        ax3 = fig.add_subplot(level_count + 1, 4, row * 4)
        ax3.imshow(cD, cmap='gray')
        ax3.set_title(f"HH{level_index}")
        ax3.axis('off')

    plt.tight_layout()
    plt.show()


def extract_features(coeffs):
    features = {}

    def calc_entropy(matrix):
        p = np.abs(matrix.flatten())
        p = p / (np.sum(p) + 1e-10)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    cA = coeffs[0]
    level = len(coeffs) - 1
    features[f'LL{level}_Energy'] = np.sum(cA ** 2)
    features[f'LL{level}_Mean'] = np.mean(cA)

    for i, (cH, cV, cD) in enumerate(coeffs[1:]):
        curr_lvl = level - i
        for band, name in zip([cH, cV, cD], ['LH', 'HL', 'HH']):
            features[f'{name}{curr_lvl}_Energy'] = np.sum(band ** 2)
            features[f'{name}{curr_lvl}_Entropy'] = calc_entropy(band)
            features[f'{name}{curr_lvl}_StdDev'] = np.std(band)

    return pd.DataFrame([features]).T


# --- MAIN EXECUTION ---
if __name__ == "__main__":

    # Upload birdip.jpg to Colab before running
    image_path = "birdip.jpg"

    original_img = load_image(image_path)

    plt.figure(figsize=(6, 6))
    plt.imshow(original_img, cmap='gray')
    plt.title("Original Input Image")
    plt.axis('off')
    plt.show()

    coeffs = pywt.wavedec2(original_img, 'db2', level=3)

    print("Displaying Decomposition Stages...")
    visualize_decomposition(coeffs)

    print("\nExtracting Features...")
    feats = extract_features(coeffs)

    print("=" * 40)
    print("FINAL FEATURE VECTOR")
    print("=" * 40)
    print(feats)
