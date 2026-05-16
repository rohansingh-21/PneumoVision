# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

def overlay_gradcam(img_array, heatmap):
    img = np.uint8(255 * img_array)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    output = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def display_gradcam(img_array_original, heatmap):
    gradcam_img = overlay_gradcam(img_array_original, heatmap)
    img = np.uint8(255 * img_array_original)

    plt.figure(figsize=(10, 5))
    for i, (image, title) in enumerate([
        (img, "Original"),
        (gradcam_img, "Grad-CAM")
    ]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
    plt.show()
