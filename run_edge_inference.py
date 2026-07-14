import cv2
import numpy as np
import ncnn
import time
import os

# 1. Initialize NCNN structures
net = ncnn.Net()
net.load_param("unet_jit.ncnn.param")
net.load_model("unet_jit.ncnn.bin")

# 2. Automatically crawl the directory to find the real image file
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
frame = None

for root, dirs, files in os.walk("."):
    if any(ignored in root for ignored in [".git", "venv", "__pycache__"]):
        continue
    for file in files:
        if file.lower().endswith(valid_extensions):
            file_path = os.path.join(root, file)
            test_img = cv2.imread(file_path)
            if test_img is not None and test_img.size > 5000:
                frame = test_img
                break
    if frame is not None:
        break

if frame is None:
    print("Error: Could not find a valid test image.")
    exit()

orig_h, orig_w = frame.shape[:2]

# 3. Standardize dimensions & normalize data according to UNet specifications
resized_frame = cv2.resize(frame, (256, 256))
rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
input_tensor = np.transpose(rgb_frame, (2, 0, 1)).astype(np.float32) / 255.0
mat_in = ncnn.Mat(input_tensor)

# 4. Extract forward layer passes
start_time = time.time()
ex = net.create_extractor()
ex.input("in0", mat_in)
ret_val, mat_out = ex.extract("out0")
latency = (time.time() - start_time) * 1000

print(f"\n==========================================")
print(f"      EDGE ENGINE RUNTIME METRICS        ")
print(f"==========================================")
print(f" Inference Processing Speed: {latency:.2f} ms")
print(f"==========================================\n")

# 5. Extract output matrix map
output_mask = np.array(mat_out)[0]

# DYNAMIC THRESHOLD RULE: Calculate the top 90th percentile of the output array values
# This isolates the top 10% strongest features (the actual lanes) and drops the background haze.
cutoff_val = np.percentile(output_mask, 90)

# Build a binary mask using our dynamic cutoff line
binary_mask = (output_mask > cutoff_val).astype(np.uint8) * 255
final_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

# 6. Target-Specific Mask Overlay Blending (Wipes out the full-screen green tint)
blended_output = frame.copy()
green_layer = np.zeros_like(frame)
green_layer[:, :, 1] = 255  # Pure solid green channel map

# Identify only the precise coordinates where the model scores are high
lane_pixels = final_mask > 0

# Alpha blend ONLY the pixels matching the lane locations!
blended_output[lane_pixels] = cv2.addWeighted(
    frame, 0.3, green_layer, 0.7, 0
)[lane_pixels]

# 7. Render Output Window Dashboard
cv2.imshow("Custom Edge AI Segmentation Dashboard", blended_output)
cv2.waitKey(0)
cv2.destroyAllWindows()