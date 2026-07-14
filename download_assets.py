from huggingface_hub import hf_hub_download

print("Downloading parameter file...")
hf_hub_download(repo_id="nickpai/lane-detection-unet-ncnn", filename="unet_depthwise/unet_depthwise_jit.ncnn.param", local_dir=".")

print("Downloading binary weights file...")
hf_hub_download(repo_id="nickpai/lane-detection-unet-ncnn", filename="unet_depthwise/unet_depthwise_jit.ncnn.bin", local_dir=".")

print("Done! Check your folder structure now.")