# 3D-Unet Segmentation of Lung lobes CT volumes

This project aims at predicting segmented images based on CT scans of Lungs (3d volumes).

I uploaded the .py files and a Google Colab notebook where I trained the 3d-Unet model on the free GPU.

# Dataset:

format = (slices x rows x columns)

scans = 51 .nrrd files of size 256x256x256.

masks = 51 .nrrd files of size 256x256x256.

Orininal size:

![](/original_img_and_mask.png)

Patch of size 128x128x128.

![](/scan_and_mask.png)

# Code:

1. Importing files.
2. Patch Extraction.
3. Model
4. Training
5. Cross validation
6. Testing
7. Loss and Loss Validation Plots
8. Comparing predicted segmantation with scans and masks
9. Future work for improving results.
