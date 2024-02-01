import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import threshold_otsu, threshold_li, try_all_threshold
from skimage.morphology import area_closing, area_opening, binary_erosion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

filepaths = glob('dataset/*.jpg')
# fig, axis = plt.subplots(1, len(filepaths), figsize=(16, 8))
# for x, ax in zip(filepaths, axis.flatten()):
#     ax.imshow(imread(x))
#     ax.set_title(x.split('/')[1])
# print("The shape of the image is:", imread(filepaths[0]).shape)

oneFrame = imread(filepaths[2])
gray_oneFrame = rgb2gray(oneFrame)
thresh = threshold_otsu(gray_oneFrame)
binarized = gray_oneFrame < thresh
# imshow(binarized)
# fig, ax = try_all_threshold(gray_oneFrame, figsize=(10, 8), verbose=False)


closed = area_closing(binarized)
opened = area_opening(closed)
eroded = binary_erosion(opened)
# imshow(eroded)
# plt.show()

label_im = label(opened)
regions = regionprops(label_im)
# imshow(label_im)
# plt.show()

masks = []
bbox = []
for num, x in enumerate(regions):
    area = x.area
    convex_area = x.convex_area
    if num != 0 and num != 1 and x.area >= 100:
        masks.append(regions[num].convex_image)
        bbox.append(regions[num].bbox)
count = len(masks)
fig, axis = plt.subplots(10, int(count / 10), figsize=(15, 6))
for ax, box, mask in zip(axis.flatten(), bbox, masks):
    image = gray_oneFrame[box[0]:box[2], box[1]:box[3]] * mask
    ax.imshow(image)
    ax.axis('off')
plt.show()
properties = ['area', 'convex_area', 'bbox_area',
              'major_axis_length', 'minor_axis_length',
              'perimeter', 'equivalent_diameter',
              'mean_intensity', 'solidity', 'eccentricity', 'centroid']


# Create the DataFrame
# df = pd.DataFrame(regionprops_table(label_im, gray_oneFrame, properties=properties))

# Export the DataFrame to a CSV file
# df.to_csv('region_properties.csv', index=False)