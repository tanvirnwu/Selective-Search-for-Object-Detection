'''
First install the required packages using the following commands if needed:
Install the required packages:
!pip install selectivesearch
!pip install torch_snippets
'''
# Import the required Libraries
import numpy as np
from torch_snippets import read
from skimage.segmentation import felzenszwalb
import selectivesearch
from Utils import visualization, saving_bounding_boxes

# Read the image using the read() function from torch_snippets
image = read('E:\PyTorch\Object Detection\Selective Search\Images\KCS_Cat_1.jpg',1)
# Perform felzenszwalb segmentation on the image
segments_fz = felzenszwalb(image, scale=200)

'''
Now we will visualize the image and the segmented image using both the torch_snippets and matplotlib libraries. 
I am running the visualization() function from Utils.py to display the original image, the segmented image and then saving the
plot to a file in the Images folder in this repo.
'''
# Creating subplots using torch_snippets
# subplots([image, segments_fz], titles=['Original Image','Image post \nfelzenszwalb segmentation'], figsize=(10,10), nc=2)
visualization(image, segments_fz) # Visualize the image and the segmented image using the visualization() function from Utils.py

'''
Now we will extract the candidates from the image using the extract_candidates() function from SelectiveSearch.py.
'''
# Extract Candidates
def extract_candidates(image):
    img_lbl, regions = selectivesearch.selective_search(image, scale=200, min_size=100) # Perform selective search on the input image
    img_area = np.prod(image.shape[:2]) # Calculate the total area of the image
    candidates = []
    for r in regions: # Iterate over the regions returned by selective search on the image
        if r['rect'] in candidates: continue # Skip if the rectangle is already added
        if r['size'] < (0.05*img_area): continue # Skip if the size of the rectangle is less than 5% of the total area of the image
        if r['size'] > (1*img_area): continue # Skip if the size of the rectangle is greater than 100% of the total area of the image
        x, y, w, h = r['rect'] # Extract the coordinates of the rectangle
        candidates.append(list(r['rect'])) # Append the coordinates of the rectangle to the candidates list
    return candidates


# Extracting the candidates from the image using the extract_candidates()
candidates = extract_candidates(image)
# Display the image with bounding boxes drawn on it using the show() function from torch_snippets
# show(image, bbs = candidates)

# Saving the image with bounding boxes drawn on it using the saving_bounding_boxes() function from Utils.py
saving_bounding_boxes(image, candidates)
