import matplotlib.pyplot as plt
import cv2

# Creating subplots using matplotlib | it will also save the plot to a file
def visualization(image, segments_fz):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    axes[0].imshow(image)  # Displaying the original image
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # Hides the axis

    axes[1].imshow(segments_fz)  # Displaying the segmented image
    axes[1].set_title('Image post Felzenszwalb segmentation')
    axes[1].axis('off')
    # Save the figure to a file
    plt.savefig('E:\PyTorch\Object Detection\Selective Search\Images\Felzenszwalb_Generated_Image_1.jpg')
    plt.show()
    plt.close(fig)  # Close the plot to free up memory


def saving_bounding_boxes(image, candidates):

    # Create a copy of the image to draw bounding boxes on
    image_with_bbs = image.copy()

    # Draw each bounding box on the image
    for x, y, w, h in candidates:
        cv2.rectangle(image_with_bbs, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite('E:\PyTorch\Object Detection\Selective Search\Images\Bounding_Box_Selective_Search_1.jpg', image_with_bbs) # Save the image with bounding boxes drawn on it using imwrite() from OpenCV
    cv2.imshow('Image with Bounding Boxes', image_with_bbs) # Display the image with bounding boxes drawn on it using imshow() from OpenCV
    cv2.waitKey(0)
    cv2.destroyAllWindows()
