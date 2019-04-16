# Image-stitcher

This is a basic image stitching program written in Python3. Most implementations of image stitching in python make use of the opencv library. This implementation is aimed towards understanding the concept behind image stitching and how it is done.

## Libraries Used
1. Numpy
2. Opencv
3. Matplotlib
4. random
5. itertools

## Steps to stitch two images
1. Identify which of the two images is the left image and which one is the right image. This is to be done manually. There are some programs that will allow you to "Recognize Panaromas" but that is beyond the scope of this project.
2. Convert both images to grayscale
3. Find the harris features in both the images. Other methods to find feature points can be used too to identify features in a given image. These features are usually the distinguising characteristics between the multiple components in the image. Eg: edge, corners, etc.
4. Find the descriptors of these feature points using SIFT. Again, other methods can be used too. You can try to make a matrix of the neighboring pixels of the feature point and flatten these out to create a descriptor for that feature.
5. Do these steps for both the images.
6. Find the distance between each descriptor in both the images using euclidean distance or hamming distance. Any other method is okay too.
7. Select the points that have the smallest distance between their descriptors. You can do this either by setting a threshold value that decides what the maximum distance should be or you could sort the point pairs (Matches) based on their distances and select the top few hundered points. These are the putative points. These points will represent the common part of both the images.
8. Using the putative points, find the homography matrix of one of the images with respect to the other (For example, homography of the left image with respect to the right image). This can be done by running the RANSAC algorithm.
9. Once the homography matrix is found, use this to translate the original image (Left image).
10. Append the other image (Right image) to this translated image.
11. The result is the stitched image

#### The following illustrates the images chosen and their harris features and putative points and the final result.

##### Left Image

![left.gif](https://github.com/kushalmehta13/Image-stitcher/blob/master/left.gif)

##### Right Image
![right.gif](https://github.com/kushalmehta13/Image-stitcher/blob/master/Right.gif)

#### Result


