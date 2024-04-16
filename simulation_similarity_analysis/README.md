# Similarity analysis
This directory contains the file and the result of the similarity analysis
- `utils.py`: Contains a **Comparator** class that allows easy and faster (with multiprocessing) comparison between sample from one or multiple datasets.
- `similarity_study.py`: Use of the Comparator class to make plot from the comparison done.
- `xxx_analysis`: Folder containing the plots for the study of xxx parameters

The analysis contains the mean of the similarity (or the difference ) between 2 samples separated by a specific difference in the initial starting parameter.
Here is some of the methods used to compare the image:
- Absolute difference: It returns an image that is simply the absolute difference between the 2 images to compare. Easy to visualise but not as useful as a scalar indicator.
- The mean absolute error: The mean of the index above over all the image
- The root mean squared error: The root of the mean of the squared of the first method
- The max absolute error: The maximum of the first method
- Histogram correlation: We take the histogram of the 2 images, and we compute the correlation between each other.
- SSIM: Structural Similarity Index Measure: A common comparison method between 2 images. [wikipedia](https://en.wikipedia.org/wiki/Structural_similarity_index_measure)