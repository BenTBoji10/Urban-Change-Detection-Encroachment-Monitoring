# Urban-Change-Detection-Encroachment-Monitoring
Urban expansion often leads to unauthorized constructions and reduction of green cover. This project presents an automated Urban Change Detection system that analyzes aerial images to classify land cover into:

🏢 Buildings

🛣 Roads

🌳 Vegetation

🌊 Water Bodies

The system uses image segmentation techniques to identify and quantify these regions, enabling efficient monitoring of land use changes.

🎯 Objective

The main objective of this project is to:

Perform land cover classification using image processing techniques

Detect potential encroachments

Calculate the percentage area distribution of each land type

Support sustainable urban planning decisions

🛠 Technologies Used

Python

OpenCV

NumPy

⚙️ Methodology

Image Preprocessing

Resize input image

Convert from BGR to HSV color space

Segmentation

Vegetation detection using green HSV range

Water detection using blue HSV range

Road detection using grayscale thresholding

Buildings identified as remaining regions

Noise Removal

Morphological operations (Opening & Closing)

Contour Detection

Draw contours for visual representation

Area Analysis

Compute percentage of each land cover category

📊 Output

Segmented image with highlighted contours

Individual masks for vegetation, water, roads, and buildings

Percentage distribution of land cover classes
