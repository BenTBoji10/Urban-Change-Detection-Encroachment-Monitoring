import cv2
import numpy as np

# -----------------------------
# LOAD IMAGE (CHANGE PATH)
# -----------------------------
image = cv2.imread(r"C:\Users\Boji Thomas\Desktop\urbanproject\aerial.jpg")

if image is None:
    print("Error: Image not found. Check your file path.")
    exit()

# Resize for faster processing
image = cv2.resize(image, (800, 600))

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Kernel for cleaning
kernel = np.ones((5, 5), np.uint8)

# -----------------------------
# 1. VEGETATION (GREEN)
# -----------------------------
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# 2. WATER (BLUE)
# -----------------------------
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# 3. ROADS (LIGHT AREAS)
# -----------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

# Remove vegetation & water from road mask
mask_road = cv2.bitwise_and(bright_mask, cv2.bitwise_not(mask_green))
mask_road = cv2.bitwise_and(mask_road, cv2.bitwise_not(mask_blue))
mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, kernel)

# -----------------------------
# 4. BUILDINGS (REMAINING AREA)
# -----------------------------
combined = cv2.bitwise_or(mask_green, mask_blue)
combined = cv2.bitwise_or(combined, mask_road)

mask_building = cv2.bitwise_not(combined)
mask_building = cv2.morphologyEx(mask_building, cv2.MORPH_OPEN, kernel)

# -----------------------------
# AREA CALCULATION
# -----------------------------
total_pixels = image.shape[0] * image.shape[1]

green_percent = (cv2.countNonZero(mask_green) / total_pixels) * 100
blue_percent = (cv2.countNonZero(mask_blue) / total_pixels) * 100
road_percent = (cv2.countNonZero(mask_road) / total_pixels) * 100
building_percent = (cv2.countNonZero(mask_building) / total_pixels) * 100

print("\n===== LAND COVER ANALYSIS =====")
print(f"Vegetation: {green_percent:.2f}%")
print(f"Water: {blue_percent:.2f}%")
print(f"Roads: {road_percent:.2f}%")
print(f"Buildings: {building_percent:.2f}%")

# -----------------------------
# DRAW CONTOURS
# -----------------------------
output = image.copy()

contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_road, _ = cv2.findContours(mask_road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_building, _ = cv2.findContours(mask_building, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(output, contours_green, -1, (0, 255, 0), 2)      # Green
cv2.drawContours(output, contours_blue, -1, (255, 0, 0), 2)      # Blue
cv2.drawContours(output, contours_road, -1, (0, 0, 255), 2)      # Red
cv2.drawContours(output, contours_building, -1, (0, 255, 255), 1) # Yellow

# -----------------------------
# DISPLAY
# -----------------------------
cv2.imshow("Segmented Image", output)
cv2.imshow("Vegetation Mask", mask_green)
cv2.imshow("Water Mask", mask_blue)
cv2.imshow("Road Mask", mask_road)
cv2.imshow("Building Mask", mask_building)

cv2.waitKey(0)
cv2.destroyAllWindows()
