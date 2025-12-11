import cv2

def ball_detect(img, color_finder, hsv_values, min_area=100): # Added min_area param
    ball_x = 0
    ball_y = 0
    
    if img is None:
        return img, ball_x, ball_y

    imggray_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, mask = color_finder.update(imggray_hsv, hsv_values)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour_index = -1
        max_area = 0
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # CRITICAL FIX: Only consider contours larger than noise
            if area > max_area:
                max_area = area
                largest_contour_index = i
        
        # Only proceed if we actually found a contour bigger than the noise threshold
        if largest_contour_index != -1 and max_area > min_area:
            largest_contour = contours[largest_contour_index]
            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                ball_x = int(M["m10"] / M["m00"])
                ball_y = int(M["m01"] / M["m00"])

    # Return the ORIGINAL image without drawings
    return img, ball_x, ball_y