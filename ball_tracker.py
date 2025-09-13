import cv2

def ball_detect(img, color_finder, hsv_values):
    ball_x = 0
    ball_y = 0
    img_with_contours = None
    if img is None:
        return None, ball_x, ball_y
    imggray_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, mask = color_finder.update(imggray_hsv, hsv_values)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour_index = 0
        max_area = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour_index = i
        largest_contour = contours[largest_contour_index]
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            ball_x = int(M["m10"] / M["m00"])
            ball_y = int(M["m01"] / M["m00"])
        else:
            ball_x = 0
            ball_y = 0
        img_with_contours = img.copy()
        cv2.drawContours(img_with_contours, [largest_contour], -1, (255, 0, 0), 2)
        cv2.circle(img_with_contours, (ball_x, ball_y), 5, (255, 0, 0), -1)
    return img_with_contours, ball_x, ball_y
