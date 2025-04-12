import cv2
import requests
import numpy as np

# Replace with your IP camera URL and IP address
ip_camera_url = "http://<YOUR_IP_CAMERA_URL>/video"
camera_ip = "rtsp://admin:admin@192.168.11.60:554/2"
post_url = "http://192.168.11.101:5000/motion-detected"  # Replace with your API endpoint

# Initialize video capture
cap = cv2.VideoCapture(camera_ip)

# Initialize the first frame for motion detection
first_frame = None
print("oop started")
# Initialize the count outside the conditions
count = 0
motion_detected_sent = False  # Flag to track if the motion detected message has been sent
while True:
    # Read the current frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize the first frame
    if first_frame is None:
        first_frame = gray
        continue

    # Compute the absolute difference between the current frame and the first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the threshold image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the thresholded image
    cv2.imshow("IP Camera d", thresh)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    # print("looking for contiurs")
    for contour in contours:
        if cv2.contourArea(contour) < 900:  # Adjust the threshold as needed
            motion_detected = True
            break

        

    # Assuming motion_detected is a boolean indicating if motion is currently detected

    if motion_detected:
        if motion_detected_sent:
            if count >= 60:
                print("Motion stopped! time limit Sending POST request...")
                response = requests.post(post_url, json={"status": "motion_stopped", "camera_ip": camera_ip})
                print(f"Response: {response.status_code}, {response.text}")
                motion_detected_sent = False  # Reset the flag to allow for future detections
                count = 0
            else:
                count += 1
        else:
            print("Motion detected! Sending POST request...")
            response = requests.post(post_url, json={"status": "motion_detected", "camera_ip": camera_ip})
            print(f"Response: {response.status_code}, {response.text}")
            motion_detected_sent = True  # Set the flag to indicate the message was sent
    else:
        if motion_detected_sent:
            print("Motion stopped! Sending POST request...")
            response = requests.post(post_url, json={"status": "motion_stopped", "camera_ip": camera_ip})
            print(f"Response: {response.status_code}, {response.text}")
            motion_detected_sent = False  # Reset the flag to allow for future detections
            count = 0


    # Optional: Display the frame (for testing)
    cv2.imshow("IP Camera Feed", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
