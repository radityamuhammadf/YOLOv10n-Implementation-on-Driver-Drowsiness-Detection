import cv2
from ultralytics import YOLOv10
import time
import os
import supervision as sv

current_directory = os.getcwd()

def main():
    # Load the model
    model = YOLOv10(os.path.join(current_directory, r"training_result\best.pt"))

    
    # Start the webcam
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(os.path.join(current_directory, r"test_video\10-MaleGlasses-Trim.avi"))  
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps
    frame_number = 0
    print("FPS: ", fps)

    # Initialize the dictionary to keep track of detection times and last durations
    detections = {
        'closed-eyes': {'duration': 0, 'frame_count': 0, 'last_seen_frame': None},
        'yawn': {'duration': 0, 'frame_count': 0, 'last_seen_frame': None}
    }

    drowsy_state = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1 

        # Perform inference
        results = model.predict(frame, conf=0.6)

        # Track which classes are currently detected
        current_detections = set()

        # Draw the bounding boxes by iterating over the results
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, class_id = r
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_name = model.names[int(class_id)]
                current_detections.add(class_name)
                print(result)

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Display class label and confidence
                cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)

                # Update detection frame counts
                if detections[class_name]['last_seen_frame'] is None:
                    detections[class_name]['last_seen_frame'] = frame_number
                else:
                    detections[class_name]['frame_count'] += frame_number - detections[class_name]['last_seen_frame']
                    detections[class_name]['last_seen_frame'] = frame_number
                detections[class_name]['was_detected'] = True  # params to state the detected state 

        # Reset the durations when the class are not detected anymore
        for class_name in detections:
            if class_name not in current_detections:  # check whether this current frame still detecting the same class
                detections[class_name]['was_detected'] = False  # reset the previously detected class state
                detections[class_name]['frame_count'] = 0  # reset counted frame value
                detections[class_name]['last_seen_frame'] = None  # reset previously seen frame

        # Convert frame counts to time using FPS
        for class_name in detections:
            # this default frames per second are 30FPS -> and the duration for each frame are 1/30 ~ 0.33
            # for instance, if the closed-eyes class are detected for 65 frames consecutively
            # that means the durations of detected closed-eyes are 65*0.03 which are 1.95s
            detections[class_name]['duration'] = detections[class_name]['frame_count'] * frame_duration

        # Detect drowsiness based on the duration of closed-eyes and yawn
        closed_eyes_duration = detections['closed-eyes']['duration']
        yawn_duration = detections['yawn']['duration']
        
        # Logic for detecting drowsiness
        if closed_eyes_duration > 0.5 or yawn_duration > 5.0:  # thresholds in seconds
            drowsy_state = True
        else:
            drowsy_state = False

        # debugging print state
        print(f"Closed-eyes duration: {closed_eyes_duration:.2f} seconds")
        print(f"Yawn duration: {yawn_duration:.2f} seconds")
        print(f"Drowsy state: {drowsy_state}")
        
        # Drowsy State Branch Logic
        if drowsy_state is True:
            cv2.rectangle(frame, (500, 20), (640, 60), (255, 255, 255), -1)
            cv2.putText(frame, 'Drowsy', (500, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        """
        Static Information Display Function
        """
        # drawing and writing the annotation
        cv2.rectangle(frame, (0, 10), (200, 60), (255, 255, 255), -1)
        y_offset = 30
        cv2.putText(frame, f'closed-eyes: {closed_eyes_duration:.2f} s', (10, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2)
        y_offset += 20
        cv2.putText(frame, f'yawn: {yawn_duration:.2f} s', (10, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2)
        y_offset += 20

        # Display the frame
        cv2.imshow('Inference-YOLOv10n', frame)

        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
