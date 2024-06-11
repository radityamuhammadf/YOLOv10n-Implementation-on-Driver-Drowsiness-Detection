import cv2
from ultralytics import YOLOv10
import time
import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import openpyxl
from datetime import datetime

def main():
    current_directory = os.getcwd()
    # Load the model
    model = YOLOv10(os.path.join(current_directory, r"training_result\best.pt"))

    # Excel Export Workbook Initiation
    # Create Excel workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Videos Metadata"
    # Create Header Row
    # Write the header row
    headers = [
        "Video Name", "Light Sufficient", "Looking LR", 
        "Detection Accuracy", "False Positive Rate", 
        "Inference Time", "Profiler Result"
    ]
    ws.append(headers)

    #creating videos_metadata dictionary
    videos_metadata = {
        'webcam_input': {
            'path': 0,
            'light_sufficient': True,
            'looking_lr': False,
            'detected_drowsiness': [],
            'ground_truth_drowsiness': [],
            'detection_accuracy':0,
            'false_positive_rate':0,
            'inference_time':0,
            'profiler_result':""
        },
        'debug_video_sample': {
            'path': os.path.join(current_directory, r'test_video\debugging_sample.avi'),
            'light_sufficient': True,
            'looking_lr': False,
            'detected_drowsiness': [],
            'ground_truth_drowsiness': [],
            'detection_accuracy':0,
            'false_positive_rate':0,
            'inference_time':0,
            'profiler_result':""
        },
    }
    # #example of adding metadata (used later for the four video data)
    # videos_metadata['video_name'] = {
    #     'path': os.path.join(current_directory, 'path/to/another/vid'),
    #     'light_sufficient': False,
    #     'looking_lr': True,
    #     'detected_drowsiness': [0.1, 0.3, 0.5],  # Example list of floats
    #     'ground_truth_drowsiness': [0.2, 0.4, 0.6],  # Example list of floats,
    #     'detection_accuracy':0,
    #     'false_positive_rate':0,
    #     'inference_time':0,
    #     'profiler_result':""
    # }

    #iterating every metadata element in every 'video_name' (videos_metadata members) element
    for video_name, metadata in videos_metadata.items():
        temp_inference_time = []
        video_path = metadata['path']
        # Start the webcam
        cap = cv2.VideoCapture(video_path)
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

        # videos_metadata

        # Drowsy state declaration
        prev_drowsy_state=False
        drowsy_state = False

        #USING TRY-EXCEPT-FINALLY Block to prevent RuntimeError: Profiler didn't finish running error
        #oh the memory of the OOP course
        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_number += 1 

                    # Perform inference - record resource and inference time
                    with record_function("model_inference"):
                        inference_start=time.time()
                        results = model.predict(frame, conf=0.6)
                        inference_end=time.time()
                        temp_inference_time.append(inference_end-inference_start) 

                    # Track which classes are currently detected
                    current_detections = set()

                    #TRACKING LOGIC HERE
                    #IF RESULTS IS NULL THEN (i still don't know tho)
                    
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
                        #record the first time drowsiness detected by multiplying current frame number with frame duration
                        if prev_drowsy_state is False:
                            metadata['detected_drowsiness'].append(frame_number*frame_duration)                            
                        prev_drowsy_state=True
                    else:
                        prev_drowsy_state = False


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
                    cv2.imshow('Inference-YOLOv8n', frame)
                    
                    # Break the loop
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # For Inference Time (counting inference time average):
                metadata['inference_time'] = sum(temp_inference_time) / len(temp_inference_time)
            cap.release()    

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Filter profiling results to show only model_inference resource usage
            key_averages = prof.key_averages()
            inference_key_averages = [evt for evt in key_averages if "model_inference" in evt.key]
            # Accessing the profiling result
            for evt in inference_key_averages:
                metadata['profiler_result']=f"{evt}"
            
            print("Profiling Result:\n",metadata['profiler_result']) #debugging prompt

            # Measure and print other metadata
            # Detection Accuracy Measurements
            if len(metadata['ground_truth_drowsiness']) != 0:
                if len(metadata['detected_drowsiness'])<=len(metadata['ground_truth_drowsiness']): #stating if there's no false positive by make sure the ground truth are more than equal to detected state 
                    metadata['detection_accuracy']=len(metadata['detected_drowsiness'])/len(metadata['ground_truth_drowsiness'])
                    metadata['false_positive_rate']=0
                else: #for if the detected state exceed the ground truth 
                    # find exceeding values by subsracting detected drowsiness with ground truth value
                    exceed_value=len(metadata['detected_drowsiness'])-len(metadata['ground_truth_drowsiness'])
                    metadata['detection_accuracy']=(len(metadata['ground_truth_drowsiness'])-exceed_value)/len(metadata['ground_truth_drowsiness'])
                    metadata['false_positive_rate']=exceed_value/len(metadata['ground_truth_drowsiness'])
            print(f"Average Infqerence Time: {metadata['inference_time']*1000:.3f}ms") # debugging prompt
            
            #Append Row Using Video Metadata Information
            row = [
                video_name,
                metadata['light_sufficient'],
                metadata['looking_lr'],
                metadata['detection_accuracy'],
                metadata['false_positive_rate'],
                metadata['inference_time'],
                metadata['profiler_result']
            ]
            ws.append(row)
    
    # Export Excel File
    # Generate filename
    now = datetime.now()
    date_str = now.strftime("%b%d-%H%M")
    dynamic_filename = f"YOLOv10-VideoTest-Debug-{date_str}.xlsx" #For Debugging
    # dynamic_filename = f"VideoTest-Main-{date_str}.xlsx" #For The Main Test Set
    output_file=os.path.join(current_directory, f"video-test_result/{dynamic_filename}")
    wb.save(output_file)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
