from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8n model

cap = cv2.VideoCapture(0)  # For opening camera or video file, to use video file replace 0 with 'path_to_video.mp4'

while True:
    ret, frame = cap.read() #for reading a frame, if ret is True, then the cap.read() was successful
    if not ret:
        print("Failed to grab frame")
        break
    
    results = model(frame, stream=True)  # Perform inference on the frame, inference meaning running a model on new data to get predictions 
    
    #basically look at the frame from the model and tell me what the objects are/predictions aka inference
    
    # What happens internally here during inference, 1. Preprocess the image (resizing, normalizing colors, converting to tensor)
    #2.Feed the preprocessed image into the neural network (YOLO model) to get raw predictions like bounding boxes(boxes around the people), class IDs(for this case person), and confidence scores(how sure it is a person).
    #3.Postprocess the raw predictions to filter out low-confidence detections and apply non-maximum suppression to remove overlapping boxes. (NMS for removing the overlaps in the boxes)
    #4 return the structured results or the predictions! :DD
    
    person_count = 0
    
    # Loop through detections found in frame
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        for box in boxes:
            cls_id = int(box.cls[0])  # Class ID
            if cls_id == 0:  # Class ID 0 corresponds to 'person' in COCO dataset
                person_count += 1
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0]) #Coordinates of bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw rectangle around person, the 0,255,0 is the color green in BGR format
                cv2.putText(frame, f'Person {person_count}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Draws a text label above the bounding box
                
    # Display person count on frame
    cv2.putText(frame, f'Total Persons: {person_count}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow('YOLOv8 People Counter', frame)  # Show the frame with detections
    
    if cv2.waitKey(1) & 0xFF == ord("q"): # Exit on 'q' key press
        break
    
cap.release()
cv2.destroyAllWindows()
    

