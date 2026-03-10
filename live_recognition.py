import face_recognition
import cv2
import pickle
import os
import numpy as np

def run_live_recognition(encodings_file):
    """
    Runs a live camera feed to identify and highlight both known and unknown faces.
    
    Args:
        encodings_file (str): Path to the file containing known face encodings.
    """
    # Load the known face encodings and names from the file
    if not os.path.exists(encodings_file):
        print("Error: encodings file not found. Please run the training script first.")
        return
    
    try:
        with open(encodings_file, 'rb') as f:
            known_faces_encodings, known_faces_names = pickle.load(f)
    except (IOError, pickle.UnpicklingError) as e:
        print(f"Error loading encodings file: {e}")
        return

    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)  # 0 is the default webcam

    print("Live detection started. Known faces will be green, unknown faces will be red. Press 'q' to quit or Ctrl+C to stop.")

    try:
        while True:
            # Grab a single frame from the video feed
            ret, frame = video_capture.read()
            
            if not ret:
                print("Failed to grab frame.")
                break

            # Convert the frame from BGR (OpenCV) to RGB (face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find all faces and their encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Loop through each face found
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                box_color = (0, 0, 255)  # Red for unknown

                # Use face_distance to find the best match
                face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
                
                # Find the index of the best match (the smallest distance)
                best_match_index = np.argmin(face_distances)
                
                # Check if the best match is within the allowed tolerance (default is 0.6)
                if face_distances[best_match_index] < 0.6:
                    name = known_faces_names[best_match_index]
                    box_color = (0, 255, 0)  # Green for known

                # Draw the bounding box and name for both known and unknown faces
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the video stream
            cv2.imshow('Visitor Recognition', frame)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nProgram stopped by user. Releasing resources...")
        
    finally:
        # This block ensures the webcam and windows are always closed
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    encodings_file = "encodings.pkl"
    run_live_recognition(encodings_file)