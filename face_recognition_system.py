import face_recognition
import cv2
import pickle
import os
import shutil
import sys
import numpy as np

# --- Utility Function to Handle Camera Access ---
def get_video_capture():
    """
    Attempts to get a video capture object from the default camera.
    Tries multiple camera indices if the default fails.
    """
    print("Attempting to access your webcam...")
    for i in range(5):  # Try indices 0, 1, 2, 3, 4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Successfully opened camera at index {i}.")
            return cap
        cap.release()
    
    print("❌ Failed to open any webcam. Please ensure your camera is connected and not in use by another application.")
    print("If you are on Windows, check 'Privacy & security > Camera' settings.")
    return None

# --- Part 1: Training Script with a Camera Registration Feature ---
def register_new_face():
    """
    Displays a live webcam feed, captures an image on key press,
    prompts for a name, and adds it to the known faces dataset.
    """
    print("\n--- Face Registration ---")
    print("A live camera feed will now open. Press 'c' to capture your face.")

    video_capture = get_video_capture()
    if not video_capture:
        return

    # Create the known_faces folder if it doesn't exist
    known_faces_folder = "known_faces"
    if not os.path.exists(known_faces_folder):
        os.makedirs(known_faces_folder)

    frame_to_save = None
    face_detected = False

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("❌ Failed to grab a frame from the camera.")
            break

        # Display the live feed
        cv2.imshow('Face Registration - Press "c" to capture', frame)
        
        # Capture on 'c' key press
        if cv2.waitKey(1) & 0xFF == ord('c'):
            frame_to_save = frame
            # Check for a face in the captured image
            rgb_frame = cv2.cvtColor(frame_to_save, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) > 0:
                face_detected = True
            break
            
    # Release camera and destroy windows after capture
    video_capture.release()
    cv2.destroyAllWindows()

    if face_detected:
        name = input("Face detected! Please enter the name of the person: ").strip()
        if not name:
            print("Name cannot be empty. Registration cancelled.")
            return

        filename = f"{name.replace(' ', '_').lower()}.jpg"
        image_path = os.path.join(known_faces_folder, filename)
        cv2.imwrite(image_path, frame_to_save)
        print(f"✅ Face for '{name}' successfully registered and saved as {filename}.")
    else:
        print("❌ No face found in the captured image. Please try again with better lighting and a clear view.")

def train_and_save_encodings(folder_path, output_file):
    """
    Loads images from a folder, encodes faces, and saves them to a file.
    """
    known_faces_encodings = []
    known_faces_names = []

    print("\n--- Training the System ---")
    print("Starting to load and encode known faces...")
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            person_name = os.path.splitext(filename)[0].replace("_", " ").title()
            
            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) > 0:
                    known_faces_encodings.append(face_encodings[0])
                    known_faces_names.append(person_name)
                    print(f"✅ Encoded face for: {person_name}")
                else:
                    print(f"❌ No face found in {filename}. Skipping...")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}. Skipping...")
    
    with open(output_file, 'wb') as f:
        pickle.dump((known_faces_encodings, known_faces_names), f)
    
    print(f"\nTraining complete! Encodings saved to {output_file}")

# --- Part 2: Live Detection Script ---
def run_live_recognition(encodings_file):
    """
    Runs a live camera feed to recognize known faces.
    """
    print("\n--- Live Recognition ---")
    if not os.path.exists(encodings_file):
        print("Error: Encodings file not found. Please register faces first and train the system.")
        return
        
    try:
        with open(encodings_file, 'rb') as f:
            known_faces_encodings, known_faces_names = pickle.load(f)
    except (IOError, pickle.UnpicklingError) as e:
        print(f"Error loading encodings file: {e}")
        return

    video_capture = get_video_capture()
    if not video_capture:
        return

    print("Live detection started. Press 'q' to quit.")
    cv2.namedWindow('Visitor Recognition')

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab a frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                box_color = (0, 0, 255)  # Red for unknown

                face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] < 0.6:
                        name = known_faces_names[best_match_index]
                        box_color = (0, 255, 0)  # Green for known

                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imshow('Visitor Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nProgram stopped by user. Releasing resources...")
        
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

# --- Main Menu ---
if __name__ == "__main__":
    encodings_file = "encodings.pkl"
    known_faces_folder = "known_faces"

    while True:
        print("\n--- Face Recognition System Menu ---")
        print("1. Register a new face from camera")
        print("2. Train the system (after adding/removing faces)")
        print("3. Start live face recognition")
        print("4. Clear all registered faces")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ").strip()

        if choice == '1':
            register_new_face()
        elif choice == '2':
            if os.path.exists(known_faces_folder):
                train_and_save_encodings(known_faces_folder, encodings_file)
            else:
                print("The 'known_faces' folder does not exist. Please register faces first.")
        elif choice == '3':
            run_live_recognition(encodings_file)
        elif choice == '4':
            if os.path.exists(known_faces_folder):
                shutil.rmtree(known_faces_folder)
                if os.path.exists(encodings_file):
                    os.remove(encodings_file)
                print("All registered faces and encodings have been cleared.")
            else:
                print("No faces to clear. The folder does not exist.")
        elif choice == '5':
            print("Exiting the application.")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")