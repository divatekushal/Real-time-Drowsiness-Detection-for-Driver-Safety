import cv2
import numpy as np
import winsound
import tensorflow as tf
import tkinter
import customtkinter
from PIL import ImageTk, Image

new_model = tf.keras.models.load_model("my_model.h5")
detection_running = False
import threading

def start_drowsiness_detection():
    global detection_running_thread
    detection_running_thread = threading.Thread(target=drowsiness_detection_loop)
    detection_running_thread.start()

def drowsiness_detection_loop():
    global detection_running

    # Set the flag to indicate that the detection loop is running
    detection_running = True

    # Initialize cascade classifiers for face and eyes
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Initialize the webcam
    cap = cv2.VideoCapture(1)  # Change the argument to 0 if you're using the default webcam

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FPS, 5)
    counter = 0

    while detection_running:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        # Detect only the first face
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Extracting coordinates of the first face

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Get the region of interest (ROI) which is the face region
            faceROI = gray[y:y+h, x:x+w]

            # Detect eyes within the face region
            eyes = eyeCascade.detectMultiScale(faceROI)

            for (ex, ey, ew, eh) in eyes:
                # Extract and preprocess the eye region for prediction
                eye_img = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
                eye_img_resized = cv2.resize(eye_img, (224, 224))
                eye_img_normalized = eye_img_resized / 255.0
                eye_img_final = np.expand_dims(eye_img_normalized, axis=0)

                # Predict using the pre-trained model
                predictions = new_model.predict(eye_img_final)

                # Check if eyes are closed
                if predictions[0][0] < 0.5:  # Assuming predictions is a 2D array
                    counter += 1
                    if counter > 10:
                        # Alert for drowsiness detected
                        cv2.putText(frame, "Sleeping", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        winsound.Beep(2500, 1500)
                        counter = 10  # Reset the counter back to 10
                else:
                    # Reset the counter to 0 if eyes are open
                    counter = 0

        # Display the frame
        cv2.imshow("Drowsiness Detection", frame)

        # Break the loop if 'q' is pressed or detection_running is set to False
        if cv2.waitKey(1) & 0xFF == ord('q') or not detection_running:
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

def stop_drowsiness_detection():
    global detection_running
    detection_running = False

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("green")  # Themes: blue (default), dark-blue, green

app = customtkinter.CTk()  # creating custom tkinter window
app.geometry("600x440")
app.title('ENHANCED DRIVER SAFETY SYSTEM')

img1 = ImageTk.PhotoImage(Image.open("pattern.png"))
l1 = customtkinter.CTkLabel(master=app, image=img1)
l1.pack()

# creating custom frame
frame = customtkinter.CTkFrame(master=l1, width=320, height=360, corner_radius=15)
frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

l2=customtkinter.CTkLabel(master=frame, text="Drowsiness Detection",font=('Century Gothic',25,))
l2.place(x=43, y=45,)

# Create custom button for drowsiness detection
button1 = customtkinter.CTkButton(master=frame, height=80, width=220, text="Start",font=("",20), command=start_drowsiness_detection, corner_radius=6)
button1.place(x=50, y=110)

# Create custom button to stop drowsiness detection
button2 = customtkinter.CTkButton(master=frame, height=80, width=220, text="Stop", font=("",20),command=stop_drowsiness_detection, corner_radius=6)
button2.place(x=50, y=220)

app.mainloop()
