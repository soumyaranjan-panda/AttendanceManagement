import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import time

class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.geometry("800x600")
        self.root.title("Attendance Management System")
        
        # Initialize variables for user input
        self.name_var = tk.StringVar()
        self.regd_var = tk.StringVar()
        
        # Load face detection model
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Define file paths
        self.dataset_path = "dataset"
        self.attendance_file = "attendance.csv"
        self.registered_file = "registered_users.csv"
        
        # Create GUI components
        self.create_widgets()
        
        # Check and prepare necessary files and directories
        self.check_files()

    def create_widgets(self):
        # Input fields for user details
        ttk.Label(self.root, text="Name:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(self.root, textvariable=self.name_var).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.root, text="Registration No:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(self.root, textvariable=self.regd_var).grid(row=1, column=1, padx=5, pady=5)
        
        # Buttons for actions
        ttk.Button(self.root, text="Add User", command=self.add_user).grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(self.root, text="Mark Attendance", command=self.mark_attendance).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Video panel for live camera feed
        self.video_label = ttk.Label(self.root)
        self.video_label.grid(row=4, column=0, columnspan=2, pady=10)


    def check_files(self):
        # Create necessary directories and files
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            
        if not os.path.exists(self.registered_file):
            pd.DataFrame(columns=["Name", "RegdNo", "ID"]).to_csv(self.registered_file, index=False)
            
        if not os.path.exists(self.attendance_file):
            pd.DataFrame(columns=["ID", "Name", "RegdNo", "Timestamp"]).to_csv(self.attendance_file, index=False)

    def add_user(self):
        # Capture user images for training
        name = self.name_var.get()
        regd = self.regd_var.get()
        
        if not name or not regd:
            messagebox.showerror("Error", "Please fill all fields")
            return
            
        user_id = len(os.listdir(self.dataset_path)) + 1
        cam = cv2.VideoCapture(0)
        cam.set(3, 320)  # Width
        cam.set(4, 240)  # Height
        
        count = 0
        while count < 30:  # Capture 30 samples
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                count += 1
                cv2.imwrite(f"{self.dataset_path}/User.{user_id}.{count}.jpg", gray[y:y+h,x:x+w])
                
            cv2.imshow('Registering User', img)
            if cv2.waitKey(1) == 27:
                break
                
        cam.release()
        cv2.destroyAllWindows()
        self.train_model()
        
        # Save user details
        pd.DataFrame([[name, regd, user_id]], columns=["Name", "RegdNo", "ID"]).to_csv(self.registered_file, mode='a', header=False, index=False)
        
    def train_model(self):
        # Train face recognizer
        faces = []
        ids = []
        
        for image_path in os.listdir(self.dataset_path):
            if image_path.startswith("User."):
                img = cv2.imread(os.path.join(self.dataset_path, image_path), cv2.IMREAD_GRAYSCALE)
                user_id = int(image_path.split(".")[1])
                faces.append(img)
                ids.append(user_id)
                
        self.recognizer.train(faces, np.array(ids))
        self.recognizer.save("trainer.yml")
        

    def mark_attendance(self):
        # Real-time attendance marking
        if not os.path.exists("trainer.yml"):
            messagebox.showerror("Error", "Model not trained yet. please add users first.")
        else: 
            self.train_model()
        
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        def update_frame():
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

                for (x,y,w,h) in faces:
                    id_, confidence = self.recognizer.predict(gray[y:y+h,x:x+w])
                    if confidence < 70:
                        user = pd.read_csv(self.registered_file)
                        user = user[user["ID"] == id_]
                        if not user.empty:
                            # Get user details
                            name = user["Name"].values[0]
                            regd_no = user["RegdNo"].values[0]
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Log attendance
                            data = [[id_, name, regd_no, timestamp]]
                            pd.DataFrame(data).to_csv(self.attendance_file, mode='a', header=False, index=False)
                            
                            # Show attendance confirmation
                            messagebox.showinfo("Attendance Marked", f"Name: {name}\nRegd No: {regd_no}\nDate & Time: {timestamp}")
                            
                            # Ask if the user wants to mark another attendance
                            if messagebox.askyesno("Continue", "Do you want to mark attendance for another person?"):
                                break  # Exit the current loop to continue updating frames
                            else:
                                cap.release()
                                cv2.destroyAllWindows()
                                self.root.quit()
                                return

                frame = cv2.resize(frame, (640, 480))
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                self.video_label.after(10, update_frame)

        update_frame()


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.mainloop()
