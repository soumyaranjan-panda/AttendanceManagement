import calendar
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import time
from datetime import datetime


class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.geometry("800x600")
        self.root.title("Attendance Management System")

        # Initialize variables for user input
        self.name_var = tk.StringVar()
        self.regd_var = tk.StringVar()
        self.password_var = tk.StringVar()

        # Load face detection model
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
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
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root)

        # Create tabs for different functionalities
        self.add_user_tab = ttk.Frame(self.notebook)
        self.mark_attendance_tab = ttk.Frame(self.notebook)
        self.calendar_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.add_user_tab, text="Add User")
        self.notebook.add(self.mark_attendance_tab, text="Mark Attendance")
        self.notebook.add(self.calendar_tab, text="Attendance Calendar")
        self.notebook.pack(expand=True, fill="both")

        # Add User Tab Widgets
        ttk.Label(self.add_user_tab, text="Name:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(self.add_user_tab, textvariable=self.name_var).grid(
            row=0, column=1, padx=5, pady=5
        )

        ttk.Label(self.add_user_tab, text="Registration No:").grid(
            row=1, column=0, padx=5, pady=5
        )
        ttk.Entry(self.add_user_tab, textvariable=self.regd_var).grid(
            row=1, column=1, padx=5, pady=5
        )

        ttk.Label(self.add_user_tab, text="Password:").grid(
            row=2, column=0, padx=5, pady=5
        )
        ttk.Entry(self.add_user_tab, textvariable=self.password_var, show="*").grid(
            row=2, column=1, padx=5, pady=5
        )

        ttk.Button(
            self.add_user_tab, text="Add User", command=self.authenticate_and_add_user
        ).grid(row=3, column=0, columnspan=2, pady=10)

        # Mark Attendance Tab Widgets
        self.video_label = ttk.Label(self.mark_attendance_tab)
        self.video_label.grid(row=0, column=0)

        ttk.Button(
            self.mark_attendance_tab,
            text="Start Attendance",
            command=self.mark_attendance,
        ).grid(row=1, column=0)

        ttk.Button(
            self.mark_attendance_tab, text="Exit", command=self.exit_attendance
        ).grid(row=1, column=1)

        # Calendar Tab Widgets
        ttk.Button(
            self.calendar_tab,
            text="Show Attendance Calendar",
            command=self.show_attendance_calendar,
        ).pack(pady=20)

    def check_files(self):
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        if not os.path.exists(self.registered_file):
            pd.DataFrame(columns=["Name", "RegdNo", "ID"]).to_csv(
                self.registered_file, index=False
            )

        if not os.path.exists(self.attendance_file):
            pd.DataFrame(columns=["ID", "Name", "RegdNo", "Timestamp"]).to_csv(
                self.attendance_file, index=False
            )

    def authenticate_and_add_user(self):
        password = self.password_var.get()

        if password != "soumya":  # Replace with actual password
            messagebox.showerror("Error", "Authentication failed. Incorrect password.")
            return

        name = self.name_var.get()
        regd = self.regd_var.get()

        if not name or not regd:
            messagebox.showerror("Error", "Please fill all fields")
            return

        user_id = len(os.listdir(self.dataset_path)) + 1
        cam = cv2.VideoCapture(0)

        count = 0

        while count < 30:  # Capture 30 samples
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1

                cv2.imwrite(
                    f"{self.dataset_path}/User.{user_id}.{count}.jpg",
                    gray[y : y + h, x : x + w],
                )

            cv2.imshow("Registering User", img)

            if cv2.waitKey(1) == 27:  # Escape key to stop capturing
                break

        cam.release()

        cv2.destroyAllWindows()

        # Train model after adding user images
        self.train_model()

        # Save user details
        pd.DataFrame([[name, regd, user_id]], columns=["Name", "RegdNo", "ID"]).to_csv(
            self.registered_file,
            mode="a",
            header=False,
            index=False,
        )

    def train_model(self):
        faces = []
        ids = []

        for image_path in os.listdir(self.dataset_path):
            if image_path.startswith("User."):
                img = cv2.imread(
                    os.path.join(self.dataset_path, image_path), cv2.IMREAD_GRAYSCALE
                )
                user_id = int(image_path.split(".")[1])
                faces.append(img)
                ids.append(user_id)

        if len(faces) > 0 and len(ids) > 0:
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.save("trainer.yml")

    def mark_attendance(self):
        if not os.path.exists("trainer.yml"):
            messagebox.showerror(
                "Error", "Model not trained yet. Please add users first."
            )
            return
        else:
            self.train_model()

        cap = cv2.VideoCapture(0)

        def update_frame():
            ret, frame = cap.read()

            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    id_, confidence = self.recognizer.predict(
                        gray[y : y + h, x : x + w]
                    )

                    if confidence < 70:
                        user = pd.read_csv(self.registered_file)
                        user = user[user["ID"] == id_]

                        if not user.empty:
                            name = user["Name"].values[0]
                            regd_no = user["RegdNo"].values[0]
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                            data = [[id_, name, regd_no, timestamp]]
                            pd.DataFrame(data).to_csv(
                                self.attendance_file,
                                mode="a",
                                header=False,
                                index=False,
                            )

                            messagebox.showinfo(
                                "Attendance Marked",
                                f"Name: {name}\nRegd No: {regd_no}\nDate & Time: {timestamp}",
                            )

                            if not messagebox.askyesno(
                                "Continue",
                                "Do you want to mark attendance for another person?",
                            ):
                                self.exit_attendance()

                frame_resized = cv2.resize(frame, (640, 480))
                cv_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGBA)
                img_pil = Image.fromarray(cv_image)
                imgtk = ImageTk.PhotoImage(image=img_pil)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                self.video_label.after(10, update_frame)
            else:
                cap.release()

        update_frame()

    def exit_attendance(self):
        cv2.destroyAllWindows()
        self.root.quit()

    def show_attendance_calendar(self):
        cal_window = tk.Toplevel(self.root)
        cal_window.title("Attendance Calendar")
        cal_window.geometry("800x600")

        user_list_frame = ttk.Frame(cal_window)
        user_list_frame.pack(side=tk.LEFT)

        calendar_frame = ttk.Frame(cal_window)
        calendar_frame.pack(side=tk.RIGHT)

        users_df = pd.read_csv(self.registered_file)
        user_listbox = tk.Listbox(user_list_frame)

        for _, row in users_df.iterrows():
            user_listbox.insert(tk.END, f"{row['Name']} ({row['RegdNo']})")

        user_listbox.pack()
        selected_user_id_var = tk.StringVar()

        # Initialize variables for calendar display
        self.year_month_frame = None  # Use self to maintain scope
        self.month_days = []  # List to hold day labels
        self.day_date_str = tk.StringVar()  # StringVar for day display
        self.day_color = tk.StringVar()  # StringVar for day color

        def show_calendar():
            selected_index = user_listbox.curselection()
            if not selected_index:
                return

            selected_user_id_var.set(users_df.iloc[selected_index[0]]["ID"])

            # Destroy previous frame if it exists
            if self.year_month_frame is not None:
                self.year_month_frame.destroy()

            # Create a new frame for year and month display
            self.year_month_frame = ttk.Frame(calendar_frame)
            self.year_month_frame.pack(pady=(10))

            # Clear previous month days
            for label in self.month_days:
                label.destroy()
            self.month_days.clear()

            # Create labels for days of the month
            for week_num in range(6):  # Up to 6 weeks in a month
                for day_num in range(7):  # Days of the week (Mon-Sun)
                    day_label = ttk.Label(
                        self.year_month_frame, text="", background="white"
                    )
                    day_label.grid(row=(week_num + 1), column=(day_num))
                    self.month_days.append(day_label)

            current_year = datetime.now().year
            current_month = datetime.now().month

            # Populate the calendar with days and colors based on attendance
            for day in range(1, calendar.monthrange(current_year, current_month)[1] + 1):
                current_date = datetime(current_year, current_month, day)
                attendance_data = pd.read_csv(self.attendance_file)
                present_dates = [
                    datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").date()
                    for ts in attendance_data["Timestamp"]
                ]

                color = (
                    "green"
                    if current_date.date() in present_dates
                    else "red"
                    if current_date.date() < datetime.today().date()
                    else "white"
                )
                day_label = self.month_days[day - 1]  # Adjust index for day label
                day_label.config(text=str(day), background=color)

        show_calendar_button = tk.Button(
            user_list_frame, text="Show Calendar", command=show_calendar
        )
        show_calendar_button.pack()



if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.mainloop()
