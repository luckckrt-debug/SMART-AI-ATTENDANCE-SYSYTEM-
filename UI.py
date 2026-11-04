import tkinter as tk
from tkinter import messagebox as mess
from tkinter.ttk import Treeview, Scrollbar, Progressbar
from PIL import Image, ImageTk
from threading import Thread
import csv
import datetime
import time
from pathlib import Path

from Generate_Dataset import Generate_Data
from Model_train import Model_Training
from Recognizer import Recognition

BASE_DIR = Path(__file__).resolve().parent


def launch_ui():
    window = tk.Tk()

    class FullScreenApp(object):
        def __init__(self, master):
            self.master = master
            pad = 3
            self._geom = '200x200+0+0'
            master.geometry("{0}x{1}+0+0".format(
                master.winfo_screenwidth() - pad, master.winfo_screenheight() - pad))
            master.bind('<Escape>', self.toggle_geom)

        def toggle_geom(self, event):
            geom = self.master.winfo_geometry()
            self.master.geometry(self._geom)
            self._geom = geom

    app = FullScreenApp(window)

    class Example(tk.Frame):
        def __init__(self, master, *pargs):
            super().__init__(master, *pargs)
            img_path = BASE_DIR / "landscape.png"
            if img_path.exists():
                self.image = Image.open(str(img_path))
                self.img_copy = self.image.copy()
                self.background_image = ImageTk.PhotoImage(self.image)
                self.background = tk.Label(self, image=self.background_image)
                self.background.pack(fill=tk.BOTH, expand=tk.YES)
                self.background.bind('<Configure>', self._resize_image)

        def _resize_image(self, event):
            if hasattr(self, 'img_copy'):
                new_width = max(1, event.width)
                new_height = max(1, event.height)
                resized = self.img_copy.resize((new_width, new_height))
                self.background_image = ImageTk.PhotoImage(resized)
                self.background.configure(image=self.background_image)

    e = Example(window)
    e.pack(fill=tk.BOTH, expand=tk.YES)

    ts = time.time()
    date_str = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    day, month, year = date_str.split("-")
    mont = {
        '01': 'January', '02': 'February', '03': 'March', '04': 'April',
        '05': 'May', '06': 'June', '07': 'July', '08': 'August',
        '09': 'September', '10': 'October', '11': 'November', '12': 'December'
    }

    window.title("Smart Attendance System Using Facial Recognition")

    message3 = tk.Label(window, text="Smart Attendance System", fg="white", bg="#8a2e7f", width=55, height=1, font=('Helvetica', 30, 'bold'))
    message3.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

    frame4 = tk.Frame(window, bg="#8a2e7f")
    frame4.place(relx=0.5, rely=0.17, relwidth=0.2, relheight=0.05, anchor=tk.CENTER)

    datef = tk.Label(frame4, text=day + "-" + mont.get(month, month) + "-" + year, fg="white", bg="#262523", width=55, height=1, font=('Helvetica', 20, 'bold'))
    datef.pack(fill='both', expand=1)

    frame1 = tk.Frame(window, bg="#ffffff")
    frame1.place(relx=0.057, rely=0.21, relwidth=0.35, relheight=0.78)

    head1 = tk.Label(frame1, text="                      For Already Registered                       ", fg="white", bg="#8a2e7f", font=('Helvetica', 17, 'bold'))
    head1.place(x=0, y=0)

    frame2 = tk.Frame(window, bg="#ffffff")
    frame2.place(relx=0.59, rely=0.21, relwidth=0.35, relheight=0.78)

    head2 = tk.Label(frame2, text="                       For New Enrollment                       ", fg="white", bg="#8a2e7f", font=('Helvetica', 17, 'bold'))
    head2.grid(row=0, column=0)

    lbl = tk.Label(frame2, text="Enter Roll Number", width=20, height=1, fg="white", bg="#8a2e7f", font=('Helvetica', 13, 'bold'))
    lbl.place(relx=0.3, rely=0.1)

    txt = tk.Entry(frame2, width=16, fg="black", font=('Helvetica', 13))
    txt.place(relx=0.32, rely=0.16)

    lbl2 = tk.Label(frame2, text="Enter Student's Name", width=20, height=1, fg="white", bg="#8a2e7f", font=('Helvetica', 13, 'bold'))
    lbl2.place(relx=0.3, rely=0.27)

    txt2 = tk.Entry(frame2, width=20, fg="black", font=('Helvetica', 13))
    txt2.place(relx=0.3, rely=0.33)

    def get_data():
        name = txt2.get().strip()
        roll = txt.get().strip()
        if not name or not roll:
            mess.showwarning("Incomplete Data", "Please enter student name and roll number.")
            return
        success, msg, count = Generate_Data(name, roll)
        if success:
            mess.showinfo("Success", msg)
        else:
            mess.showerror("Error", msg)

    takeImg = tk.Button(frame2, text="Take Images", command=get_data, fg="white", bg="#8a2e7f", width=25, height=1, activebackground="white", font=('Helvetica', 13, 'bold'))
    takeImg.place(x=100, y=230)

    def progress_measure():
        progress['value'] = 20
        window.update_idletasks()
        time.sleep(1)
        progress['value'] = 50
        window.update_idletasks()
        time.sleep(1)
        progress['value'] = 100
        window.update_idletasks()

    def bar():
        Thread(target=Model_Training, kwargs={'epochs': 40}, daemon=True).start()
        Thread(target=progress_measure, daemon=True).start()

    trainImg = tk.Button(frame2, text="Train The Model", fg="white", command=bar, bg="#507d2a", width=34, height=1, activebackground="white", font=('Helvetica', 15, 'bold'))
    trainImg.place(x=37, y=350)

    progress = Progressbar(frame2, length=200, orient=tk.HORIZONTAL, mode='determinate')
    progress.place(x=150, y=435)

    lbl3 = tk.Label(frame1, text="Select Class", width=20, fg="white", bg="#8a2e7f", height=1, font=('Helvetica', 13, 'bold'))
    lbl3.place(x=140, y=50)

    v = tk.IntVar()
    v.set(1)

    r1 = tk.Radiobutton(frame1, text="Hindi", variable=v, value=1, bg="#ffffff", activeforeground="#507d2a", borderwidth=10, font=('Helvetica', 14, 'bold'))
    r1.place(x=130, y=110, anchor=tk.CENTER)

    r2 = tk.Radiobutton(frame1, text="English", variable=v, value=2, bg="#ffffff", activeforeground="#507d2a", borderwidth=10, font=('Helvetica', 14, 'bold'))
    r2.place(x=260, y=88)

    def run_recognition():
        subj = "Hindi" if v.get() == 1 else "English"
        Recognition(subj)

    trackImg = tk.Button(frame1, text="Take Attendance", fg="white", bg="#507d2a", command=run_recognition, width=35, height=1, activebackground="white", font=('Helvetica', 15, 'bold'))
    trackImg.place(x=35, y=150)

    tv = Treeview(frame1, columns=('roll_no', 'name', 'attendance'), selectmode='browse', show=["headings"])
    tv.column('roll_no', width=155, anchor=tk.CENTER)
    tv.column('name', width=155, anchor=tk.CENTER)
    tv.column('attendance', width=155, anchor=tk.CENTER)
    tv.grid(row=2, column=0, padx=(0, 0), pady=(250, 0), columnspan=4)

    tv.heading('roll_no', text='Roll Number')
    tv.heading('name', text='Name')
    tv.heading('attendance', text='Attendance')

    scroll = Scrollbar(frame1, orient='vertical', command=tv.yview)
    scroll.grid(row=2, column=4, padx=(0, 100), pady=(250, 0), sticky='ns')
    tv.configure(yscrollcommand=scroll.set)

    def clear_tree():
        for child in tv.get_children():
            tv.delete(child)

    def show_attendance():
        clear_tree()
        subj_name = "Hindi" if v.get() == 1 else "English"
        csv_file = BASE_DIR / f"{subj_name}_attendance" / f"{subj_name}_Attendance.csv"
        if not csv_file.exists():
            mess.showwarning("Not Found", f"No attendance CSV found for {subj_name}.")
            return

        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                roll = row.get('Roll Number', '')
                name = row.get('Name', '')
                att = row.get('Attendance', '0')
                tv.insert("", 0, values=(roll, name, att))

    Show_button = tk.Button(frame1, text="Show Attendance", fg="white", command=show_attendance, bg="green", width=15, height=1, activebackground="white", font=('Helvetica', 12, 'bold'))
    Show_button.place(x=35, y=200)

    Show_button1 = tk.Button(frame1, text="Clear", fg="white", command=clear_tree, bg="red", width=15, height=1, activebackground="white", font=('Helvetica', 12, 'bold'))
    Show_button1.place(x=285, y=200)

    quitWindow = tk.Button(frame1, text="Quit", fg="white", bg="red", width=35, height=1, command=window.destroy, activebackground="white", font=('Helvetica', 15, 'bold'))
    quitWindow.place(x=30, y=490)

    window.mainloop()


if __name__ == "__main__":
    launch_ui()
