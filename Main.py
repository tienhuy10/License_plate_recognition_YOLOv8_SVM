import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
from tkinter import *
from tkinter import messagebox, ttk
from PIL import ImageTk, Image
import sqlite3

# GUI
root = Tk()
root.geometry('1380x750')
root.resizable(width=False, height=False)
root.title("Hệ Thống Nhận Diện Biển Số Xe")
root.configure(bg='white')

tentruong = Label(root, text="TRƯỜNG ĐẠI HỌC VINH", bg='white', font=('Time 25 bold'), fg='blue')
tentruong.pack(side=TOP)

a = Label(root, text="      ", bg='white', font=('Time 5 bold'))
a.pack(side=TOP)

title = Label(root, text="Hệ Thống Nhận Diện Biển Số Xe", bg='white', font=('Time 22 bold'))
title.pack(side=TOP)

canvas_cam = Canvas(root, width=800, height=500, bg="white")
canvas_cam.place(x=50, y=150)

canvas_plate = Canvas(root, width=20, height=15, bg="blue")
canvas_plate.place(x=900, y=190)
label_plate = Label(canvas_plate, bg="blue")
label_plate.pack(expand=True, fill=BOTH)

title_img = Label(root, text="Hình ảnh biển số xe", bg='white', font=('Time 15 bold'))
title_img.place(x = 900, y = 150)

# title_list = Label(root, text="Danh sách biển số", bg='white', font=('Time 15 bold'))
# title_list.place(x = 1200, y = 150)

#Model
model = YOLO('yolov8s.pt')
np_model = YOLO('best.pt')

cap = cv2.VideoCapture('video.mov')

# Cấu hình mô hình SVM
digit_w = 30
digit_h = 60
model_svm = cv2.ml.SVM_load('svm.xml')
plate_info = ""
list_plate = []
text_userif = None

with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

count = -1
tracker = Tracker()
cy1 = 192
offset = 6
ret = True
counter = []

tree = ttk.Treeview(root, columns=("ID", "Họ và tên", "SĐT", "CMND", "Biển số xe"), show="headings")
tree.heading("ID", text="ID")
tree.heading("Họ và tên", text="Họ và tên")
tree.heading("SĐT", text="SĐT")
tree.heading("CMND", text="CMND")
tree.heading("Biển số xe", text="Biển số xe")
tree.column("ID", width=50)
tree.column("Họ và tên", width=100)
tree.column("SĐT", width=100)
tree.column("CMND", width=100)
tree.column("Biển số xe", width=100)
tree.place(x=900, y=400)

tree_height = 6 
tree_height_pixels = tree_height * tree.winfo_reqheight()
tree.configure(height=tree_height)

# Biến lưu trữ biển số xe đã xử lý
processed_plates = set()

def find_user_by_license_plate(license_plate):
    conn = sqlite3.connect('userDB.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user WHERE license_plate=?", (license_plate,))
    user_info = cursor.fetchone()
    conn.close()
    return user_info

def update_label(plate_info):
    text_plate.config(text=plate_info)
    root.update()

def process_license_plate(roi, id):
    license_plates = np_model(roi)[0]
    plate_info = ""

    for license_plate in license_plates.boxes.data.tolist():
        plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate
        plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
        plate = cv2.resize(plate, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('a.jpg', plate)

        plate_pil = Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        plate_tk = ImageTk.PhotoImage(image=plate_pil)
        label_plate.configure(image=plate_tk)
        label_plate.image = plate_tk

        plate = cv2.GaussianBlur(plate, (5, 5), 0)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        
        _, plate_threshold = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imshow('imga', plate_threshold)

        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(plate_threshold, cv2.MORPH_CLOSE, kernel3)
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cont = sorted(cont, key=lambda x: cv2.boundingRect(x)[1])

        rows = []
        current_row = []
        previous_y = None

        for c in cont:
            (x, y, w, h) = cv2.boundingRect(c)
            if previous_y is None or abs(y - previous_y) < 10:
                current_row.append(c)
            else:
                rows.append(sorted(current_row, key=lambda x: cv2.boundingRect(x)[0]))
                current_row = [c]
            previous_y = y

        if current_row:
            rows.append(sorted(current_row, key=lambda x: cv2.boundingRect(x)[0]))

        for row in rows:
            for c in row:
                (x, y, w, h) = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                if 0.2 < aspect_ratio < 1.0 and h > 20:
                    cv2.rectangle(plate, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.imshow('aaa', plate)
                    # cv2.waitKey(1)

                    curr_num = thre_mor[y:y + h, x:x + w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num, dtype=np.float32)
                    curr_num = curr_num.reshape(-1, digit_w * digit_h)

                    result = model_svm.predict(curr_num)[1]
                    result = int(result[0, 0])

                    if result <= 9:
                        result = str(result)
                    else:
                        result = chr(result)
                    plate_info += result
                    update_label(plate_info)

            user_info = find_user_by_license_plate(plate_info)

            text_userif = Label(root, text="Họ Tên", bg='white', font=('Time 15 bold'))
            text_userif.place(x=1200, y=250)
            text_userif = Label(root, text="ㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤ", fg="blue", bg='white', font=('Time 15 bold'))
            text_userif.place(x=1200, y=285)
            if user_info:
                text_userif = Label(root, text=user_info[1], fg="blue", bg='white', font=('Time 15 bold'))
                text_userif.place(x=1200, y=285)
            if user_info and id not in processed_plates:  
                # Hiển thị dữ liệu trong Treeview
                tree.insert("", 0, values=(user_info[0], user_info[1], user_info[2], user_info[4], user_info[5]))
                processed_plates.add(id)
            else:
                print("Không tìm thấy thông tin người dùng cho biển số xe:", plate_info)

text_plate = Label(root, text=plate_info, fg="blue", bg='white', font=('Time 15 bold'))
text_plate.place(x=1200, y=185)

title_plate = Label(root, text="Biển số xe", bg='white', font=('Time 15 bold'))
title_plate.place(x=1200, y=150)


# Main loop
while True:
    ret, frame = cap.read()
    count += 1
    frame = cv2.resize(frame, (800, 500))

    if ret:
        results = model.predict(frame)
        a = results[0].boxes.data
        a_cpu = a.cpu()
        px = pd.DataFrame(a_cpu.numpy()).astype("float")

        bbox_list = []
        for index, row in px.iterrows():
            x1, y1, x2, y2, _, d = map(int, row)
            c = class_list[d]
            if 'car' in c:
                bbox_list.append([x1, y1, x2, y2])

        bbox_id = tracker.update(bbox_list)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int(2 * (x3 + x4)) // 5
            cy = int(y3 + y4) // 2
            if cy1 - offset < cx < cy1 + offset:
                roi = frame[y3:y4, x3:x4]
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
                cv2.circle(frame, (cx, cy), 3, (255, 0, 255), -3)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                plate_info = ""
                process_license_plate(roi, id)
                if counter.count(id) == 0:
                    counter.append(id)

        sum = len(counter)
        tong_xe = Label(root, text=sum, height=2, fg="red", bg='white', font=('Time 15 bold'))
        tong_xe.place(x=900, y=610)
        title_tong = Label(root, text="Tổng số xe ô tô", bg='white', font=('Time 15 bold'))
        title_tong.place(x=900, y=580)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image1 = ImageTk.PhotoImage(image=Image.fromarray(frame))
        canvas_cam.create_image(0, 0, image=image1, anchor=NW)
        root.update()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
