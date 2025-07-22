import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import threading
import queue

class BlackOverlay:
    def __init__(self, command_queue):
        self.command_queue = command_queue
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg='black')
        self.root.attributes("-topmost", True)
        self.root.withdraw()
        self.visible = False
        self.root.after(100, self.check_queue)

    def show(self):
        if not self.visible:
            self.visible = True
            self.root.deiconify()

    def hide(self):
        if self.visible:
            self.visible = False
            self.root.withdraw()

    def check_queue(self):
        while not self.command_queue.empty():
            cmd = self.command_queue.get()
            if cmd == "show":
                self.show()
            elif cmd == "hide":
                self.hide()
        self.root.after(100, self.check_queue)

    def run(self):
        self.root.mainloop()

def eye_tracker(cmd_queue):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    def is_looking_away(landmarks, img_w):
        left_eye = [landmarks[33], landmarks[133]]
        right_eye = [landmarks[362], landmarks[263]]
        eye_x = np.mean([p.x for p in left_eye + right_eye]) * img_w
        screen_center = img_w / 2
        return abs(eye_x - screen_center) > img_w * 0.15

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            if is_looking_away(landmarks, w):
                cmd_queue.put("show")
            else:
                cmd_queue.put("hide")
        else:
            cmd_queue.put("show")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    command_queue = queue.Queue()
    threading.Thread(target=eye_tracker, args=(command_queue,), daemon=True).start()
    app = BlackOverlay(command_queue)
    app.run()
