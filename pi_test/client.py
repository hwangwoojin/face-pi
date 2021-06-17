import tkinter as tk
import tkinter.font
import tkinter.messagebox
import tkinter.scrolledtext
import threading
import numpy as np
import cv2
import asyncio
import websockets
import json
import base64

from PIL import Image, ImageTk

uri = 'ws://localhost:3000'


class Client(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        # URI
        self.uri = uri
        # Cascade Model
        self.cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        # OpenCV
        # self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # for webcam?
        self.cap = cv2.VideoCapture(0) # for camera module
        self.cam_width = 800
        self.cam_height = 800
        self.cap.set(3, self.cam_width)
        self.cap.set(4, self.cam_height)
        # Application Function
        # detecting square
        self.detecting_square = (500, 500)
        self.rectangle_color = (0, 0, 255)
        # tkinter GUI
        self.width = 800
        self.height = 900
        self.parent = parent
        self.parent.title("Facial-Payment")
        self.parent.geometry("%dx%d+100+100" % (self.width, self.height))
        self.pack()
        self.create_widgets()
        # Event loop and Thread
        self.event_loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.mainthread)
        self.thread.start()


    def create_widgets(self):
        image = np.zeros([self.cam_height, self.cam_width, 3], dtype=np.uint8)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        font = tk.font.Font(family="courier", size=15)
        # alert, label, log
        self.alert = tk.Label(self, text="", font=font)
        self.alert.grid(row=0, column=0, columnspan=20)
        self.label = tk.Label(self, image=image)
        self.label.grid(row=1, column=0, columnspan=20)
        self.log = tk.scrolledtext.ScrolledText(self, wrap = tk.WORD, state=tk.DISABLED, width = 96, height = 10)
        self.log.grid(row=2, column=0, columnspan=20)
        # quit
        self.quit = tk.Button(self, text="Exit", fg="red", command=self.stop)
        self.quit.grid(row=3, column=10)


    def logging(self, text):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tkinter.END, text)
        self.log.insert(tkinter.END, '\n')
        self.log.config(state=tk.DISABLED)


    def detect_face(self, frame):
        # detect face with cascade
        faces = self.cascade.detectMultiScale(
            frame,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(150, 150)
        )
        for (x, y, w, h) in faces:
            image = frame[y:y+h, x:x+w]
            return image, True
        return [], False


    def mainthread(self):
        t = threading.currentThread()
        asyncio.set_event_loop(self.event_loop)
        x1 = int(self.cam_width / 2 - self.detecting_square[0] / 2)
        x2 = int(self.cam_width / 2 + self.detecting_square[0] / 2)
        y1 = int(self.cam_height / 2 - self.detecting_square[1] / 2)
        y2 = int(self.cam_height / 2 + self.detecting_square[1] / 2)
        while getattr(t, "do_run", True):
            ret, frame = self.cap.read()
            # BGR to RGB
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # detect face
            image, detected = self.detect_face(converted[y1:y2, x1:x2])
            if detected:
                self.event_loop.run_until_complete(self.send_face(image))
            # make rectangle frame
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), self.rectangle_color, 3)
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            converted = cv2.flip(converted,1)
            image = Image.fromarray(converted)
            image = ImageTk.PhotoImage(image)
            self.label.configure(image=image)
            self.label.image = image # kind of double buffering


    @asyncio.coroutine
    def set_rectangle(self):
        self.rectangle_color = (255, 0, 0)
        yield from asyncio.sleep(2)
        self.rectangle_color = (0, 0, 255)


    async def send_face(self, image):
        try:
            async with websockets.connect(uri) as websocket:
                img = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
                send = json.dumps({'action': 'veryfy', 'image': img})
                await websocket.send(send)
                recv = await websocket.recv()
                data = json.loads(recv)
                # success test
                if data['status'] == 'success':
                    self.logging('success')
                    asyncio.ensure_future(self.set_rectangle())
                # fail test
                elif data['status'] == 'fail':
                    self.logging('fail')
                    asyncio.ensure_future(self.set_rectangle())
        except Exception as e:
            self.logging(e)


    def stop(self):
        self.thread.do_run = False
        # self.thread.join() # there is a freeze problem
        self.event_loop.close()
        self.cap.release()
        self.parent.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    Client(root)
    root.mainloop()