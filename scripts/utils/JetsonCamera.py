# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import time
try:
    from  Queue import  Queue
except ModuleNotFoundError:
    from  queue import  Queue

import  threading
import signal
import sys



#GST_ARGUS: 4656 x 3496 FR = 10.000000 fps Duration = 100000000 ; Analog Gain range min 1.000000, max 16.000000; Exposure Range min 13000, max 683709000;

#GST_ARGUS: 3840 x 2160 FR = 21.000000 fps Duration = 47619048 ; Analog Gain range min 1.000000, max 16.000000; Exposure Range min 13000, max 683709000;

#GST_ARGUS: 1920 x 1080 FR = 59.999999 fps Duration = 16666667 ; Analog Gain range min 1.000000, max 16.000000; Exposure Range min 13000, max 683709000;

#GST_ARGUS: 1280 x 720 FR = 120.000005 fps Duration = 8333333 ; Analog Gain range min 1.000000, max 16.000000; Exposure Range min 13000, max 683709000;

global mode

mode = 3

WIDTH=[4656,3840,1920,1280]
HEIGHT=[3496,2160,1080,720]
DISP_WIDTH=[1164,960,860,640]
DISP_HEIGHT=[874,540,540,360]
FRAMERATE=[10,20,60,120]
def gstreamer_pipeline(
   
    capture_width=WIDTH[mode], #4656,#3840,#1920,#1280
    capture_height=HEIGHT[mode], #3496,#2160,#1080,#720
    display_width=DISP_WIDTH[mode],
    display_height=DISP_HEIGHT[mode], 
    framerate=FRAMERATE[mode], #60,#20,#10
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class FrameReader(threading.Thread):
    queues = []
    _running = True
    camera = None
    def __init__(self, camera, name):
        threading.Thread.__init__(self)
        self.name = name
        self.camera = camera
 
    def run(self):
        while self._running:
            _, frame = self.camera.read()
            while self.queues:
                queue = self.queues.pop()
                queue.put(frame)
    
    def addQueue(self, queue):
        self.queues.append(queue)

    def getFrame(self, timeout = None):
        queue = Queue(1)
        self.addQueue(queue)
        return queue.get(timeout = timeout)

    def stop(self):
        self._running = False

class Previewer(threading.Thread):
    window_name = "Visualizador Camara IMX519 Enfoque/Modo Camara"
    _running = True
    camera = None
    def __init__(self, camera, name):
        threading.Thread.__init__(self)
        self.name = name
        self.camera = camera
    
    def run(self):
        self._running = True
        while self._running:
            cv2.imshow(self.window_name, self.camera.getFrame(2000))
            keyCode = cv2.waitKey(16) & 0xFF
        cv2.destroyWindow(self.window_name)

    def start_preview(self):
        self.start()
    def stop_preview(self):
        self._running = False

class Camera(object):
    frame_reader = None
    cap = None
    previewer = None

    def __init__(self, width=DISP_WIDTH[mode], height=DISP_HEIGHT[mode]):
        self.open_camera(width, height)

    def open_camera(self, width=DISP_WIDTH[mode], height=DISP_HEIGHT[mode]):
        self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, display_width=width, display_height=height), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera!")
        if self.frame_reader == None:
            self.frame_reader = FrameReader(self.cap, "")
            self.frame_reader.daemon = True
            self.frame_reader.start()
        self.previewer = Previewer(self.frame_reader, "")

    def getFrame(self, timeout = None):
        return self.frame_reader.getFrame(timeout)

    def start_preview(self):
        self.previewer.daemon = True
        self.previewer.start_preview()

    def stop_preview(self):
        self.previewer.stop_preview()
        self.previewer.join()
    
    def close(self):
        self.frame_reader.stop()
        self.cap.release()

if __name__ == "__main__":
    
    camera = Camera()
    camera.start_preview()
    time.sleep(10)
    camera.stop_preview()
    camera.close()
