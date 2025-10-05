import cv2
class OpenCVCamera:
    def __init__(self, source=0, width=1280, height=720, fps=60):
        self.cap=cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,width); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height); self.cap.set(cv2.CAP_PROP_FPS,fps)
    def read(self): ok,frame=self.cap.read(); return ok,frame
    def release(self): self.cap.release()
