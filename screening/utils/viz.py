import time, cv2
class FPSMeter:
    def __init__(self): self.t=time.time(); self.n=0; self.fps=0.0
    def tick(self):
        self.n+=1; now=time.time()
        if now-self.t>=1.0: self.fps=self.n/(now-self.t); self.n=0; self.t=now
        return self.fps
def draw_hud(img,text,y=24): cv2.putText(img,text,(12,y),0,0.6,(0,255,255),2); return img
