import argparse, cv2, torch, json
from model import build_model
from utils.camera_stream import OpenCVCamera
from utils.actuator import Actuator
from utils.viz import FPSMeter, draw_hud
def load_labels(p): import json; return json.load(open(p))['classes']
def decide(out,labels,conf=0.6):
    if len(out['scores'])==0: return 'REJECT'
    import numpy as np
    for s,c in zip(out['scores'].cpu().numpy(), out['labels'].cpu().numpy()):
        if s<conf: continue
        if labels[c] in ('PE','PP'): return 'TARGET'
    return 'REJECT'
def draw(img,out,labels,conf=0.6):
    import numpy as np
    b=out['boxes'].cpu().numpy(); s=out['scores'].cpu().numpy(); c=out['labels'].cpu().numpy()
    for bb,ss,cc in zip(b,s,c):
        if ss<conf: continue
        x1,y1,x2,y2=bb.astype(int); name=labels[cc] if cc<len(labels) else str(cc)
        col=(0,200,0) if name in ('PE','PP') else (0,0,255)
        cv2.rectangle(img,(x1,y1),(x2,y2),col,2); cv2.putText(img,f"{name}:{ss:.2f}",(x1,y1-6),0,0.6,col,2)
    return img
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default=0); ap.add_argument('--weights',default='torchvision://maskrcnn_resnet50_fpn')
    ap.add_argument('--labels',default='labels.json'); ap.add_argument('--device',default='cuda'); ap.add_argument('--conf',type=float,default=0.6)
    args=ap.parse_args()
    labels=load_labels(args.labels); model=build_model(num_classes=len(labels),pretrained=True).to(args.device)
    if not args.weights.startswith('torchvision://'): model.load_state_dict(torch.load(args.weights,map_location=args.device))
    model.eval(); cam=OpenCVCamera(source=0 if str(args.source).isdigit() else args.source); act=Actuator(); fps=FPSMeter()
    while True:
        ok,frame=cam.read()
        if not ok: break
        rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); ten=torch.from_numpy(rgb).permute(2,0,1).float()/255.0
        with torch.no_grad(): out=model([ten.to(args.device)])[0]
        route=decide(out,labels,conf=args.conf); act.route_target() if route=='TARGET' else act.route_reject()
        vis=draw(frame.copy(),out,labels,conf=args.conf); draw_hud(vis,f'Route:{route} | FPS:{fps.tick():.1f}')
        cv2.imshow('POISE Screening', vis)
        if cv2.waitKey(1)&0xFF==27: break
    cam.release(); cv2.destroyAllWindows()
if __name__=='__main__': main()
