import argparse, os, cv2, torch, json
from model import build_model
def load_labels(p): return json.load(open(p))['classes']
def draw(img,out,labels,conf=0.6):
    b=out['boxes'].cpu().numpy(); s=out['scores'].cpu().numpy(); c=out['labels'].cpu().numpy()
    for bb,ss,cc in zip(b,s,c):
        if ss<conf: continue
        x1,y1,x2,y2=bb.astype(int); name=labels[cc] if cc<len(labels) else str(cc)
        col=(0,200,0) if name in ('PE','PP') else (0,0,255)
        cv2.rectangle(img,(x1,y1),(x2,y2),col,2); cv2.putText(img,f"{name}:{ss:.2f}",(x1,y1-6),0,0.6,col,2)
    return img
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights',required=True); ap.add_argument('--source',required=True)
    ap.add_argument('--labels',default='labels.json'); ap.add_argument('--conf',type=float,default=0.6)
    ap.add_argument('--device',default='cuda'); ap.add_argument('--out',default='out_vis')
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    labels=load_labels(args.labels); model=build_model(num_classes=len(labels),pretrained=True).to(args.device)
    if not args.weights.startswith('torchvision://'): model.load_state_dict(torch.load(args.weights,map_location=args.device))
    model.eval()
    paths=[os.path.join(args.source,f) for f in os.listdir(args.source)] if os.path.isdir(args.source) else [args.source]
    for p in paths:
        img=cv2.imread(p); ten=torch.from_numpy(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).permute(2,0,1).float()/255.0
        with torch.no_grad(): out=model([ten.to(args.device)])[0]
        cv2.imwrite(os.path.join(args.out, os.path.basename(p)), draw(img.copy(), out, labels, conf=args.conf))
if __name__=='__main__': main()
