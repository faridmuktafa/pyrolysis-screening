import argparse, os, yaml, torch
from torch.utils.data import DataLoader
from dataset_coco import CocoLikeDataset, collate_fn
from model import build_model
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True); ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--bs',type=int,default=2); ap.add_argument('--lr',type=float,default=5e-4)
    ap.add_argument('--out',default='runs/mrcnn'); ap.add_argument('--device',default='cuda')
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    cfg=yaml.safe_load(open(args.data))
    ds_tr=CocoLikeDataset(cfg['train_images'], cfg['train_ann'])
    ds_va=CocoLikeDataset(cfg['val_images'], cfg['val_ann'])
    dl_tr=DataLoader(ds_tr,batch_size=args.bs,shuffle=True,collate_fn=collate_fn)
    model=build_model(num_classes=cfg['num_classes'],pretrained=True).to(args.device)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for ep in range(args.epochs):
        model.train()
        for imgs,targets in dl_tr:
            imgs=[im.to(args.device) for im in imgs]
            targets=[{k:v.to(args.device) for k,v in t.items()} for t in targets]
            loss=sum(model(imgs,targets).values())
            opt.zero_grad(); loss.backward(); opt.step()
        torch.save(model.state_dict(), os.path.join(args.out,f'epoch_{ep+1:03d}.pth'))
    torch.save(model.state_dict(), os.path.join(args.out,'best.pth'))
if __name__=='__main__': main()
