import json, os, cv2, numpy as np, torch
from torch.utils.data import Dataset
class CocoLikeDataset(Dataset):
    def __init__(self, images_dir, anno_json, transforms=None):
        self.images_dir = images_dir; self.transforms = transforms
        with open(anno_json,'r') as f: coco=json.load(f)
        self.images={im['id']:im for im in coco['images']}
        self.anns={}; [self.anns.setdefault(a['image_id'],[]).append(a) for a in coco['annotations']]
        self.ids=list(self.images.keys())
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        img_id=self.ids[idx]; im=self.images[img_id]
        path=os.path.join(self.images_dir, im['file_name'])
        img=cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        H,W=img.shape[:2]; anns=self.anns.get(img_id,[])
        boxes,labels,masks=[],[],[]
        for a in anns:
            x,y,w,h=a['bbox']; boxes.append([x,y,x+w,y+h]); labels.append(a['category_id'])
            if 'segmentation' in a and isinstance(a['segmentation'], list):
                mask=np.zeros((H,W),dtype=np.uint8)
                for poly in a['segmentation']:
                    pts=np.array(poly,dtype=np.int32).reshape(-1,2); cv2.fillPoly(mask,[pts],1)
                masks.append(mask)
            else: masks.append(np.zeros((H,W),dtype=np.uint8))
        target={'boxes': torch.as_tensor(np.array(boxes),dtype=torch.float32) if boxes else torch.zeros((0,4),dtype=torch.float32),
                'labels': torch.as_tensor(np.array(labels),dtype=torch.int64) if labels else torch.zeros((0,),dtype=torch.int64),
                'masks': torch.as_tensor(np.stack(masks,axis=0),dtype=torch.uint8) if masks else torch.zeros((0,H,W),dtype=torch.uint8),
                'image_id': torch.tensor([img_id])}
        img=torch.from_numpy(img).permute(2,0,1).float()/255.0
        if self.transforms: img,target=self.transforms(img,target)
        return img,target
def collate_fn(batch): return tuple(zip(*batch))
