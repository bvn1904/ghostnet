import torch
import numpy as np
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ghostnet import get_ghost_detection_model, CocoDetectionWrapper, collate_fn

def evaluate_model(model_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = get_ghost_detection_model(num_classes=91)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    val_dir = 'data/coco/val2017'
    val_ann = 'data/coco/annotations/instances_val2017.json'

    coco_gt = COCO(val_ann)
    dataset = CocoDetectionWrapper(root=val_dir, annFile=val_ann, transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)

    print("generating predictions on validation set...")
    results = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    x, y, x2, y2 = box
                    w = x2 - x
                    h = y2 - y
                    
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(score)
                    })
            
            if i % 50 == 0:
                print(f"Processed {i} batches...")

    print("calculating map...")
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    checkpoint_file = "ghost_fasterrcnn_epoch_1.pth" 
    evaluate_model(checkpoint_file)
