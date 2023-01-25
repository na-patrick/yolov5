# 1. Training Attempt #1
## Training with initial hyperparameters
```sh
python3 train.py --img 960 --batch 16 --epoch 20000 --weight '' --cfg training/CE1-MAT/yolov5l.yaml --data training/CE1-MAT/training.yaml --hyp training/CE1-MAT/aug1.yaml --save-period 100 --cache ram --patience 300
```

## Beginning of Training Log
```
train: weights=, cfg=training/CE1-MAT/yolov5l.yaml, data=training/CE1-MAT/training.yaml, hyp=training/CE1-MAT/aug1.yaml, epochs=20000, batch_size=16, imgsz=960, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=300, freeze=[0], save_period=100, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
remote: Enumerating objects: 10, done.
remote: Counting objects: 100% (8/8), done.
remote: Compressing objects: 100% (6/6), done.
remote: Total 10 (delta 2), reused 4 (delta 2), pack-reused 2
Unpacking objects: 100% (10/10), 18.75 KiB | 9.38 MiB/s, done.
From https://github.com/ultralytics/yolov5
   589edc7..064365d  master          -> origin/master
 * [new branch]      fix_zero_labels -> origin/fix_zero_labels
github: ⚠️ YOLOv5 is out of date by 9 commits. Use `git pull` or `git clone https://github.com/ultralytics/yolov5` to update.
YOLOv5 🚀 v7.0-63-gcdd804d Python-3.8.10 torch-1.13.1+cu117 CUDA:0 (NVIDIA TITAN RTX, 24212MiB)

hyperparameters: lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.3, cls_pw=1.0, obj=0.7, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10.0, translate=0.1, scale=0.5, shear=0, perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0
ClearML: run 'pip install clearml' to automatically track, visualize and remotely train YOLOv5 🚀 in ClearML
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/

                 from  n    params  module                                  arguments                     
  0                -1  1      7040  models.common.Conv                      [3, 64, 6, 2, 2]              
  1                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  2                -1  3    156928  models.common.C3                        [128, 128, 3]                 
  3                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  4                -1  6   1118208  models.common.C3                        [256, 256, 6]                 
  5                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  6                -1  9   6433792  models.common.C3                        [512, 512, 9]                 
  7                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 2]             
  8                -1  3   9971712  models.common.C3                        [1024, 1024, 3]               
  9                -1  1   2624512  models.common.SPPF                      [1024, 1024, 5]               
 10                -1  1    525312  models.common.Conv                      [1024, 512, 1, 1]             
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  3   2757632  models.common.C3                        [1024, 512, 3, False]         
 14                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  3    690688  models.common.C3                        [512, 256, 3, False]          
 18                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  3   2495488  models.common.C3                        [512, 512, 3, False]          
 21                -1  1   2360320  models.common.Conv                      [512, 512, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  3   9971712  models.common.C3                        [1024, 1024, 3, False]        
 24      [17, 20, 23]  1     80775  models.yolo.Detect                      [10, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 1024]]
YOLOv5l summary: 368 layers, 46186759 parameters, 46186759 gradients, 108.4 GFLOPs

AMP: checks passed ✅
optimizer: SGD(lr=0.01) with parameter groups 101 weight(decay=0.0), 104 weight(decay=0.0005), 104 bias
train: Scanning /home/patrick/NeuronAware/yolov5/training/CE1-MAT/train_imgs... 863 images, 160 backgrounds, 0 corrupt: 100%|██████████| 863/863 [00:00<00:00, 13777.6
train: New cache created: /home/patrick/NeuronAware/yolov5/training/CE1-MAT/train_imgs.cache
train: Caching images (1.4GB ram): 100%|██████████| 863/863 [00:03<00:00, 270.20it/s]
val: Scanning /home/patrick/NeuronAware/yolov5/training/CE1-MAT/validation_imgs... 215 images, 40 backgrounds, 0 corrupt: 100%|██████████| 215/215 [00:00<00:00, 5590.
val: New cache created: /home/patrick/NeuronAware/yolov5/training/CE1-MAT/validation_imgs.cache
val: Caching images (0.3GB ram): 100%|██████████| 215/215 [00:00<00:00, 269.82it/s]

AutoAnchor: 4.64 anchors/target, 0.974 Best Possible Recall (BPR). Anchors are a poor fit to dataset ⚠️, attempting to improve...
AutoAnchor: Running kmeans for 9 anchors on 8525 points...
AutoAnchor: Evolving anchors with Genetic Algorithm: fitness = 0.9235: 100%|██████████| 1000/1000 [00:00<00:00, 2198.35it/s]
AutoAnchor: thr=0.25: 1.0000 best possible recall, 6.17 anchors past thr
AutoAnchor: n=9, img_size=960, metric_all=0.401/0.923-mean/best, past_thr=0.515-mean: 32,20, 53,60, 90,36, 107,38, 75,74, 158,38, 446,24, 58,206, 158,78
AutoAnchor: Done ✅ (optional: update model *.yaml to use these anchors in the future)

Plotting labels to runs/train/exp/labels.jpg... 
Image sizes 960 train, 960 val
Using 8 dataloader workers
Logging results to runs/train/exp
Starting training for 20000 epochs...
```

## End of Training Log
```
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
 1192/19999      22.4G   0.008747    0.01146  0.0005985        115        960: 100%|██████████| 54/54 [00:28<00:00,  1.87it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 7/7 [00:02<00:00,  3.21it/s]
                   all        215       2121      0.999          1      0.995      0.884
Stopping training early as no improvement observed in last 300 epochs. Best results observed at epoch 892, best model saved as best.pt.
To update EarlyStopping(patience=300) pass a new patience value, i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.

1193 epochs completed in 10.928 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 93.0MB
Optimizer stripped from runs/train/exp/weights/best.pt, 93.0MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
YOLOv5l summary: 267 layers, 46156743 parameters, 0 gradients, 107.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 7/7 [00:05<00:00,  1.37it/s]
                   all        215       2121      0.999          1      0.995      0.892
              PAD NO.1        215        254      0.999          1      0.995      0.928
              PAD NO.2        215        254      0.999          1      0.995      0.872
              PAD NO.3        215        127      0.999          1      0.995      0.884
              PAD NO.4        215        254      0.999          1      0.995      0.929
              PAD NO.5        215        126      0.998          1      0.995      0.988
              PAD NO.6        215        254      0.999          1      0.995       0.97
              PAD NO.8        215        254      0.999          1      0.995      0.918
              PAD NO.9        215        127          1          1      0.995      0.745
          STRAP-WASHER        215        121      0.999          1      0.995      0.853
     T/VERSE HOOK HOLE        215        350          1          1      0.995      0.837
Results saved to runs/train/exp
```

# 2. Exporting training result to ONNX
```sh
python3 export.py --weights runs/train/exp4/weights/*.pt --device 0 --include onnx --data training/CE1-UPR/training.yaml --img 960 --opset 15
```
