1. Training with max input size - exp 3
  - img: 960x960
  - batch: 16
  - model: yolov5l.yaml
  - hyp: no-augmentation
  - save-period: 100
  - `python3 train.py --img 960 --batch 16 --epoch 30000 --weight '' --cfg training/CE1-UPR/yolov5l.yaml --data training/CE1-UPR/training.yaml --hyp training/CE1-UPR/aug.yaml --save-period 100 --cache ram`
  - BETTER mAP so far

2. Training with bigger model - exp 8
  - img: 800x800 (unable to load into GPU memory since 960x960 is too big)
  - batch: 16
  - model: yolov5x.yaml
  - hyp: no-augmentation
  - save-period: 100
  - `python3 train.py --img 800 --batch 16 --epoch 30000 --weight '' --cfg training/CE1-UPR/yolov5x.yaml --data training/CE1-UPR/training.yaml --hyp training/CE1-UPR/aug.yaml --save-period 100 --cache ram`

3. Training with augmentation based on exp 3 - exp 4
  - WARNING ⚠️ NMS time limit 2.100s exceeded

4. Training #3 result was the best -> No early stopping & continuous training based on exp 4- exp 5
5. 
