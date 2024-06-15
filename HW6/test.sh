python train.py --config configs/default.yaml --test --opts MODEL.CHECKPOINT "segmentation/best_model_epoch_100.pth"
python create_submission.py --pred prediction --save-file pred.csv
