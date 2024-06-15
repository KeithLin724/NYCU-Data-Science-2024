python train.py --config configs/default.yaml --save-path test_1 --test --opts MODEL.CHECKPOINT "segmentation/best_model_epoch_100.pth"
python create_submission.py --pred test_1 --save-file pred.csv
