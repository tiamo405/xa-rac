# Xả rác
## 1. Data
```
data---train---video    ---name_video_1
            |           ---name_video_2
            |--images   ---name_video_1 ---id_person_1 ---frame_1.jpg
                        |               |              ---json_1.json
                        |               |              ---frame_2.jpg
                        |               |              ---json_2.json
                        |                ---id_person_2---frame_1.jpg
                        |                              ---json_1.json
                        |                              ---frame_2.jpg
                        |                              ---json_2.json
                        ---name_video_2
    ---test
```
* mỗi video tracking từng người , lưu các ảnh người đó vào 1 folder, trong đó detect các toạn độ pose dạng json
## 2. clone yolov5
```
https://github.com/ultralytics/yolov5.git
```

## 3. dowload weight
```
bash dowl_weight.sh
```
hoặc
```
https://drive.google.com/drive/folders/1Y4pj2DYfYnUyDYbquIXkcPBPHd0uXmfI?usp=sharing
```
checkpoints ---LSTM
            ---body_25.pth
            ---body_pose_model.pth
            ---hand_pose_model
            ---yolov5n.pt
## 4. cài đăt thư viện 
```
conda create --name name_env python=3.9
conda activate name_env
pip install -r requirements.txt

```

* cài đặt pytorch
sử dụng link hướng dẫn để cài đặt theo máy tính bản thân
```
https://pytorch.org/get-started/locally/
```


5. crop video, create json pose
```
python preprocessing/preprocessing.py
```



