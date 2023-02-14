# Xả rác
1. Data
```
data---train---video    ---name_video_1
            |           ---name_vodeo_2
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
2. clone yolov5
```
https://github.com/ultralytics/yolov5.git
```

3. weight
```
bash dowl_weight.sh
```
checkpoints ---LSTM
            ---body_25.pth
            ---body_pose_model.pth
            ---hand_pose_model
            ---yolov5n.pt
4. crop video, create json pose
```
python debug.py
```
path_video="data/person.mp4", 
path_save="data/images"


