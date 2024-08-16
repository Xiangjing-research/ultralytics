<h1>多模态YOLOv8</h1>

<summary><h2>网络结构图</h2></summary>
<img src="image\net2.png"  width="964" height="776">

<h3>DFMDA结构图</h3>
<img src="image\DFMDA.png" width="548" height="261" >

<h3>多头空洞注意力</h3>
<img src="image\MHDWA2.png" width="362" height="440">

<summary><h2>安装</h2></summary>

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

[![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

```bash
pip install -e .
```

<summary><h2>训练</h2></summary>

### Python

YOLOv8 may also be used directly in a Python environment, and accepts the same [arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

```python
from ultralytics.models.yolo.multispectral import MultispectralDetectionTrainer

args = dict(task='multispectral', mode='train', model='../../../cfg/models/v8/yolov8l-C2f_RepVit-CSFusion.yaml',
            data=' ../../../cfg/datasets/LLVIP.yaml', epochs=1, batch=4, project='v8_multispectral',
            name='train_det')
trainer = MultispectralDetectionTrainer(overrides=args)
trainer.train()
```

<summary><h2>测试</h2></summary>

```python
model = YOLO(model='../../../cfg/models/v8/yolov8l-C2f_RepVit-CSFusion.yaml', task='multispectral',verbose=True)
model.predict(
    ['D:\\DataSets\\LLVIP\\images\\visible\\train\\020344.jpg', 'D:\\DataSets\\LLVIP\\images\\infrared\\train\\020344.jpg'],
    save=True, name='predict')
```

<summary>Pedestrian Detection (LLVIP)</summary>
coming soon

| Model<br><sup> | Modality<br><sup> | Param<br><sup> | GFLOPs<br><sup> | AP50<br><sup> | AP75<br><sup> | mAP<br><sup> |
|----------------|-------------------|----------------|-----------------|---------------|---------------|--------------|
| CFR            | RGB+T             | -              | -               | -             | -             | -            |
| GAFF           | RGB+T             | -              | -               | -             | -             | -            |
| ProbEn         | RGB+T             | -              | -               | -             | -             | -            |
| CTF            | RGB+T             | -              | -               | -             | -             | -            |
| CMDFT          | RGB+T             | -              | -               | -             | -             | -            |


<summary><h2>可视化</h2></summary>
<figure class="half">
    <img src="image\LLVIP\ours_visible_081305.jpg"  width="482">
    <img src="image\LLVIP\ours_infrared_081305.jpg"  width="482">
</figure>
