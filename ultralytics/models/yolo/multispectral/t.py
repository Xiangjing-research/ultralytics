import torch
import inspect

from torchvision import models
 # 引用显存跟踪代码
from ultralytics.models.yolo.multispectral.gpu_mem_track import MemTracker
from ultralytics import YOLO
from ultralytics.utils import TQDM

if __name__ == '__main__':

    # device = torch.device('cuda:0')
    #
    # # frame = inspect.currentframe()
    # gpu_tracker = MemTracker()      # 创建显存检测对象
    #
    # gpu_tracker.track()
    # model = YOLO(model='../../../cfg/models/v8/yolov8l-C2f_RepVit-CSFusion.yaml', task='multispectral').cuda()
    # gpu_tracker.track()

    from PIL import Image
    import os

    # 设置新尺寸
    new_size = (640, 512)

    # 获取当前文件夹路径
    folder_path = 'D:\\DataSets\\LLVIP\\images\\infrared\\val'

    # 遍历文件夹中的所有文件
    for filename in TQDM(os.listdir(folder_path)):
        # 如果是图片文件
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # 打开图片文件
            with Image.open(os.path.join(folder_path, filename)) as img:
                # 将图片调整为新尺寸
                img = img.resize(new_size, resample=Image.LANCZOS)
                # 保存图片文件
                img.save(os.path.join(folder_path, filename))
