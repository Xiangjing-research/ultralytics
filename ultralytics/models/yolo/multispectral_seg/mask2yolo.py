import os
import cv2
from tqdm import tqdm
import json
import os
import glob
import os.path as osp

mask_path = 'D:\\DataSets\\PST900_RGBT_Dataset\\test\\seg_labels'

data_files = os.listdir(mask_path)

# color_list = []
# for data_file in tqdm(data_files):
#     img_file_path = os.path.join(data_path,data_file)
#     img = cv2.imread(img_file_path)
#     for x in range(img.shape[0]):
#         for y in range(img.shape[1]):
#             color = img[x,y]
#             color = list(color)
#             if color not in color_list:
#                 color_list.append(color)
#
#
# print(color_list)




import cv2
import os
import json
from PIL import Image
import io
import base64

# def rgb_to_gray_value(RGB):
#     R = RGB[0]
#     G = RGB[1]
#     B = RGB[2]
#     Gray = (R * 299 + G * 587 + B * 114) / 1000
#     return round(Gray)
#
#
# def bgr_2_rgb(color):
#     color[0], color[2] = color[2], color[0]
#     return color


class_dict = {
    "Fire-Extinguisher": 1,
    "Backpack": 2,
    "Hand-Drill": 3,
    "Survivor": 4
}


def func(file: str) -> dict:
    png = cv2.imread(file)
    gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    img_file_path = os.path.join(img_path, os.path.basename(file).split('.')[0] + '.png')
    img = Image.open(img_file_path)
    # imgData = img_tobyte(img)
    dic = {"version": "5.1.1", "flags": {}, "shapes": list(), "imagePath": os.path.basename(file),
           "imageHeight": png.shape[0], "imageWidth": png.shape[1]}

    for k, v in class_dict.items():

        binary = gray.copy()
        binary[binary != v] = 0
        binary[binary == v] = 255
        # 只检测外轮廓并存储所有的轮廓点
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            temp = list()
            if len(contour) < 4:
                continue
            for point in contour:
                temp.append([float(point[0][0]), float(point[0][1])])
            dic["shapes"].append({"label": k, "points": temp, "group_id": None,
                                  "shape_type": "polygon", "flags": {}})

    return dic


def labelme2yolov2Seg(jsonfilePath="", resultDirPath="", classList=["dusty", "defect", "damaged"]):
    """
    此函数用来将labelme软件标注好的数据集转换为yolov5_7.0sege中使用的数据集
    :param jsonfilePath: labelme标注好的*.json文件所在文件夹
    :param resultDirPath: 转换好后的*.txt保存文件夹
    :param classList: 数据集中的类别标签
    :return:
    """
    # 0.创建保存转换结果的文件夹
    if (not os.path.exists(resultDirPath)):
        os.mkdir(resultDirPath)

    # 1.获取目录下所有的labelme标注好的Json文件，存入列表中
    jsonfileList = glob.glob(osp.join(jsonfilePath, "*.json"))
    print(jsonfileList)  # 打印文件夹下的文件名称

    # 2.遍历json文件，进行转换
    for jsonfile in jsonfileList:
        # 3. 打开json文件
        with open(jsonfile, "r") as f:
            file_in = json.load(f)

            # 4. 读取文件中记录的所有标注目标
            shapes = file_in["shapes"]

            # 5. 使用图像名称创建一个txt文件，用来保存数据
            with open(resultDirPath + "\\" + jsonfile.split("\\")[-1].replace(".json", ".txt"), "w") as file_handle:
                # 6. 遍历shapes中的每个目标的轮廓
                for shape in shapes:
                    # 7.根据json中目标的类别标签，从classList中寻找类别的ID，然后写入txt文件中
                    file_handle.writelines(str(classList.index(shape["label"])) + " ")

                    # 8. 遍历shape轮廓中的每个点，每个点要进行图像尺寸的缩放，即x/width, y/height
                    for point in shape["points"]:
                        x = point[0] / file_in["imageWidth"]  # mask轮廓中一点的X坐标
                        y = point[1] / file_in["imageHeight"]  # mask轮廓中一点的Y坐标
                        file_handle.writelines(str(x) + " " + str(y) + " ")  # 写入mask轮廓点

                    # 9.每个物体一行数据，一个物体遍历完成后需要换行
                    file_handle.writelines("\n")
            # 10.所有物体都遍历完，需要关闭文件
            file_handle.close()
        # 10.所有物体都遍历完，需要关闭文件
        f.close()


if __name__ == "__main__":
    # mask2json
    img_path = 'D:\\DataSets\\PST900_RGBT_Dataset\\train\\rgb'
    mask_path = 'D:\\DataSets\\PST900_RGBT_Dataset\\train\\labels'
    save_path = 'D:\\DataSets\\PST900_RGBT_Dataset\\train\\anno_json'

    os.makedirs(save_path, exist_ok=True)

    mask_files = os.listdir(mask_path)
    for mask_file in mask_files:
        mask_file_path = os.path.join(mask_path, mask_file)
        save_file = mask_file.split('.')[0] + '.json'
        save_file_path = os.path.join(save_path, save_file)
        with open(save_file_path, mode='w', encoding='utf-8') as f:
            json.dump(func(mask_file_path), f)


    # json2yolo
    jsonfilePath = "D:\\DataSets\\PST900_RGBT_Dataset\\train\\anno_json"  # 要转换的json文件所在目录
    resultDirPath = "D:\\DataSets\\PST900_RGBT_Dataset\\train\\txt"  # 要生成的txt文件夹
    labelme2yolov2Seg(jsonfilePath=jsonfilePath, resultDirPath=resultDirPath, classList=["Fire-Extinguisher",
                                                                                         "Backpack",
                                                                                         "Hand-Drill",
                                                                                         "Survivor"])

    ## mask 可视化
    # import cv2
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    #
    # imgfile = 'D:\\DataSets\\PST900_RGBT_Dataset\\test\\rgb\\58_bag21_rect_rgb_frame0000000440.png'
    # pngfile = 'D:\\DataSets\\PST900_RGBT_Dataset\\test\\labels\\58_bag21_rect_rgb_frame0000000440.png'
    # img = cv2.imread(imgfile, 1)
    # mask = cv2.imread(pngfile, 0)
    #
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    #
    # img = img[:, :, ::-1]
    # img[..., 2] = np.where(mask == 4, 255, img[..., 2])
    #
    # plt.imshow(img)
    # plt.show()
    # # cv2.imwrite("visual/00001.jpg", img)

    # 获取灰度值
    gray_list = []

    for data_file in tqdm(data_files):
        img_file_path = os.path.join(mask_path, data_file)
        img = cv2.imread(img_file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                value = gray[x, y]
                if value not in gray_list:
                    gray_list.append(value)

    print(gray_list)
