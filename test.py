"""
从 test_images 读取第一张图片，检测后保存到 test_results，并显示结果。
标签样式与 GUI 保持一致（带背景、白字、细字体）。
"""

import os
import sys
import re
from pathlib import Path
import cv2
import torch

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.dataloaders import letterbox


def detect_one_image(model, img0, device, stride, img_size, conf_thres, iou_thres):
    """对单张图片进行检测，返回标注后的图片（标签样式与 GUI 一致）"""
    img, ratio, (dw, dh) = letterbox(img0, img_size, stride=stride, auto=True)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    det = pred[0]
    result_img = img0.copy()

    if len(det):
        det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            # 颜色
            color = (0, 0, 255) if c == 0 else (0, 180, 0)
            label_text = f"{'no_helmet' if c == 0 else 'helmet'} {conf:.2f}"

            x1, y1, x2, y2 = [int(v) for v in xyxy]

            # 1. 绘制检测框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

            # 2. 绘制标签背景
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            # 标签位置自适应
            if y1 - th - 6 > 10:          # 上方有足够空间
                label_y1 = y1 - th - 6
                label_y2 = y1
                text_y = y1 - 4
            else:                         # 放在框下方
                label_y1 = y2
                label_y2 = y2 + th + 6
                text_y = y2 + th + 4

            label_x1 = x1
            label_x2 = x1 + tw + 6

            # 绘制背景矩形
            cv2.rectangle(result_img, (label_x1, label_y1), (label_x2, label_y2), color, -1)
            # 绘制白色文字
            cv2.putText(result_img, label_text, (x1 + 3, text_y),
                        font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    return result_img


def get_next_filename(save_dir, prefix="result", ext=".jpg"):
    """自动生成下一个可用的文件名"""
    os.makedirs(save_dir, exist_ok=True)
    pattern = re.compile(rf"^{prefix}_(\d+)\.jpg$")
    max_num = 0
    for f in os.listdir(save_dir):
        match = pattern.match(f)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    next_num = max_num + 1
    return os.path.join(save_dir, f"{prefix}_{next_num:04d}{ext}")


def main():
    #  配置参数 
    weights_path = "runs/train/exp1/weights/best.pt"
    conf_thres = 0.5
    iou_thres = 0.45
    img_size = 640

    input_dir = "test_images"      # 输入图片文件夹
    output_dir = "test_results"    # 输出结果文件夹

    #  检查输入目录 
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"请将测试图片放入 '{input_dir}' 文件夹后重新运行。")
        return

    # 获取所有图片文件
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_exts)]
    files.sort()   # 按文件名排序，取第一个

    if not files:
        print(f"在 '{input_dir}' 中没有找到图片文件。")
        return

    img_filename = files[0]
    img_path = os.path.join(input_dir, img_filename)

    #  加载模型 
    device = select_device('')
    model = attempt_load(weights_path, device=device)
    stride = int(model.stride.max())
    model.names = ['no_helmet', 'helmet']

    #  读取图片并检测 
    img0 = cv2.imread(img_path)
    if img0 is None:
        print(f"无法读取图片: {img_path}")
        return

    result_img = detect_one_image(
        model, img0, device, stride, img_size,
        conf_thres, iou_thres
    )

    #  生成保存路径并保存 
    save_path = get_next_filename(output_dir)
    cv2.imwrite(save_path, result_img)
    print(f"检测完成！结果已保存为: {save_path}")

    #  显示结果 
    cv2.imshow("Detection Result", result_img)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()