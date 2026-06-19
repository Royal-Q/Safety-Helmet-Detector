"""
安全帽佩戴检测系统 GUI
基于 YOLOv5 实现图片、视频和摄像头实时检测
"""

import sys
import os
import time
import threading
from pathlib import Path
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox


class HelmetDetectionGUI:
    """安全帽检测GUI主类"""

    def __init__(self, root):
        self.root = root
        self.root.title("厂区人员安全帽佩戴检测系统")
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        self.root.geometry("1400x850")
        self.root.resizable(True, True)

        # 颜色配置
        self.bg_color = "#f0f2f5"
        self.primary_color = "#1a73e8"
        self.success_color = "#34a853"
        self.danger_color = "#ea4335"
        self.root.configure(bg=self.bg_color)

        # 模型参数
        self.model = None
        self.device = None
        self.stride = None
        self.names = ['no_helmet', 'helmet']
        self.weights_path = "runs/train/exp1/weights/best.pt"
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.img_size = 640

        # 视频处理
        self.video_cap = None
        self.is_playing = False
        self.is_paused = False
        self.is_camera = False
        self.total_frames = 0
        self.current_frame_num = 0
        self.video_fps = 30
        self.current_video_path = None

        # 目标跟踪去重
        self.tracked_objects = []
        self.unique_helmet = 0
        self.unique_no_helmet = 0
        self.frame_counter = 0
        self.tracking_threshold = 30

        # 保存结果
        self.last_result_img = None
        self.last_video_frames = []
        self.current_mode = None

        # 告警防抖
        self._last_alert_time = 0

        # 告警存储目录
        os.makedirs("alerts/screenshots", exist_ok=True)
        self.alert_log_path = "alerts/alert_log.txt"

        self.create_widgets()

        # 启动时自动加载模型
        self.status_var.set("正在加载模型...")
        threading.Thread(target=self.load_model, daemon=True).start()

    def create_widgets(self):
        """创建所有界面组件"""
        main_frame = Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # 左侧面板
        left_panel = Frame(main_frame, bg=self.bg_color, width=240)
        left_panel.pack(side=LEFT, fill=Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        Label(left_panel, text="安全帽佩戴检测", font=("微软雅黑", 18, "bold"),
              fg=self.primary_color, bg=self.bg_color).pack(pady=(0, 15))

        # 模型状态框
        model_frame = LabelFrame(left_panel, text="模型状态", font=("微软雅黑", 10),
                                 bg=self.bg_color, fg="#333", padx=10, pady=5)
        model_frame.pack(fill=X, pady=5)
        self.model_status = Label(model_frame, text="加载中...", font=("微软雅黑", 10),
                                  fg="#666", bg=self.bg_color)
        self.model_status.pack(anchor=W, pady=2)
        self.status_var = StringVar(value="就绪")
        Label(model_frame, textvariable=self.status_var, font=("微软雅黑", 9),
              fg="#888", bg=self.bg_color).pack(anchor=W, pady=2)

        # 功能按钮
        btn_frame = Frame(left_panel, bg=self.bg_color)
        btn_frame.pack(fill=X, pady=10)

        self.img_btn = Button(btn_frame, text="图片检测", font=("微软雅黑", 11),
                              bg=self.primary_color, fg="white", relief=FLAT,
                              command=self.detect_image, height=2)
        self.img_btn.pack(fill=X, pady=4)

        self.video_btn = Button(btn_frame, text="视频检测", font=("微软雅黑", 11),
                                bg="#4285f4", fg="white", relief=FLAT,
                                command=self.detect_video, height=2)
        self.video_btn.pack(fill=X, pady=4)

        self.cam_btn = Button(btn_frame, text="摄像头实时检测", font=("微软雅黑", 11),
                              bg="#0d652d", fg="white", relief=FLAT,
                              command=self.toggle_camera, height=2)
        self.cam_btn.pack(fill=X, pady=4)

        # 检测参数
        param_frame = LabelFrame(left_panel, text="检测参数", font=("微软雅黑", 10),
                                 bg=self.bg_color, fg="#333", padx=10, pady=5)
        param_frame.pack(fill=X, pady=10)

        conf_frame = Frame(param_frame, bg=self.bg_color)
        conf_frame.pack(fill=X, pady=2)
        Label(conf_frame, text="置信度:", font=("微软雅黑", 9), bg=self.bg_color).pack(side=LEFT)
        self.conf_var = DoubleVar(value=0.5)
        Scale(conf_frame, from_=0.1, to=0.9, orient=HORIZONTAL,
              variable=self.conf_var, resolution=0.05, bg=self.bg_color,
              length=130, highlightthickness=0,
              command=lambda v: self.conf_label.config(text=f"{float(v):.2f}")).pack(side=RIGHT)
        self.conf_label = Label(conf_frame, text="0.5", font=("微软雅黑", 9), bg=self.bg_color, width=4)
        self.conf_label.pack(side=RIGHT, padx=(5, 0))

        iou_frame = Frame(param_frame, bg=self.bg_color)
        iou_frame.pack(fill=X, pady=2)
        Label(iou_frame, text="IoU:", font=("微软雅黑", 9), bg=self.bg_color).pack(side=LEFT)
        self.iou_var = DoubleVar(value=0.45)
        Scale(iou_frame, from_=0.1, to=0.9, orient=HORIZONTAL,
              variable=self.iou_var, resolution=0.05, bg=self.bg_color,
              length=130, highlightthickness=0,
              command=lambda v: self.iou_label.config(text=f"{float(v):.2f}")).pack(side=RIGHT)
        self.iou_label = Label(iou_frame, text="0.45", font=("微软雅黑", 9), bg=self.bg_color, width=4)
        self.iou_label.pack(side=RIGHT, padx=(5, 0))

        # 统计信息
        stats_frame = LabelFrame(left_panel, text="检测统计", font=("微软雅黑", 10),
                                 bg=self.bg_color, fg="#333", padx=10, pady=5)
        stats_frame.pack(fill=X, pady=5)

        self.stats_text = Text(stats_frame, font=("微软雅黑", 9), bg="#fafafa",
                               height=6, relief=FLAT, wrap=WORD)
        self.stats_text.pack(fill=X, pady=2)
        self.stats_text.config(state=NORMAL)
        self.stats_text.delete(1.0, END)
        self.stats_text.insert(END, "等待检测...")
        self.stats_text.config(state=DISABLED)

        # 告警记录列表
        alert_frame = LabelFrame(left_panel, text="告警记录", font=("微软雅黑", 10),
                                 bg=self.bg_color, fg="#333", padx=10, pady=5)
        alert_frame.pack(fill=BOTH, expand=True, pady=5)

        listbox_frame = Frame(alert_frame, bg=self.bg_color)
        listbox_frame.pack(fill=BOTH, expand=True)

        v_scrollbar = Scrollbar(listbox_frame, orient=VERTICAL)
        v_scrollbar.pack(side=RIGHT, fill=Y)

        h_scrollbar = Scrollbar(listbox_frame, orient=HORIZONTAL)
        h_scrollbar.pack(side=BOTTOM, fill=X)

        self.alert_listbox = Listbox(listbox_frame, font=("微软雅黑", 8),
                                     bg="#fafafa", relief=FLAT, height=8, width=35,
                                     yscrollcommand=v_scrollbar.set,
                                     xscrollcommand=h_scrollbar.set)
        self.alert_listbox.pack(side=LEFT, fill=BOTH, expand=True)

        v_scrollbar.config(command=self.alert_listbox.yview)
        h_scrollbar.config(command=self.alert_listbox.xview)

        # 右侧显示区域
        right_panel = Frame(main_frame, bg=self.bg_color)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True)

        self.display_frame = Frame(right_panel, bg="#333", relief=SUNKEN, bd=2)
        self.display_frame.pack(fill=BOTH, expand=True)

        self.image_label = Label(self.display_frame, bg="#333", text="请选择检测源",
                                 font=("微软雅黑", 16), fg="#888")
        self.image_label.pack(fill=BOTH, expand=True)

        # 底部控制栏
        bottom_bar = Frame(right_panel, bg=self.bg_color)
        bottom_bar.pack(fill=X, pady=(5, 0))

        progress_frame = Frame(bottom_bar, bg=self.bg_color)
        progress_frame.pack(fill=X, pady=(0, 4))
        self.progress_var = DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            maximum=100, length=400)
        self.progress_bar.pack(fill=X)
        self.progress_label = Label(progress_frame, text="就绪", font=("微软雅黑", 8),
                                    bg=self.bg_color, fg="#888")
        self.progress_label.pack(anchor=W)

        # 控制按钮行
        ctrl_frame = Frame(bottom_bar, bg=self.bg_color, height=40)
        ctrl_frame.pack(fill=X, pady=(2, 0))
        ctrl_frame.pack_propagate(False)

        left_spacer = Frame(ctrl_frame, bg=self.bg_color, width=100)
        left_spacer.pack(side=LEFT, fill=X, expand=True)

        self.control_group = Frame(ctrl_frame, bg=self.bg_color)

        self.resume_btn = Button(self.control_group, text="暂停", font=("微软雅黑", 10),
                                 bg="#fbbc04", fg="white", relief=FLAT,
                                 command=self.resume_detection, width=8, state=DISABLED)
        self.resume_btn.pack(side=LEFT, padx=2)

        self.stop_btn = Button(self.control_group, text="停止", font=("微软雅黑", 10),
                               bg="#ea4335", fg="white", relief=FLAT,
                               command=self.stop_detection, width=8, state=DISABLED)
        self.stop_btn.pack(side=LEFT, padx=2)

        self.replay_btn = Button(self.control_group, text="重新播放", font=("微软雅黑", 10),
                                 bg="#4285f4", fg="white", relief=FLAT,
                                 command=self.replay_video, width=10, state=DISABLED)
        self.replay_btn.pack_forget()

        self.control_group.pack_forget()

        right_spacer = Frame(ctrl_frame, bg=self.bg_color, width=100)
        right_spacer.pack(side=RIGHT, fill=X, expand=True)

        self.save_btn = Button(ctrl_frame, text="保存", font=("微软雅黑", 10, "bold"),
                               bg="#34a853", fg="white", relief=FLAT,
                               command=self.save_result, width=6, state=DISABLED)
        self.save_btn.pack_forget()

        # 底部状态信息
        info_frame = Frame(right_panel, bg=self.bg_color, height=28)
        info_frame.pack(fill=X, pady=(2, 0))
        self.info_label = Label(info_frame, text="就绪 | 欢迎使用安全帽佩戴检测系统",
                                font=("微软雅黑", 9), bg=self.bg_color, fg="#666", anchor=W)
        self.info_label.pack(fill=X)

    def update_progress(self, current, total):
        """更新进度条"""
        if total > 0:
            pct = (current / total) * 100
            self.progress_var.set(pct)
            self.progress_label.config(text=f"进度: {current}/{total} 帧 ({pct:.1f}%)")
        else:
            self.progress_var.set(0)
            self.progress_label.config(text="处理中...")

    def reset_progress(self):
        """重置进度条"""
        self.progress_var.set(0)
        self.progress_label.config(text="就绪")

    def show_controls(self, mode='hide'):
        """显示或隐藏控制按钮"""
        if mode == 'hide':
            self.control_group.pack_forget()
            self.resume_btn.pack_forget()
            self.stop_btn.pack_forget()
            self.replay_btn.pack_forget()
            self.resume_btn.config(state=DISABLED)
            self.stop_btn.config(state=DISABLED)
            self.replay_btn.config(state=DISABLED)

        elif mode == 'normal':
            self.control_group.pack(side=LEFT)
            self.resume_btn.pack(side=LEFT, padx=2)
            self.stop_btn.pack(side=LEFT, padx=2)
            self.replay_btn.pack_forget()
            self.resume_btn.config(state=NORMAL, text="暂停", bg="#fbbc04")
            self.stop_btn.config(state=NORMAL)
            self.replay_btn.config(state=DISABLED)

        elif mode == 'replay':
            self.control_group.pack(side=LEFT)
            self.resume_btn.pack_forget()
            self.stop_btn.pack_forget()
            self.replay_btn.pack(side=LEFT, padx=2)
            self.resume_btn.config(state=DISABLED)
            self.stop_btn.config(state=DISABLED)
            self.replay_btn.config(state=NORMAL)

    def show_save_button(self, show=True):
        """显示或隐藏保存按钮"""
        if show:
            self.save_btn.pack(side=RIGHT, padx=2)
            self.save_btn.config(state=NORMAL)
        else:
            self.save_btn.pack_forget()
            self.save_btn.config(state=DISABLED)

    def reset_tracking(self):
        """重置目标跟踪数据"""
        self.tracked_objects = []
        self.unique_helmet = 0
        self.unique_no_helmet = 0
        self.frame_counter = 0

    def load_model(self):
        """加载YOLOv5模型"""
        try:
            self.device = select_device('')
            self.model = attempt_load(self.weights_path, device=self.device)
            self.stride = int(self.model.stride.max())
            self.model.names = ['no_helmet', 'helmet']
            self.names = self.model.names

            # warmup
            dummy = torch.zeros(1, 3, self.img_size, self.img_size).to(self.device)
            self.model(dummy)

            self.root.after(0, lambda: self.model_status.config(
                text="模型已加载", fg=self.success_color))
            self.root.after(0, lambda: self.status_var.set("就绪"))
            self.root.after(0, lambda: self.info_label.config(
                text="模型加载成功 | 类别: no_helmet, helmet"))

        except Exception as e:
            self.root.after(0, lambda: self.model_status.config(
                text="加载失败", fg=self.danger_color))
            self.root.after(0, lambda: self.status_var.set(f"错误: {str(e)[:50]}"))
            messagebox.showerror("模型加载失败", f"请检查权重文件:\n{self.weights_path}\n\n错误: {e}")

    def match_object(self, cx, cy, frame_num):
        """匹配已有目标，返回匹配索引"""
        for i, (tx, ty, last_frame, cls) in enumerate(self.tracked_objects):
            dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if dist < self.tracking_threshold:
                self.tracked_objects[i] = (cx, cy, frame_num, cls)
                return i
        return None

    def prune_old_objects(self, frame_num, max_age=50):
        """移除超过max_age帧未出现的目标"""
        self.tracked_objects = [obj for obj in self.tracked_objects if frame_num - obj[2] < max_age]

    def process_frame(self, frame, frame_num=None):
        """核心检测函数，处理单帧图像"""
        if self.model is None:
            return frame

        try:
            img0 = frame.copy()
            img, ratio, (dw, dh) = letterbox(img0, self.img_size, stride=self.stride, auto=True)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.model(img_tensor)[0]

            pred = non_max_suppression(pred, self.conf_var.get(), self.iou_var.get())

            det = pred[0]
            if len(det):
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape).round()

                helmet_cnt = 0
                no_helmet_cnt = 0

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label_text = f'{self.names[c]} {conf:.2f}'
                    color = (0, 0, 180) if c == 0 else (0, 180, 0)

                    x1, y1, x2, y2 = [int(v) for v in xyxy]

                    # 绘制检测框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # 绘制标签
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                    if y1 - th - 6 > 10:
                        label_y1 = y1 - th - 6
                        label_y2 = y1
                        text_y = y1 - 4
                    else:
                        label_y1 = y2
                        label_y2 = y2 + th + 6
                        text_y = y2 + th + 4

                    label_x1 = x1
                    label_x2 = x1 + tw + 6
                    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
                    cv2.putText(frame, label_text, (x1 + 3, text_y),
                                font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

                    # 统计和跟踪
                    if c == 0:
                        no_helmet_cnt += 1
                        self.root.after(0, lambda conf=conf, f=frame.copy(): self.add_alert(conf, f))
                    else:
                        helmet_cnt += 1

                    if frame_num is not None:
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        if self.match_object(cx, cy, frame_num) is None:
                            if c == 0:
                                self.unique_no_helmet += 1
                            else:
                                self.unique_helmet += 1
                            self.tracked_objects.append((cx, cy, frame_num, c))

                if frame_num is None:
                    self.root.after(0, lambda h=helmet_cnt, n=no_helmet_cnt: self.update_stats_simple(h, n))
                else:
                    self.root.after(0, self.update_stats_tracked)

                if frame_num is not None and frame_num % 10 == 0:
                    self.prune_old_objects(frame_num)

            return frame

        except Exception as e:
            print(f"检测错误: {e}")
            return frame

    def update_stats_simple(self, helmet, no_helmet):
        """更新统计信息（图片模式）"""
        total = helmet + no_helmet
        stats = f"检测人数: {total}\n"
        stats += f"佩戴安全帽: {helmet}\n"
        stats += f"未佩戴安全帽: {no_helmet}\n"
        if total > 0:
            stats += f"佩戴率: {helmet / total * 100:.1f}%"
        else:
            stats += "佩戴率: 0.0%"

        self.stats_text.config(state=NORMAL)
        self.stats_text.delete(1.0, END)
        self.stats_text.insert(END, stats)
        self.stats_text.config(state=DISABLED)

    def update_stats_tracked(self):
        """更新统计信息（视频/摄像头模式，含去重）"""
        total = self.unique_helmet + self.unique_no_helmet
        stats = f"人员总数: {total}\n"
        stats += f"佩戴安全帽: {self.unique_helmet}\n"
        stats += f"未佩戴安全帽: {self.unique_no_helmet}\n"
        if total > 0:
            stats += f"佩戴率: {self.unique_helmet / total * 100:.1f}%"
        else:
            stats += "佩戴率: 0.0%"

        self.stats_text.config(state=NORMAL)
        self.stats_text.delete(1.0, END)
        self.stats_text.insert(END, stats)
        self.stats_text.config(state=DISABLED)

    def add_alert(self, conf, frame=None):
        """添加告警记录，保存截图和日志"""
        timestamp_full = time.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_short = time.strftime("%m-%d %H:%M")
        alert_msg = f"[{timestamp_short}] 未佩戴安全帽 (置信度: {conf:.2f})"

        self.alert_listbox.insert(0, alert_msg)
        if self.alert_listbox.size() > 50:
            self.alert_listbox.delete(END)
        self.alert_listbox.see(0)

        screenshot_path = None
        if frame is not None:
            screenshot_dir = "alerts/screenshots"
            os.makedirs(screenshot_dir, exist_ok=True)
            time_str = time.strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(screenshot_dir, f"alert_{time_str}.jpg")
            cv2.imwrite(screenshot_path, frame)

        log_entry = f"{timestamp_full} | 置信度:{conf:.2f} | 截图:{screenshot_path if screenshot_path else '未保存'}\n"
        with open(self.alert_log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

        # 摄像头模式弹窗告警（3秒防抖）
        if self.current_mode == 'camera':
            current_time = time.time()
            if current_time - self._last_alert_time < 3:
                return
            self._last_alert_time = current_time

            messagebox.showwarning(
                "安全告警",
                f"检测到未佩戴安全帽！\n\n"
                f"时间: {timestamp_full}\n"
                f"置信度: {conf:.2f}\n"
                f"截图已保存: {screenshot_path}\n\n"
                f"请立即处理！",
                parent=self.root
            )

    def detect_image(self):
        """图片检测功能"""
        if self.model is None:
            messagebox.showwarning("提示", "模型正在加载中...")
            return

        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif")]
        )
        if not file_path:
            return

        self.stop_detection()
        self.reset_tracking()
        self.alert_listbox.delete(0, END)
        self.reset_progress()
        self.show_controls('hide')
        self.show_save_button(False)
        self.last_video_frames = []
        self.current_video_path = None
        self.current_mode = 'image'

        self.stats_text.config(state=NORMAL)
        self.stats_text.delete(1.0, END)
        self.stats_text.insert(END, "检测中...")
        self.stats_text.config(state=DISABLED)

        img0 = cv2.imread(file_path)
        if img0 is None:
            messagebox.showerror("错误", "无法读取图片")
            return

        self.info_label.config(text="正在检测图片...")
        result_img = self.process_frame(img0, frame_num=None)

        if result_img is not None:
            self.last_result_img = result_img.copy()
            self.display_image(result_img)
            self.info_label.config(text=f"检测完成: {os.path.basename(file_path)}")
            self.show_save_button(True)
            self.progress_var.set(100)
            self.progress_label.config(text="检测完成")
        else:
            self.info_label.config(text="检测失败，请重试")

    def detect_video(self):
        """视频检测功能"""
        if self.model is None:
            messagebox.showwarning("提示", "模型正在加载中...")
            return

        file_path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv")]
        )
        if not file_path:
            return

        self.stop_detection()
        self.reset_tracking()
        self.alert_listbox.delete(0, END)
        self.reset_progress()
        self.show_save_button(False)
        self.last_result_img = None
        self.current_video_path = file_path
        self.last_video_frames = []
        self.current_mode = 'video'

        self.stats_text.config(state=NORMAL)
        self.stats_text.delete(1.0, END)
        self.stats_text.insert(END, "检测中...")
        self.stats_text.config(state=DISABLED)

        self.video_cap = cv2.VideoCapture(file_path)
        if not self.video_cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件")
            return

        self.is_playing = True
        self.is_paused = False
        self.is_camera = False
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30
        self.current_frame_num = 0

        self.show_controls('normal')
        self.info_label.config(text=f"正在播放: {os.path.basename(file_path)} ({self.total_frames}帧)")

        threading.Thread(target=self.play_video, daemon=True).start()

    def play_video(self):
        """视频播放线程"""
        try:
            frame_count = 0
            while self.is_playing and self.video_cap is not None:
                if self.is_paused:
                    time.sleep(0.1)
                    continue

                ret, frame = self.video_cap.read()
                if not ret:
                    break

                frame_count += 1
                self.current_frame_num = frame_count
                self.frame_counter += 1

                result_frame = self.process_frame(frame, frame_num=self.frame_counter)

                if result_frame is not None:
                    self.last_video_frames.append(result_frame.copy())
                    self.root.after(0, lambda f=result_frame: self.display_image(f))
                    self.root.after(0, lambda c=frame_count: self.info_label.config(
                        text=f"处理帧: {c}/{self.total_frames}"))
                    self.root.after(0, lambda c=frame_count: self.update_progress(c, self.total_frames))

                time.sleep(1 / self.video_fps if self.video_fps > 0 else 0.03)

            self.root.after(0, lambda: self.info_label.config(text="视频检测完成"))
            self.root.after(0, lambda: self.progress_label.config(text="检测完成"))
            self.root.after(0, lambda: self.show_controls('hide'))
            if len(self.last_video_frames) > 0:
                self.root.after(0, lambda: self.show_save_button(True))
        finally:
            self.cleanup_video()

    def toggle_camera(self):
        """切换摄像头状态"""
        if self.model is None:
            messagebox.showwarning("提示", "模型正在加载中...")
            return

        if self.is_camera:
            self.stop_detection()
            return

        self.stop_detection()
        self.reset_tracking()
        self.alert_listbox.delete(0, END)
        self.reset_progress()
        self.show_save_button(False)
        self.last_result_img = None
        self.current_video_path = None
        self.last_video_frames = []
        self.current_mode = 'camera'

        self.stats_text.config(state=NORMAL)
        self.stats_text.delete(1.0, END)
        self.stats_text.insert(END, "检测中...")
        self.stats_text.config(state=DISABLED)

        self.video_cap = cv2.VideoCapture(0)
        if not self.video_cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return

        self.is_playing = True
        self.is_paused = False
        self.is_camera = True
        self.total_frames = 0

        self.show_controls('normal')
        self.cam_btn.config(text="关闭摄像头", bg=self.danger_color)
        self.info_label.config(text="摄像头实时检测中...")

        threading.Thread(target=self.process_camera, daemon=True).start()

    def process_camera(self):
        """摄像头处理线程"""
        try:
            while self.is_playing and self.video_cap is not None:
                if self.is_paused:
                    time.sleep(0.1)
                    continue

                ret, frame = self.video_cap.read()
                if not ret:
                    break

                self.frame_counter += 1
                result_frame = self.process_frame(frame, frame_num=self.frame_counter)

                if result_frame is not None:
                    self.root.after(0, lambda f=result_frame: self.display_image(f))

                time.sleep(0.03)

            self.root.after(0, lambda: self.show_controls('hide'))
            self.root.after(0, lambda: self.cam_btn.config(text="摄像头实时检测", bg="#0d652d"))
        finally:
            self.cleanup_video()

    def resume_detection(self):
        """暂停/继续检测"""
        if self.is_playing:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.resume_btn.config(text="继续", bg="#fbbc04")
                self.info_label.config(text="已暂停")
            else:
                self.resume_btn.config(text="暂停", bg="#fbbc04")
                self.info_label.config(text="继续检测中...")

    def stop_detection(self):
        """停止检测"""
        self.is_playing = False
        self.is_paused = False

        if self.is_camera:
            self.is_camera = False
            self.cam_btn.config(text="摄像头实时检测", bg="#0d652d")
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None
            self.show_controls('hide')
            self.show_save_button(False)
            self.info_label.config(text="已停止检测")
            self.reset_tracking()
            return

        if self.current_mode == 'video' and self.current_video_path:
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None

            self.show_controls('replay')
            self.info_label.config(text="已停止，点击重新播放从头开始")
            if len(self.last_video_frames) > 0:
                self.show_save_button(True)
        else:
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None
            self.show_controls('hide')
            self.show_save_button(False)
            self.info_label.config(text="已停止检测")
        self.reset_tracking()

    def replay_video(self):
        """重新播放视频"""
        if not self.current_video_path:
            messagebox.showwarning("提示", "没有可重新播放的视频")
            return

        self.last_video_frames = []
        self.reset_tracking()
        self.alert_listbox.delete(0, END)
        self.reset_progress()
        self.show_save_button(False)

        self.stats_text.config(state=NORMAL)
        self.stats_text.delete(1.0, END)
        self.stats_text.insert(END, "检测中...")
        self.stats_text.config(state=DISABLED)

        self.video_cap = cv2.VideoCapture(self.current_video_path)
        if not self.video_cap.isOpened():
            messagebox.showerror("错误", "无法重新打开视频文件")
            return

        self.is_playing = True
        self.is_paused = False
        self.is_camera = False
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30
        self.current_frame_num = 0

        self.show_controls('normal')
        self.info_label.config(text=f"重新播放: {os.path.basename(self.current_video_path)} ({self.total_frames}帧)")

        threading.Thread(target=self.play_video, daemon=True).start()

    def cleanup_video(self):
        """释放视频资源"""
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.is_playing = False
        self.root.after(0, lambda: self.progress_label.config(text="播放结束"))

    def save_result(self):
        """保存检测结果"""
        if self.current_mode == 'image':
            if self.last_result_img is None:
                messagebox.showwarning("提示", "没有可保存的图片")
                return
            path = filedialog.asksaveasfilename(
                title="保存图片",
                defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
            )
            if path:
                cv2.imwrite(path, self.last_result_img)
                messagebox.showinfo("成功", f"图片已保存至:\n{path}")

        elif self.current_mode == 'video':
            if not self.last_video_frames:
                messagebox.showwarning("提示", "没有可保存的视频")
                return
            path = filedialog.asksaveasfilename(
                title="保存视频",
                defaultextension=".mp4",
                filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")]
            )
            if not path:
                return
            try:
                h, w = self.last_video_frames[0].shape[:2]
                fps = self.video_fps if self.video_fps > 0 else 30
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(path, fourcc, fps, (w, h))
                for frame in self.last_video_frames:
                    out.write(frame)
                out.release()
                messagebox.showinfo("成功", f"视频已保存至:\n{path}\n总帧数: {len(self.last_video_frames)}")
            except Exception as e:
                messagebox.showerror("失败", f"保存出错:\n{e}")

        else:
            messagebox.showwarning("提示", "没有可保存的结果")

    def display_image(self, img):
        """在界面上显示图片"""
        try:
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()

            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img

            h, w = img_rgb.shape[:2]
            display_w = self.display_frame.winfo_width() or 800
            display_h = self.display_frame.winfo_height() or 600

            scale = min(display_w / w, display_h / h) * 0.95
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_rgb, (new_w, new_h))

            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(img_pil)

            self.image_label.config(image=img_tk, text="")
            self.image_label.image = img_tk

        except Exception as e:
            print(f"显示错误: {e}")

    def on_closing(self):
        """窗口关闭时释放资源"""
        self.stop_detection()
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = HelmetDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()