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
import ttkbootstrap as tb
from ttkbootstrap.constants import *

# 动态添加模型工具路径
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox


class HelmetDetectionGUI:
    """安全帽佩戴检测系统主界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("厂区人员安全帽佩戴检测系统")
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass

        # 界面与模型参数初始化
        self.current_theme = "lumen"
        self.model = None
        self.device = None
        self.stride = None
        self.names = ['no_helmet', 'helmet']
        self.weights_path = "runs/train/exp1/weights/best.pt"
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.img_size = 640

        # 视频/摄像头相关状态
        self.video_cap = None
        self.is_playing = False
        self.is_paused = False
        self.is_camera = False
        self.total_frames = 0
        self.current_frame_num = 0
        self.video_fps = 30
        self.current_video_path = None

        # 目标跟踪统计
        self.tracked_objects = []
        self.unique_helmet = 0
        self.unique_no_helmet = 0
        self.frame_counter = 0
        self.tracking_threshold = 30

        # 结果缓存
        self.last_result_img = None
        self.last_video_frames = []
        self.current_mode = None

        # 告警相关
        self._last_alert_time = 0
        os.makedirs("alerts/screenshots", exist_ok=True)
        self.alert_log_path = "alerts/alert_log.txt"

        # 跳帧与显示控制
        self.seek_frame = -1
        self.current_display_image = None
        self.last_image_path = None

        self.create_widgets()

        self.status_var.set("正在加载模型...")
        threading.Thread(target=self.load_model, daemon=True).start()

    #  UI 构建 
    def create_widgets(self):
        """创建界面所有控件"""
        main_frame = tb.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=20, pady=15)

        # 左侧控制面板
        left_panel = tb.Frame(main_frame, width=300)
        left_panel.pack(side=LEFT, fill=Y, padx=(0, 20))
        left_panel.pack_propagate(False)

        title = tb.Label(left_panel, text="安全帽佩戴检测", font=("微软雅黑", 18, "bold"), bootstyle="primary")
        title.pack(pady=(0, 20))

        # 模型状态区域
        model_frame = ttk.LabelFrame(left_panel, text="模型状态")
        model_frame.pack(fill=X, pady=5)
        self.model_status = tb.Label(model_frame, text="加载中...", font=("微软雅黑", 10))
        self.model_status.pack(anchor=W, pady=5, padx=5)
        self.status_var = StringVar(value="就绪")
        tb.Label(model_frame, textvariable=self.status_var, font=("微软雅黑", 9)).pack(anchor=W, pady=(0, 5), padx=5)

        # 功能按钮
        btn_frame = tb.Frame(left_panel)
        btn_frame.pack(fill=X, pady=15)
        self.img_btn = tb.Button(btn_frame, text="图片检测", bootstyle="primary", command=self.detect_image)
        self.img_btn.pack(fill=X, pady=5)
        self.video_btn = tb.Button(btn_frame, text="视频检测", bootstyle="primary", command=self.detect_video)
        self.video_btn.pack(fill=X, pady=5)
        self.cam_btn = tb.Button(btn_frame, text="摄像头实时检测", bootstyle="success", command=self.toggle_camera)
        self.cam_btn.pack(fill=X, pady=5)

        # 检测参数调节
        param_frame = ttk.LabelFrame(left_panel, text="检测参数")
        param_frame.pack(fill=X, pady=10)

        conf_frame = tb.Frame(param_frame)
        conf_frame.pack(fill=X, pady=5, padx=5)
        tb.Label(conf_frame, text="置信度:", font=("微软雅黑", 9)).pack(side=LEFT)
        self.conf_var = DoubleVar(value=0.5)
        scale = tb.Scale(conf_frame, from_=0.1, to=0.9, variable=self.conf_var,
                         orient=HORIZONTAL, length=130, bootstyle="primary")
        scale.pack(side=RIGHT, padx=(5, 0))
        self.conf_label = tb.Label(conf_frame, text="0.50", width=4, font=("微软雅黑", 9))
        self.conf_label.pack(side=RIGHT)

        iou_frame = tb.Frame(param_frame)
        iou_frame.pack(fill=X, pady=5, padx=5)
        tb.Label(iou_frame, text="IoU:", font=("微软雅黑", 9)).pack(side=LEFT)
        self.iou_var = DoubleVar(value=0.45)
        scale_iou = tb.Scale(iou_frame, from_=0.1, to=0.9, variable=self.iou_var,
                             orient=HORIZONTAL, length=130, bootstyle="primary")
        scale_iou.pack(side=RIGHT, padx=(5, 0))
        self.iou_label = tb.Label(iou_frame, text="0.45", width=4, font=("微软雅黑", 9))
        self.iou_label.pack(side=RIGHT)

        # 检测统计显示
        stats_frame = ttk.LabelFrame(left_panel, text="检测统计")
        stats_frame.pack(fill=X, pady=10)
        self.stats_text = Text(stats_frame, font=("微软雅黑", 9), bg="#fafafa",
                               height=6, relief=FLAT, wrap=WORD, borderwidth=0)
        self.stats_text.pack(fill=X, pady=5, padx=5)
        self.stats_text.config(state=NORMAL)
        self.stats_text.delete(1.0, END)
        self.stats_text.insert(END, "等待检测...")
        self.stats_text.config(state=DISABLED)

        # 告警记录列表
        alert_frame = ttk.LabelFrame(left_panel, text="告警记录")
        alert_frame.pack(fill=BOTH, expand=True, pady=5)

        listbox_frame = tb.Frame(alert_frame)
        listbox_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)

        v_scrollbar = tb.Scrollbar(listbox_frame, orient=VERTICAL)
        v_scrollbar.pack(side=RIGHT, fill=Y)
        h_scrollbar = tb.Scrollbar(listbox_frame, orient=HORIZONTAL)
        h_scrollbar.pack(side=BOTTOM, fill=X)

        self.alert_listbox = Listbox(listbox_frame, font=("微软雅黑", 8),
                                     bg="#fafafa", relief=FLAT, height=8,
                                     yscrollcommand=v_scrollbar.set,
                                     xscrollcommand=h_scrollbar.set,
                                     borderwidth=0, highlightthickness=0)
        self.alert_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        v_scrollbar.config(command=self.alert_listbox.yview)
        h_scrollbar.config(command=self.alert_listbox.xview)

        # 右侧显示区域
        right_panel = tb.Frame(main_frame)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=(0, 5))

        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=0)
        right_panel.grid_rowconfigure(2, weight=0)
        right_panel.grid_columnconfigure(0, weight=1)

        display_frame_border = tb.Frame(right_panel)
        display_frame_border.grid(row=0, column=0, sticky="nsew")

        self.display_frame = Frame(display_frame_border, bg="#e6e6e6", bd=0,
                                   highlightthickness=1, highlightbackground="#b0b0b0")
        self.display_frame.pack(fill=BOTH, expand=True, padx=2, pady=2)

        self.image_label = Label(self.display_frame, bg="#e6e6e6",
                                 text="请选择检测源",
                                 font=("微软雅黑", 24, "bold"), fg="#7a7a7a", justify=CENTER)
        self.image_label.pack(fill=BOTH, expand=True, padx=1, pady=1)
        self.display_frame.bind("<Configure>", self.on_display_resize)

        separator = ttk.Separator(right_panel, orient=HORIZONTAL)
        separator.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # 底部控制条
        bottom_frame = Frame(right_panel, bg=self.root.cget('bg'))
        bottom_frame.grid(row=2, column=0, sticky="ew", padx=2, pady=(0, 5))

        # 进度条
        progress_frame = tb.Frame(bottom_frame)
        progress_frame.pack(fill=X, pady=(8, 2), padx=5)
        self.progress_frame = progress_frame

        self.progress_var = DoubleVar(value=0)
        self.progress_slider = tb.Scale(progress_frame, variable=self.progress_var,
                                        from_=0, to=100, orient=HORIZONTAL,
                                        bootstyle="primary")
        self.progress_slider.pack(fill=X)
        self.progress_slider.bind("<ButtonRelease-1>", self.on_slider_release)
        self.progress_slider.pack_forget()

        self.progress_label = Label(progress_frame, text="就绪", font=("微软雅黑", 8),
                                    fg="#28a745", bg=self.root.cget('bg'))
        self.progress_label.pack(anchor=W)

        # 播放控制按钮
        ctrl_frame = tb.Frame(bottom_frame, height=40)
        ctrl_frame.pack(fill=X, pady=(2, 8), padx=5)
        ctrl_frame.pack_propagate(False)

        left_spacer = tb.Frame(ctrl_frame, width=100)
        left_spacer.pack(side=LEFT, fill=X, expand=True)

        self.control_group = tb.Frame(ctrl_frame)

        self.resume_btn = tb.Button(self.control_group, text="暂停", bootstyle="primary",
                                    command=self.resume_detection, width=8, state=DISABLED)
        self.resume_btn.pack(side=LEFT, padx=2)

        self.stop_btn = tb.Button(self.control_group, text="停止", bootstyle="danger",
                                  command=self.stop_detection, width=8, state=DISABLED)
        self.stop_btn.pack(side=LEFT, padx=2)

        self.replay_btn = tb.Button(self.control_group, text="重新播放", bootstyle="secondary",
                                    command=self.replay_video, width=10, state=DISABLED)
        self.replay_btn.pack_forget()

        self.control_group.pack_forget()

        right_spacer = tb.Frame(ctrl_frame, width=50)
        right_spacer.pack(side=RIGHT, fill=X, expand=True)

        self.reprocess_btn = tb.Button(ctrl_frame, text="重新检测", bootstyle="warning",
                                       command=self.reprocess, width=8, state=DISABLED)
        self.reprocess_btn.pack_forget()

        self.save_btn = tb.Button(ctrl_frame, text="保存", bootstyle="success",
                                  command=self.save_result, width=6, state=DISABLED)
        self.save_btn.pack_forget()

        # 状态信息栏
        info_frame = tb.Frame(bottom_frame, height=28)
        info_frame.pack(fill=X, pady=(0, 5), padx=5)
        self.info_label = tb.Label(info_frame, text="就绪 | 欢迎使用安全帽佩戴检测系统",
                                   font=("微软雅黑", 9))
        self.info_label.pack(anchor=W)

        # 主题切换按钮
        theme_frame = tb.Frame(bottom_frame)
        theme_frame.pack(side=RIGHT, padx=5, pady=5)
        self.theme_btn = tb.Button(theme_frame, text=" ☀️", bootstyle="secondary",
                                   command=self.toggle_theme, width=3)
        self.theme_btn.pack()

        # 绑定滑动条数值变化
        self.conf_var.trace('w', lambda *args: self.conf_label.config(text=f"{self.conf_var.get():.2f}"))
        self.iou_var.trace('w', lambda *args: self.iou_label.config(text=f"{self.iou_var.get():.2f}"))

    def toggle_theme(self):
        """切换亮色/暗色主题"""
        if self.current_theme == "lumen":
            self.current_theme = "darkly"
            self.theme_btn.config(text=" 🌙")
            self.display_frame.config(bg="#3a3a3a", highlightbackground="#5a5a5a")
            self.image_label.config(bg="#3a3a3a")
        else:
            self.current_theme = "lumen"
            self.theme_btn.config(text=" ☀️")
            self.display_frame.config(bg="#e6e6e6", highlightbackground="#b0b0b0")
            self.image_label.config(bg="#e6e6e6")
        self.root.style.theme_use(self.current_theme)

    def on_display_resize(self, event):
        """窗口尺寸变化时重新缩放当前图片"""
        if self.current_display_image is not None:
            self.display_image(self.current_display_image)

    def update_progress(self, current, total):
        """更新进度条与标签"""
        if total > 0:
            pct = (current / total) * 100
            self.progress_var.set(pct)
            self.progress_label.config(text=f"进度: {current}/{total} 帧 ({pct:.1f}%)", fg="#28a745")
        else:
            self.progress_var.set(0)
            self.progress_label.config(text="处理中...", fg="#28a745")

    def reset_progress(self):
        """重置进度显示"""
        self.progress_var.set(0)
        self.progress_label.config(text="就绪", fg="#28a745")

    def show_controls(self, mode='hide'):
        """根据模式显示/隐藏播放控制按钮"""
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
            self.resume_btn.config(state=NORMAL, text="暂停")
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

    def show_action_buttons(self, show=False):
        """显示/隐藏重新检测和保存按钮"""
        if show:
            self.reprocess_btn.pack(side=RIGHT, padx=2)
            self.reprocess_btn.config(state=NORMAL)
            self.save_btn.pack(side=RIGHT, padx=2)
            self.save_btn.config(state=NORMAL)
        else:
            self.reprocess_btn.pack_forget()
            self.reprocess_btn.config(state=DISABLED)
            self.save_btn.pack_forget()
            self.save_btn.config(state=DISABLED)

    def show_save_button(self, show=True):
        """快捷控制保存按钮显示"""
        self.show_action_buttons(show)

    def reset_tracking(self):
        """清空目标跟踪缓存"""
        self.tracked_objects = []
        self.unique_helmet = 0
        self.unique_no_helmet = 0
        self.frame_counter = 0

    #  摄像头枚举与选择 
    def enumerate_cameras(self, max_index=10):
        """扫描可用摄像头索引"""
        available = []
        for i in range(max_index):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available.append(i)
                cap.release()
            except:
                continue
        return available

    def show_camera_selection_dialog(self, available_indices):
        if not available_indices:
            messagebox.showerror("错误", "未检测到任何可用的摄像头，请检查连接。")
            return None

        dialog = Toplevel(self.root)
        dialog.title("选择摄像头")
        dialog.geometry("350x280")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.update_idletasks()
        # 修复此处
        x = (self.root.winfo_screenwidth() - dialog.winfo_width()) // 2
        y = (self.root.winfo_screenheight() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

        tb.Label(dialog, text="请选择要使用的摄像头：", font=("微软雅黑", 10)).pack(pady=15)
        listbox = Listbox(dialog, font=("微软雅黑", 9), height=5, relief=FLAT, borderwidth=0,
                          highlightthickness=1, highlightcolor="#dadce0", bg="white")
        listbox.pack(padx=20, pady=5, fill=BOTH, expand=True)
        for idx in available_indices:
            listbox.insert(END, f"摄像头 {idx}")

        selected_index = None
        def on_select():
            nonlocal selected_index
            selection = listbox.curselection()
            if selection:
                selected_index = available_indices[selection[0]]
                dialog.destroy()
            else:
                messagebox.showwarning("提示", "请先选择一个摄像头")
        def on_cancel():
            nonlocal selected_index
            selected_index = None
            dialog.destroy()

        btn_frame = tb.Frame(dialog)
        btn_frame.pack(pady=15)
        tb.Button(btn_frame, text="确定", bootstyle="primary", command=on_select, width=8).pack(side=LEFT, padx=5)
        tb.Button(btn_frame, text="取消", bootstyle="secondary", command=on_cancel, width=8).pack(side=LEFT, padx=5)

        self.root.wait_window(dialog)
        return selected_index

    #  模型加载 
    def load_model(self):
        """异步加载YOLO模型"""
        try:
            self.device = select_device('')
            self.model = attempt_load(self.weights_path, device=self.device)
            self.stride = int(self.model.stride.max())
            self.model.names = ['no_helmet', 'helmet']
            self.names = self.model.names
            dummy = torch.zeros(1, 3, self.img_size, self.img_size).to(self.device)
            self.model(dummy)
            self.root.after(0, lambda: self.model_status.config(text="模型已加载", bootstyle="success"))
            self.root.after(0, lambda: self.status_var.set("就绪"))
            self.root.after(0, lambda: self.info_label.config(text="模型加载成功 | 类别: no_helmet, helmet"))
        except Exception as e:
            self.root.after(0, lambda: self.model_status.config(text="加载失败", bootstyle="danger"))
            self.root.after(0, lambda: self.status_var.set(f"错误: {str(e)[:50]}"))
            messagebox.showerror("模型加载失败", f"请检查权重文件:\n{self.weights_path}\n\n错误: {e}")

    #  目标跟踪辅助 
    def match_object(self, cx, cy, frame_num):
        """在已跟踪对象中寻找最近匹配"""
        for i, (tx, ty, last_frame, cls) in enumerate(self.tracked_objects):
            dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if dist < self.tracking_threshold:
                self.tracked_objects[i] = (cx, cy, frame_num, cls)
                return i
        return None

    def prune_old_objects(self, frame_num, max_age=50):
        """移除长时间未更新的跟踪对象"""
        self.tracked_objects = [obj for obj in self.tracked_objects if frame_num - obj[2] < max_age]

    #  核心检测与绘制 
    def process_frame(self, frame, frame_num=None):
        """对单帧图像进行安全帽检测并绘制结果"""
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
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    # 标签背景位置自适应
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
                    # 计数
                    if c == 0:
                        no_helmet_cnt += 1
                        self.root.after(0, lambda conf=conf, f=frame.copy(): self.add_alert(conf, f))
                    else:
                        helmet_cnt += 1
                    # 帧跟踪更新
                    if frame_num is not None:
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        if self.match_object(cx, cy, frame_num) is None:
                            if c == 0:
                                self.unique_no_helmet += 1
                            else:
                                self.unique_helmet += 1
                            self.tracked_objects.append((cx, cy, frame_num, c))
                # 更新统计显示
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
        """显示单张图片检测统计"""
        total = helmet + no_helmet
        stats = f"检测人数: {total}\n佩戴安全帽: {helmet}\n未佩戴安全帽: {no_helmet}\n"
        stats += f"佩戴率: {helmet / total * 100:.1f}%" if total > 0 else "佩戴率: 0.0%"
        self.stats_text.config(state=NORMAL)
        self.stats_text.delete(1.0, END)
        self.stats_text.insert(END, stats)
        self.stats_text.config(state=DISABLED)

    def update_stats_tracked(self):
        """显示视频/摄像头累计跟踪统计"""
        total = self.unique_helmet + self.unique_no_helmet
        stats = f"人员总数: {total}\n佩戴安全帽: {self.unique_helmet}\n未佩戴安全帽: {self.unique_no_helmet}\n"
        stats += f"佩戴率: {self.unique_helmet / total * 100:.1f}%" if total > 0 else "佩戴率: 0.0%"
        self.stats_text.config(state=NORMAL)
        self.stats_text.delete(1.0, END)
        self.stats_text.insert(END, stats)
        self.stats_text.config(state=DISABLED)

    def add_alert(self, conf, frame=None):
        """记录并提示未戴安全帽告警"""
        timestamp_full = time.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_short = time.strftime("%H:%M")
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
        # 摄像头模式下限时弹窗告警
        if self.current_mode == 'camera':
            current_time = time.time()
            if current_time - self._last_alert_time < 3:
                return
            self._last_alert_time = current_time
            messagebox.showwarning("安全告警",
                                   f"检测到未佩戴安全帽！\n时间: {timestamp_full}\n置信度: {conf:.2f}\n截图: {screenshot_path}\n请立即处理！",
                                   parent=self.root)

    #  检测任务入口 
    def detect_image(self):
        """选择并检测单张图片"""
        if self.model is None:
            messagebox.showwarning("提示", "模型正在加载中...")
            return
        file_path = filedialog.askopenfilename(title="选择图片",
                                               filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif")])
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
        self.last_image_path = file_path
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
            self.current_display_image = result_img
            self.display_image(result_img)
            self.info_label.config(text=f"检测完成: {os.path.basename(file_path)}")
            self.show_save_button(True)
            self.progress_var.set(100)
            self.progress_label.config(text="检测完成", fg="#28a745")
        else:
            self.info_label.config(text="检测失败，请重试")
        self.progress_slider.pack_forget()

    def detect_video(self):
        """选择并开始检测视频文件"""
        if self.model is None:
            messagebox.showwarning("提示", "模型正在加载中...")
            return
        file_path = filedialog.askopenfilename(title="选择视频",
                                               filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv")])
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
        self.seek_frame = -1
        self.current_frame_num = 0
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
        self.progress_slider.pack(in_=self.progress_frame, fill=X)
        self.progress_label.config(text="就绪", fg="#28a745")
        self.show_controls('normal')
        self.info_label.config(text=f"正在播放: {os.path.basename(file_path)} ({self.total_frames}帧)")
        threading.Thread(target=self.play_video, daemon=True).start()

    def play_video(self):
        """视频播放与逐帧检测线程"""
        try:
            while self.is_playing and self.video_cap is not None:
                if self.seek_frame >= 0:
                    self.perform_seek()
                    if self.is_paused:
                        time.sleep(0.1)
                        continue
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                self.current_frame_num += 1
                self.frame_counter += 1
                result_frame = self.process_frame(frame, frame_num=self.frame_counter)
                if result_frame is not None:
                    # 缓存已处理帧
                    if self.current_frame_num > len(self.last_video_frames):
                        self.last_video_frames.append(result_frame.copy())
                    else:
                        self.last_video_frames[self.current_frame_num-1] = result_frame.copy()
                    self.current_display_image = result_frame
                    self.root.after(0, lambda f=result_frame: self.display_image(f))
                    self.root.after(0, lambda: self.info_label.config(
                        text=f"处理帧: {self.current_frame_num}/{self.total_frames}"))
                    self.root.after(0, lambda: self.update_progress(self.current_frame_num, self.total_frames))
                time.sleep(1 / self.video_fps if self.video_fps > 0 else 0.03)
            if self.current_display_image is not None:
                self.root.after(0, lambda: self.display_image(self.current_display_image))
            self.root.after(0, lambda: self.info_label.config(text="视频检测完成"))
            self.root.after(0, lambda: self.progress_label.config(text="检测完成", fg="#28a745"))
            self.root.after(0, lambda: self.show_controls('hide'))
            if len(self.last_video_frames) > 0:
                self.root.after(0, lambda: self.show_save_button(True))
            else:
                self.root.after(0, lambda: self.show_save_button(False))
        finally:
            self.cleanup_video(keep_last_frame=True)

    def toggle_camera(self):
        """开关摄像头实时检测"""
        if self.model is None:
            messagebox.showwarning("提示", "模型正在加载中...")
            return
        if self.is_camera:
            self.stop_detection()
            return
        available = self.enumerate_cameras(max_index=10)
        if not available:
            messagebox.showerror("错误", "未检测到任何可用摄像头，请检查连接。")
            return
        selected_idx = self.show_camera_selection_dialog(available)
        if selected_idx is None:
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
        cap = cv2.VideoCapture(selected_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(selected_idx)
        if not cap.isOpened():
            messagebox.showerror("错误", f"无法打开摄像头 {selected_idx}，请检查设备是否被占用。")
            return
        # 自动设置较高分辨率
        resolutions = [(3840,2160),(2560,1440),(1920,1080),(1920,1440),
                       (1600,1200),(1280,720),(1024,768),(800,600)]
        set_res = None
        for w, h in resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if abs(actual_w - w) / w < 0.05 and abs(actual_h - h) / h < 0.05:
                set_res = (actual_w, actual_h)
                break
        if set_res is None:
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            set_res = (actual_w, actual_h)
        print(f"摄像头最终分辨率: {set_res[0]} x {set_res[1]}")
        self.video_cap = cap
        self.is_playing = True
        self.is_paused = False
        self.is_camera = True
        self.total_frames = 0
        self.show_controls('normal')
        self.cam_btn.config(text="关闭摄像头", bootstyle="danger")
        self.info_label.config(text=f"摄像头实时检测中 ({set_res[0]:.0f}x{set_res[1]:.0f})")
        self.progress_slider.pack_forget()
        self.progress_label.config(text="实时预览", fg="#28a745")
        threading.Thread(target=self.process_camera, daemon=True).start()

    def process_camera(self):
        """摄像头实时处理线程"""
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
                    self.current_display_image = result_frame
                    self.root.after(0, lambda f=result_frame: self.display_image(f))
                time.sleep(0.03)
            self.root.after(0, lambda: self.show_controls('hide'))
            self.root.after(0, lambda: self.cam_btn.config(text="摄像头实时检测", bootstyle="success"))
        finally:
            self.cleanup_video(keep_last_frame=False)

    #  播放控制 
    def on_slider_release(self, event):
        """进度条拖动释放：跳转到指定帧"""
        if self.current_mode != 'video' or self.total_frames <= 0 or self.video_cap is None:
            return
        pct = self.progress_var.get()
        target_frame = int(pct / 100 * self.total_frames)
        if target_frame >= self.total_frames:
            target_frame = self.total_frames - 1
        if target_frame < 0:
            target_frame = 0
        self.seek_frame = target_frame
        self.is_paused = True
        self.resume_btn.config(text="继续")
        self.info_label.config(text=f"跳转到第 {target_frame+1} 帧...")

    def perform_seek(self):
        """执行跳帧并重新检测该帧"""
        if self.seek_frame < 0 or self.video_cap is None:
            return
        target = self.seek_frame
        self.seek_frame = -1
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self.current_frame_num = target
        ret, frame = self.video_cap.read()
        if ret:
            self.frame_counter += 1
            result_frame = self.process_frame(frame, frame_num=self.frame_counter)
            if result_frame is not None:
                if target >= len(self.last_video_frames):
                    for _ in range(len(self.last_video_frames), target+1):
                        self.last_video_frames.append(None)
                self.last_video_frames[target] = result_frame.copy()
                self.current_display_image = result_frame
                self.root.after(0, lambda: self.display_image(result_frame))
                self.root.after(0, lambda: self.update_progress(target+1, self.total_frames))
                self.root.after(0, lambda: self.info_label.config(text=f"跳转并检测第 {target+1} 帧"))
        else:
            self.root.after(0, lambda: self.info_label.config(text="跳转失败，无法读取该帧"))
        self.root.after(0, lambda: self.progress_var.set((target / self.total_frames) * 100))

    def resume_detection(self):
        """暂停/继续检测"""
        if self.is_playing:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.resume_btn.config(text="继续")
                self.info_label.config(text="已暂停")
            else:
                self.resume_btn.config(text="暂停")
                self.info_label.config(text="继续检测中...")

    def stop_detection(self):
        """停止当前检测任务并清理资源"""
        self.is_playing = False
        self.is_paused = False
        if self.is_camera:
            self.is_camera = False
            self.cam_btn.config(text="摄像头实时检测", bootstyle="success")
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None
            self.show_controls('hide')
            self.show_save_button(False)
            self.info_label.config(text="已停止检测")
            self.reset_tracking()
            self.display_default_message()
            return
        if self.current_mode == 'video' and self.current_video_path:
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None
            self.show_controls('replay')
            self.info_label.config(text="已停止，点击重新播放从头开始")
            if len(self.last_video_frames) > 0:
                self.show_save_button(True)
            if self.current_display_image is not None:
                self.root.after(0, lambda: self.display_image(self.current_display_image))
        else:
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None
            self.show_controls('hide')
            self.show_save_button(False)
            self.info_label.config(text="已停止检测")
            self.display_default_message()
        self.reset_tracking()
        self.seek_frame = -1
        self.progress_slider.pack_forget()
        self.progress_label.config(text="就绪", fg="#28a745")

    def display_default_message(self):
        """恢复显示区域默认提示文字"""
        self.current_display_image = None
        self.image_label.config(image="", text="请选择检测源",
                                font=("微软雅黑", 24, "bold"), fg="#9aa0a6", justify=CENTER)

    def replay_video(self):
        """重新播放当前视频"""
        if not self.current_video_path:
            messagebox.showwarning("提示", "没有可重新播放的视频")
            return
        self.last_video_frames = []
        self.reset_tracking()
        self.alert_listbox.delete(0, END)
        self.reset_progress()
        self.show_save_button(False)
        self.seek_frame = -1
        self.current_frame_num = 0
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
        self.progress_slider.pack(in_=self.progress_frame, fill=X)
        self.progress_label.config(text="就绪", fg="#28a745")
        self.show_controls('normal')
        self.info_label.config(text=f"重新播放: {os.path.basename(self.current_video_path)} ({self.total_frames}帧)")
        threading.Thread(target=self.play_video, daemon=True).start()

    def cleanup_video(self, keep_last_frame=False):
        """释放视频资源并更新界面状态"""
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.is_playing = False
        self.root.after(0, lambda: self.progress_label.config(text="播放结束", fg="#28a745"))
        if self.current_mode == 'video':
            self.root.after(0, lambda: self.progress_slider.pack_forget())
            if not keep_last_frame:
                self.display_default_message()

    #  重新检测与保存 
    def reprocess(self):
        """根据当前模式重新执行检测"""
        if self.current_mode == 'image':
            if self.last_image_path is None:
                messagebox.showwarning("提示", "没有可重新检测的图片")
                return
            self.detect_image_from_path(self.last_image_path)
        elif self.current_mode == 'video':
            self.replay_video()
        else:
            messagebox.showinfo("提示", "当前模式下没有可重新检测的内容")

    def detect_image_from_path(self, file_path):
        """直接对指定图片路径进行检测（内部重用）"""
        if self.model is None:
            messagebox.showwarning("提示", "模型正在加载中...")
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
        self.info_label.config(text="正在重新检测图片...")
        result_img = self.process_frame(img0, frame_num=None)
        if result_img is not None:
            self.last_result_img = result_img.copy()
            self.current_display_image = result_img
            self.display_image(result_img)
            self.info_label.config(text=f"重新检测完成: {os.path.basename(file_path)}")
            self.show_save_button(True)
            self.progress_var.set(100)
            self.progress_label.config(text="检测完成", fg="#28a745")
        else:
            self.info_label.config(text="重新检测失败，请重试")
        self.progress_slider.pack_forget()

    def save_result(self):
        """保存当前检测结果（图片或视频）"""
        if self.current_mode == 'image':
            if self.last_result_img is None:
                messagebox.showwarning("提示", "没有可保存的图片")
                return
            path = filedialog.asksaveasfilename(title="保存图片", defaultextension=".jpg",
                                                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
            if path:
                cv2.imwrite(path, self.last_result_img)
                messagebox.showinfo("成功", f"图片已保存至:\n{path}")
        elif self.current_mode == 'video':
            if not self.last_video_frames:
                messagebox.showwarning("提示", "没有可保存的视频")
                return
            path = filedialog.asksaveasfilename(title="保存视频", defaultextension=".mp4",
                                                filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")])
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

    #  图像显示 
    def display_image(self, img):
        """在界面显示区域绘制图像，自动缩放适配"""
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
            if display_w < 10 or display_h < 10:
                return
            scale = min(display_w / w, display_h / h) * 0.95
            if self.current_mode == 'camera' and (w >= 1280 or h >= 720):
                scale = min(display_w / w, display_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.image_label.config(image=img_tk, text="")
            self.image_label.image = img_tk
        except Exception as e:
            print(f"显示错误: {e}")

    def on_closing(self):
        """窗口关闭时停止所有检测并释放资源"""
        self.stop_detection()
        self.root.destroy()


if __name__ == "__main__":
    root = tb.Window(themename="lumen", size=(1500, 950))
    app = HelmetDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()