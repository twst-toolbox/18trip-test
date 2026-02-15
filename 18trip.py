import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
import srt
import datetime
import threading
import os
from PIL import Image, ImageTk

class SubtitleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("18TRIP 字幕提取器 V11 (突变检测版)")
        self.root.geometry("1200x850")
        
        # 默认参数
        self.rect_d = [200, 500, 900, 150] 
        self.binary_threshold = 130 
        self.diff_threshold = 3.0 # 默认灵敏度 3.0%
        
        self.video_path = ""
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.is_processing = False
        
        self._init_ui()
        
    def _init_ui(self):
        # 1. 顶部
        frame_top = tk.Frame(self.root, pady=5)
        frame_top.pack(side=tk.TOP, fill=tk.X)
        tk.Button(frame_top, text="📂 加载视频", command=self.load_video, font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
        self.lbl_status = tk.Label(frame_top, text="准备就绪", fg="gray")
        self.lbl_status.pack(side=tk.LEFT)
        tk.Button(frame_top, text="▶️ 开始提取", command=self.start_thread, bg="#ddffdd", font=("Arial", 12, "bold")).pack(side=tk.RIGHT, padx=10)

        # 2. 主体区
        frame_main = tk.Frame(self.root)
        frame_main.pack(fill=tk.BOTH, expand=True, padx=10)
        
        self.canvas_frame = tk.Frame(frame_main, bg="black")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, bg="#222")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 右侧控制区
        frame_ctrl = tk.Frame(frame_main, width=320)
        frame_ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # 3. 亮度阈值
        lf_thresh = tk.LabelFrame(frame_ctrl, text="✨ 文字亮度门槛 (0-255)", padx=5, pady=5)
        lf_thresh.pack(fill=tk.X, pady=5)
        self.scale_thresh = tk.Scale(lf_thresh, from_=50, to=255, orient=tk.HORIZONTAL, command=self.on_thresh_change)
        self.scale_thresh.set(self.binary_threshold)
        self.scale_thresh.pack(fill=tk.X)
        self.lbl_thresh_val = tk.Label(lf_thresh, text=f"当前: {self.binary_threshold}")
        self.lbl_thresh_val.pack()

        # 4. 突变灵敏度 (V11 新增)
        lf_diff = tk.LabelFrame(frame_ctrl, text="⚡️ 切分灵敏度 (突变检测)", padx=5, pady=5)
        lf_diff.pack(fill=tk.X, pady=10)
        tk.Label(lf_diff, text="数值越小越敏感 (容易切碎)\n数值越大越迟钝 (容易连读)", fg="gray", font=("Arial", 8)).pack()
        self.scale_diff = tk.Scale(lf_diff, from_=0.5, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.on_diff_change)
        self.scale_diff.set(self.diff_threshold)
        self.scale_diff.pack(fill=tk.X)
        self.lbl_diff_val = tk.Label(lf_diff, text=f"当前: {self.diff_threshold}%")
        self.lbl_diff_val.pack()

        # 5. 绿框调整
        lf_rect = tk.LabelFrame(frame_ctrl, text="🟢 扫描区域", padx=5, pady=5)
        lf_rect.pack(fill=tk.X, pady=10)
        labels = ["X (左)", "Y (上)", "W (宽)", "H (高)"]
        self.sliders = []
        for i in range(4):
            tk.Label(lf_rect, text=labels[i], anchor="w").pack(fill=tk.X)
            scale = tk.Scale(lf_rect, from_=0, to=2000, orient=tk.HORIZONTAL, resolution=1)
            scale.set(self.rect_d[i])
            scale.pack(fill=tk.X)
            scale.config(command=lambda v, idx=i: self.on_rect_change(v, idx))
            self.sliders.append(scale)

        # 6. 底部
        frame_bottom = tk.Frame(self.root, pady=5)
        frame_bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10)
        self.scale_time = tk.Scale(frame_bottom, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_time_change, showvalue=0)
        self.scale_time.pack(fill=tk.X)
        frame_info = tk.Frame(frame_bottom)
        frame_info.pack(fill=tk.X)
        self.lbl_time_val = tk.Label(frame_info, text="00:00")
        self.lbl_time_val.pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(frame_info, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

    def on_rect_change(self, val, idx):
        self.rect_d[idx] = int(float(val))
        self.update_preview()
        
    def on_thresh_change(self, val):
        self.binary_threshold = int(val)
        self.lbl_thresh_val.config(text=f"当前: {self.binary_threshold}")
        self.update_preview()

    def on_diff_change(self, val):
        self.diff_threshold = float(val)
        self.lbl_diff_val.config(text=f"当前: {self.diff_threshold}%")

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mkv *.avi")])
        if not path: return
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale_time.config(to=self.total_frames)
        self.lbl_status.config(text=f"已加载: {os.path.basename(path)}")
        for s in self.sliders: s.config(to=max(w, h))
        self.update_preview()

    def on_time_change(self, val):
        if not self.cap: return
        self.lbl_time_val.config(text=str(datetime.timedelta(seconds=int(int(val)/self.fps))))
        self.update_preview()

    def update_preview(self):
        if not self.cap or self.is_processing: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.scale_time.get()))
        ret, frame = self.cap.read()
        if ret:
            x, y, w, h = self.rect_d
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY)
                bin_color = np.zeros_like(roi)
                bin_color[:,:,1] = binary
                mask_inv = cv2.bitwise_not(binary)
                bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                final_roi = cv2.add(bg, bin_color)
                frame[y:y+h, x:x+w] = final_roi
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            if cw > 1: img.thumbnail((cw, ch))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(cw//2, ch//2, image=self.photo, anchor=tk.CENTER)

    def start_thread(self):
        if not self.video_path: return
        self.is_processing = True
        threading.Thread(target=self.run_logic, daemon=True).start()

    def run_logic(self):
        out_srt = os.path.splitext(self.video_path)[0] + ".srt"
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        subs = []
        
        is_speaking = False
        start_f = 0
        peak_density = 0.0
        sub_idx = 1
        kernel = np.ones((3,3), np.uint8)
        
        # 记录上一帧的文字形状
        last_dilated = None
        
        # 参数
        thresh_val = self.binary_threshold
        diff_limit = self.diff_threshold / 100.0 # 转换百分比
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if idx % 100 == 0:
                self.root.after(0, lambda v=(idx/total)*100: self.progress.config(value=v))
            
            x, y, w, h = self.rect_d
            roi = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 1. 提取
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(binary, kernel, iterations=1)
            density = cv2.countNonZero(dilated) / (w * h)
            
            # 2. 计算形状突变 (Diff)
            diff_score = 0.0
            if last_dilated is not None:
                diff_img = cv2.absdiff(dilated, last_dilated)
                diff_score = cv2.countNonZero(diff_img) / (w * h)
            
            last_dilated = dilated.copy()
            
            # 3. 状态机
            if not is_speaking:
                if density > 0.005: # 启动阈值
                    is_speaking = True
                    start_f = idx
                    peak_density = density
            else:
                if density > peak_density: peak_density = density
                
                cut = False
                cut_reason = ""
                
                # 条件A：字没了
                if density < 0.002: 
                    cut = True
                    cut_reason = "empty"
                
                # 条件B：字突然变少了 (峰值回落) - 对付长句变短句
                elif density < (peak_density * 0.5) and peak_density > 0.02: 
                    cut = True
                    cut_reason = "drop"
                
                # 条件C (V11核心)：画面形状突变 - 对付短句变长句/瞬时切换
                # 只有当当前这句话持续了一小会儿(>0.2s)才检测，防止打字过程中的误判
                elif diff_score > diff_limit and (idx - start_f)/self.fps > 0.2:
                    cut = True
                    cut_reason = "diff"
                
                if cut:
                    dur = (idx - start_f) / self.fps
                    # 过滤超短噪音
                    if dur > 0.2:
                        st = datetime.timedelta(seconds=start_f/self.fps)
                        et = datetime.timedelta(seconds=idx/self.fps)
                        subs.append(srt.Subtitle(index=sub_idx, start=st, end=et, content=f"Line {sub_idx}"))
                        sub_idx += 1
                    
                    # 决定是否立即开始下一句
                    if density > 0.005:
                        is_speaking = True
                        start_f = idx
                        peak_density = density
                    else:
                        is_speaking = False
                        peak_density = 0.0
            idx += 1
            
        cap.release()
        with open(out_srt, "w", encoding="utf-8") as f: f.write(srt.compose(subs))
        
        self.is_processing = False
        self.root.after(0, lambda: messagebox.showinfo("完成", f"字幕已生成:\n{out_srt}"))

if __name__ == "__main__":
    root = tk.Tk()
    app = SubtitleApp(root)
    root.mainloop()
