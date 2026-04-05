"""
桌面端：单视频 → 截帧 + 视觉模型分析 → PDF。
支持 OpenAI / Google Gemini / Anthropic Claude / Ollama 本地 等多种提供方。
依赖：FFmpeg 在 PATH。
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from video_vision_pdf import run_vision_report
from vision_client import (
    PROVIDERS,
    provider_display,
    provider_env_key,
    provider_ids,
    provider_models,
)


class VideoVisionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("视频画面分析 → PDF")
        self.minsize(600, 640)
        self.resizable(True, True)

        self._video_path: str | None = None
        self._out_dir: str | None = None
        self._worker: threading.Thread | None = None

        self._build_ui()
        self._on_provider_change()

    # ---- UI construction ----

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # -- 视频 & 输出 --
        r = ttk.Frame(self); r.pack(fill=tk.X, **pad)
        ttk.Button(r, text="选择视频", command=self._pick_video).pack(side=tk.LEFT)
        self._lbl_video = ttk.Label(r, text="未选择文件")
        self._lbl_video.pack(side=tk.RIGHT)

        r = ttk.Frame(self); r.pack(fill=tk.X, **pad)
        ttk.Button(r, text="输出文件夹", command=self._pick_folder).pack(side=tk.LEFT)
        self._lbl_folder = ttk.Label(r, text="未选择")
        self._lbl_folder.pack(side=tk.RIGHT)

        # -- 截帧模式 --
        r = ttk.Frame(self); r.pack(fill=tk.X, **pad)
        ttk.Label(r, text="截帧模式:").pack(side=tk.LEFT)
        self._mode = tk.StringVar(value="scene")
        ttk.Combobox(r, textvariable=self._mode, values=("scene", "interval"),
                     state="readonly", width=22).pack(side=tk.LEFT, padx=(8, 0))

        # -- 镜头参数 --
        r = ttk.Frame(self); r.pack(fill=tk.X, **pad)
        ttk.Label(r, text="灵敏度/合并秒/镜头上限/每段帧数:").pack(side=tk.LEFT)
        self._scene_th = tk.StringVar(value="0.32")
        self._merge_gap = tk.StringVar(value="0.12")
        self._max_scenes = tk.StringVar(value="48")
        self._fpc = tk.StringVar(value="3")
        for var, w in [(self._scene_th, 6), (self._merge_gap, 6),
                       (self._max_scenes, 5), (self._fpc, 4)]:
            ttk.Entry(r, textvariable=var, width=w).pack(side=tk.LEFT, padx=2)

        r = ttk.Frame(self); r.pack(fill=tk.X, **pad)
        ttk.Label(r, text="间隔模式-截图间隔(秒):").pack(side=tk.LEFT)
        self._interval = tk.StringVar(value="30")
        ttk.Entry(r, textvariable=self._interval, width=8).pack(side=tk.LEFT, padx=(8, 0))

        r = ttk.Frame(self); r.pack(fill=tk.X, **pad)
        ttk.Label(r, text="最多段数/帧数 (0=不限):").pack(side=tk.LEFT)
        self._max_frames = tk.StringVar(value="24")
        ttk.Entry(r, textvariable=self._max_frames, width=8).pack(side=tk.LEFT, padx=(8, 0))

        # -- 分隔 --
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=6)

        # -- 提供方 --
        r = ttk.Frame(self); r.pack(fill=tk.X, **pad)
        ttk.Label(r, text="模型提供方:").pack(side=tk.LEFT)
        prov_display = [f"{pid} — {provider_display(pid)}" for pid in provider_ids()]
        self._provider_var = tk.StringVar(value=prov_display[3])  # ollama
        self._prov_combo = ttk.Combobox(
            r, textvariable=self._provider_var, values=prov_display,
            state="readonly", width=32)
        self._prov_combo.pack(side=tk.LEFT, padx=(8, 0))
        self._prov_combo.bind("<<ComboboxSelected>>", lambda _: self._on_provider_change())

        # -- API Key --
        r = ttk.Frame(self); r.pack(fill=tk.X, **pad)
        ttk.Label(r, text="API Key:").pack(side=tk.LEFT)
        self._api_key = tk.StringVar()
        self._entry_key = ttk.Entry(r, textvariable=self._api_key, width=56, show="*")
        self._entry_key.pack(side=tk.LEFT, padx=(8, 0))
        self._lbl_key_hint = ttk.Label(r, text="", foreground="gray")
        self._lbl_key_hint.pack(side=tk.LEFT, padx=(6, 0))

        # -- Base URL --
        r = ttk.Frame(self); r.pack(fill=tk.X, **pad)
        ttk.Label(r, text="Base URL:").pack(side=tk.LEFT)
        self._api_base = tk.StringVar(value="http://127.0.0.1:11434/v1")
        self._entry_base = ttk.Entry(r, textvariable=self._api_base, width=56)
        self._entry_base.pack(side=tk.LEFT, padx=(8, 0))

        # -- 模型 --
        r = ttk.Frame(self); r.pack(fill=tk.X, **pad)
        ttk.Label(r, text="模型名称:").pack(side=tk.LEFT)
        self._model = tk.StringVar(value="qwen3-vl:30b")
        self._model_combo = ttk.Combobox(r, textvariable=self._model, width=32)
        self._model_combo.pack(side=tk.LEFT, padx=(8, 0))

        # -- 日志 --
        log_f = ttk.Frame(self)
        log_f.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 4))
        self._log = tk.Text(log_f, height=12, wrap=tk.WORD, state=tk.DISABLED)
        scroll = ttk.Scrollbar(log_f, command=self._log.yview)
        self._log.configure(yscrollcommand=scroll.set)
        self._log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._btn = ttk.Button(self, text="生成 PDF", command=self._on_start)
        self._btn.pack(pady=8)

    # ---- Dynamic UI ----

    def _current_provider_id(self) -> str:
        raw = self._provider_var.get()
        return raw.split(" — ")[0].strip() if " — " in raw else raw.strip()

    def _on_provider_change(self) -> None:
        pid = self._current_provider_id()
        models = provider_models(pid)
        self._model_combo.configure(values=models)
        cur = self._model.get().strip()
        if cur not in models and models:
            self._model.set(models[0])

        env = provider_env_key(pid)
        needs_key = pid in ("openai", "gemini", "claude")
        needs_base = pid in ("ollama", "compatible")

        if env:
            self._lbl_key_hint.configure(text=f"或环境变量 {env}")
        else:
            self._lbl_key_hint.configure(text="(无需密钥)" if not needs_key else "")

        self._entry_key.configure(state=tk.NORMAL if needs_key else tk.DISABLED)
        self._entry_base.configure(state=tk.NORMAL if needs_base else tk.DISABLED)

    # ---- Actions ----

    def _append_log(self, line: str) -> None:
        def do() -> None:
            self._log.configure(state=tk.NORMAL)
            self._log.insert(tk.END, line + "\n")
            self._log.see(tk.END)
            self._log.configure(state=tk.DISABLED)
        self.after(0, do)

    def _set_busy(self, busy: bool) -> None:
        self._btn.configure(state=tk.DISABLED if busy else tk.NORMAL)

    def _pick_video(self) -> None:
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("Video", "*.mp4 *.m4v *.webm *.mov *.mkv"), ("All", "*.*")])
        if path:
            self._video_path = path
            self._lbl_video.configure(text=Path(path).name)

    def _pick_folder(self) -> None:
        d = filedialog.askdirectory(title="选择输出文件夹")
        if d:
            self._out_dir = d
            short = d if len(d) < 48 else "..." + d[-44:]
            self._lbl_folder.configure(text=short)

    def _on_start(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("提示", "正在处理中。")
            return
        if not self._video_path:
            messagebox.showwarning("提示", "请先选择视频文件。")
            return
        if not self._out_dir:
            messagebox.showwarning("提示", "请选择输出文件夹。")
            return

        mode = (self._mode.get() or "scene").strip().lower()
        if mode not in ("scene", "interval"):
            mode = "scene"

        try:
            interval = max(1.0, float(self._interval.get().strip() or "30"))
        except ValueError:
            messagebox.showerror("参数", "截图间隔必须是数字。"); return
        try:
            max_frames = max(0, int(self._max_frames.get().strip() or "24"))
        except ValueError:
            messagebox.showerror("参数", "最多段数/帧数必须是整数。"); return
        try:
            scene_th = max(0.05, min(0.9, float(self._scene_th.get().strip() or "0.32")))
            merge_gap = max(0.0, float(self._merge_gap.get().strip() or "0"))
            max_scenes = max(1, int(self._max_scenes.get().strip() or "48"))
            fpc = max(1, min(6, int(self._fpc.get().strip() or "3")))
        except ValueError:
            messagebox.showerror("参数", "镜头参数格式错误。"); return

        pid = self._current_provider_id()
        api_key = self._api_key.get().strip()
        api_base = self._api_base.get().strip()
        model = (self._model.get() or "").strip()
        if not model:
            messagebox.showwarning("提示", "请选择或输入模型名称。"); return

        vid = Path(self._video_path)
        out_pdf = Path(self._out_dir) / f"{vid.stem or 'video'}.pdf"

        self._set_busy(True)

        def run() -> None:
            try:
                self._append_log(f"开始: {vid.name} → {out_pdf.name}")
                self._append_log(f"提供方: {provider_display(pid)} | 模型: {model}")
                run_vision_report(
                    vid, out_pdf,
                    mode=mode, interval_sec=interval, max_frames=max_frames,
                    scene_threshold=scene_th, merge_gap_sec=merge_gap,
                    max_scenes=max_scenes, frames_per_clip=fpc,
                    no_ai=False, provider=pid, model=model,
                    api_key=api_key, api_base=api_base,
                )
                self._append_log(f"完成: {out_pdf}")
                self.after(0, lambda: messagebox.showinfo("完成", f"已保存:\n{out_pdf}"))
            except Exception as e:
                self._append_log(f"错误: {e}")
                self.after(0, lambda: messagebox.showerror("错误", str(e)))
            finally:
                self.after(0, lambda: self._set_busy(False))

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()


def main() -> None:
    VideoVisionApp().mainloop()


if __name__ == "__main__":
    main()
