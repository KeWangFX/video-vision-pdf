"""
视频画面分析 → PDF 报告。

工作流：
  1. FFmpeg 截帧（镜头检测 / 固定间隔）
  2. 视觉模型逐段分析画面内容（支持 OpenAI / Gemini / Claude / Ollama 等）
  3. 汇总生成「视频主要讲什么」
  4. 拼接为 PDF（截图 + 分析文字）

依赖：FFmpeg/ffprobe 在 PATH。
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fpdf import FPDF

from core import default_font_path
from vision_client import VisionClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FRAME_PROMPT = """你是视频内容分析助手。下面这张图片来自某段视频的某一时刻截图。
请用中文简要回答（5～10 句以内）：
1）画面里主要有什么（场景、人物、物体、界面、幻灯片等）；
2）若画面中有清晰可读的文字，请摘录关键信息；
3）这一时刻可能在表达或演示什么。

若画面模糊或黑屏，请如实说明。"""

CLIP_PROMPT = """你是视频内容分析助手。下面 {n} 张图片来自**同一段连续镜头**（时间上按先后顺序排列），对应视频片段约 {range_h}。
请把这几张图当作**同一段 clip 的整体**，用中文综合回答（8～15 句以内）：

1）这一段主要在呈现什么（场景、人物、动作、界面/幻灯片、情绪或氛围）；
2）若图中有可读文字，请摘录关键信息；
3）这一段在叙事或信息上的作用（例如：引入、过渡、举例、结论、转场等）；
4）若各帧差异较大（如快切），请说明这一组画面整体想强调什么。

若画面模糊、黑屏或信息不足，请如实说明。"""

SUMMARY_PROMPT = """下面是同一支视频**按时间顺序**得到的各段分析（每段对应一个镜头或固定时间片段）。
请综合判断并用中文回答：

【视频主要讲什么】
请用分条或分段说明：整体主题、大致结构或流程（若看得出）、关键要点与结论（若有）。
若各段信息不足以下结论，请说明「信息有限」并给出最稳妥的概括。

--- 各段分析 ---
{text}
--- 结束 ---"""

# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

_PTS_TIME_RE = re.compile(r"pts_time:([\d.]+)")
_VF_SAFE = "format=yuv420p,scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=fast_bilinear"


def _ffprobe_duration(video: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe 读取时长失败: {(r.stderr or r.stdout or '')[:400]}")
    try:
        return max(0.1, float((r.stdout or "").strip()))
    except ValueError as e:
        raise RuntimeError("无法解析视频时长") from e


def _scene_cut_times(video: Path, threshold: float) -> list[float]:
    vf = f"select='gt(scene,{threshold})',showinfo"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info",
           "-i", str(video), "-vf", vf, "-vsync", "vfr", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("未找到 ffmpeg，请安装并加入 PATH。") from e
    blob = (r.stderr or "") + "\n" + (r.stdout or "")
    times: list[float] = []
    for m in _PTS_TIME_RE.finditer(blob):
        try:
            times.append(float(m.group(1)))
        except ValueError:
            continue
    return sorted(times)


def _merge_cuts(cuts: list[float], gap: float) -> list[float]:
    if gap <= 0 or len(cuts) < 2:
        return sorted(cuts)
    cuts = sorted(cuts)
    out = [cuts[0]]
    for t in cuts[1:]:
        if t - out[-1] >= gap:
            out.append(t)
    return out


def _to_segments(cuts: list[float], dur: float) -> list[tuple[float, float]]:
    bounds = sorted(set([0.0] + [t for t in cuts if 0 < t < dur] + [dur]))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] - bounds[i] > 1e-3]


def _reduce_segments(segs: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    s: list[list[float]] = [[a, b] for a, b in segs]
    while len(s) > n and len(s) > 1:
        lens = [x[1] - x[0] for x in s]
        i = min(range(len(s)), key=lambda j: lens[j])
        if i == 0:
            s[0:2] = [[s[0][0], s[1][1]]]
        elif i == len(s) - 1:
            s[-2:] = [[s[-2][0], s[-1][1]]]
        elif lens[i - 1] <= lens[i + 1]:
            s[i - 1: i + 1] = [[s[i - 1][0], s[i][1]]]
        else:
            s[i: i + 2] = [[s[i][0], s[i + 1][1]]]
    return [(a[0], a[1]) for a in s]


def _sample_times(t0: float, t1: float, n: int) -> list[float]:
    dur = t1 - t0
    n = max(1, min(n, 6))
    if dur <= 0.2 or n == 1:
        return [t0 + dur / 2]
    margin = min(0.25, dur / 6)
    lo, hi = t0 + margin, t1 - margin
    if hi <= lo:
        return [t0 + dur / 2]
    step = (hi - lo) / max(1, n - 1)
    return [lo + i * step for i in range(n)]


def _extract_frame(video: Path, t: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-strict", "unofficial",
           "-y", "-ss", f"{t:.4f}", "-i", str(video),
           "-vf", _VF_SAFE, "-vframes", "1", "-q:v", "2", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("未找到 ffmpeg，请安装并加入 PATH。") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"截帧失败 (t={t:.2f}s): {(e.stderr or '')[:500]}") from e


def _extract_interval(
    video: Path, out_dir: Path, interval: float, max_n: int,
) -> list[tuple[float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pat = str(out_dir / "frame_%04d.jpg")
    vf = f"fps=1/{interval},{_VF_SAFE}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-strict", "unofficial",
           "-y", "-i", str(video), "-vf", vf, "-q:v", "2", pat]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("未找到 ffmpeg，请安装并加入 PATH。") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 截帧失败: {(e.stderr or '')[:800]}") from e
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("未截取到任何画面，请检查视频文件。")
    if max_n > 0:
        frames = frames[:max_n]
    return [(float(i * interval), p) for i, p in enumerate(frames)]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _clock(s: float) -> str:
    s = int(round(s))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _rlabel(t0: float, t1: float) -> str:
    return _clock(t0) if abs(t1 - t0) < 0.05 else f"{_clock(t0)} – {_clock(t1)}"


# ---------------------------------------------------------------------------
# Vision analysis (delegates to VisionClient)
# ---------------------------------------------------------------------------

def _analyze_frame(vc: VisionClient, model: str, img: Path) -> str:
    return vc.chat_vision(model, FRAME_PROMPT, [img], max_tokens=800)


def _analyze_clip(vc: VisionClient, model: str, imgs: list[Path],
                  t0: float, t1: float) -> str:
    prompt = CLIP_PROMPT.format(n=len(imgs), range_h=_rlabel(t0, t1))
    return vc.chat_vision(model, prompt, imgs, max_tokens=1200)


def _summarize(vc: VisionClient, model: str, sections: list[str]) -> str:
    body = "\n\n---\n\n".join(
        f"[片段 {i + 1}]\n{t}" for i, t in enumerate(sections) if t.strip()
    )
    return vc.chat_text(
        model, SUMMARY_PROMPT.format(text=body),
        system="你是专业的视频内容归纳助手，输出中文。",
        max_tokens=1200,
    )


def _fmt_err(exc: BaseException) -> str:
    s = str(exc)
    return f"（分析失败：{s[:480]}）"


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

def _setup_font(pdf: FPDF) -> str:
    font_file = default_font_path()
    if font_file and font_file.suffix.lower() in (".ttf", ".ttc"):
        try:
            pdf.add_font("U", "", str(font_file))
            pdf.set_font("U", size=11)
            return "U"
        except Exception as e:
            logging.warning("字体加载失败: %s", e)
    pdf.set_font("Helvetica", size=11)
    return "Helvetica"


def _safe_write(pdf: FPDF, fn: str, text: str, size: int = 11, lh: float = 6) -> None:
    """安全写入文本，自动处理超长行避免 fpdf2 崩溃。"""
    pdf.set_font(fn, size=size)
    max_w = pdf.epw
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            pdf.multi_cell(max_w, lh, line)
        except Exception:
            # 超长无空格串强制按字符截断
            while line:
                chunk = line[:80]
                line = line[80:]
                try:
                    pdf.multi_cell(max_w, lh, chunk)
                except Exception:
                    pdf.multi_cell(max_w, lh, chunk[:40] + "…")
                    break


def build_pdf(
    out_pdf: Path,
    video_name: str,
    items: list[tuple[float, float, Path, str | None]],
    summary: str | None,
    font_log: list[str],
    *,
    section_subtitle: str = "按时间截取的画面与分析",
    provider_info: str = "",
) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_margins(10, 10, 10)
    fn = _setup_font(pdf)

    pdf.add_page()
    pdf.set_font(fn, size=16)
    pdf.multi_cell(0, 10, "视频画面分析报告")
    pdf.ln(2)
    pdf.set_font(fn, size=10)
    pdf.multi_cell(0, 5, f"源文件: {video_name}")
    if provider_info:
        pdf.multi_cell(0, 5, f"分析模型: {provider_info}")
    pdf.ln(4)

    if summary:
        pdf.set_font(fn, size=14)
        pdf.multi_cell(0, 8, "一、视频主要讲什么（综合归纳）")
        pdf.ln(1)
        _safe_write(pdf, fn, summary, size=11, lh=6)
        pdf.ln(4)

    pdf.set_font(fn, size=14)
    pdf.multi_cell(0, 8, f"二、{section_subtitle}")
    pdf.ln(2)

    for idx, (t0, t1, img_path, analysis) in enumerate(items, start=1):
        pdf.add_page()
        pdf.set_font(fn, size=12)
        pdf.multi_cell(0, 7, f"{_rlabel(t0, t1)}（第 {idx} 段）")
        pdf.ln(2)
        try:
            pdf.image(str(img_path), w=min(180, pdf.epw))
        except Exception as e:
            font_log.append(f"插图失败 {img_path.name}: {e}")
            pdf.set_font(fn, size=11)
            pdf.multi_cell(0, 6, f"[无法嵌入图片: {e}]")
        pdf.ln(3)
        if analysis:
            _safe_write(pdf, fn, analysis, size=11, lh=6)
            pdf.ln(2)
        else:
            pdf.set_font(fn, size=11)
            pdf.multi_cell(0, 6, "（未启用 AI 分析）")

    pdf.output(str(out_pdf))


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def run_vision_report(
    video: Path,
    out_pdf: Path,
    *,
    mode: str = "scene",
    interval_sec: float = 30.0,
    max_frames: int = 24,
    scene_threshold: float = 0.32,
    merge_gap_sec: float = 0.12,
    max_scenes: int = 48,
    frames_per_clip: int = 3,
    no_ai: bool = False,
    provider: str = "ollama",
    model: str = "qwen3-vl:30b",
    api_key: str = "",
    api_base: str = "",
    sleep_between: float = 1.0,
) -> None:
    mode = (mode or "scene").strip().lower()
    if mode not in ("scene", "interval"):
        mode = "scene"

    interval = max(1.0, float(interval_sec))
    mf = max(0, int(max_frames))
    st = max(0.05, min(0.9, float(scene_threshold)))
    mgap = max(0.0, float(merge_gap_sec))
    max_sc = max(1, int(max_scenes))
    fpc = max(1, min(6, int(frames_per_clip)))

    vc: VisionClient | None = None
    if not no_ai:
        vc = VisionClient(provider=provider, api_key=api_key, api_base=api_base)

    provider_label = f"{provider} / {model}" if not no_ai else ""
    font_log: list[str] = []

    with tempfile.TemporaryDirectory(prefix="vframes_") as tmp:
        tmp_path = Path(tmp)

        if mode == "interval":
            logging.info("模式：固定间隔，每 %.1f 秒一帧", interval)
            frames = _extract_interval(video, tmp_path, interval, mf)
            logging.info("共 %d 张截图。", len(frames))

            analyses: list[str | None] = []
            summary: str | None = None
            sub = "按固定时间间隔截取的画面与分析"

            if no_ai or vc is None:
                analyses = [None] * len(frames)
            else:
                secs: list[str] = []
                stop = False
                for i, (t_sec, pth) in enumerate(frames):
                    logging.info("分析第 %d/%d 帧…", i + 1, len(frames))
                    if stop:
                        text = "【已跳过】前面出现不可恢复错误。"
                    else:
                        try:
                            text = _analyze_frame(vc, model, pth)
                        except Exception as e:
                            logging.exception("帧分析失败")
                            text = _fmt_err(e)
                            if "额度不足" in text:
                                stop = True
                    analyses.append(text)
                    secs.append(f"时间约 {int(t_sec)}s：\n{text}")
                    time.sleep(max(0.0, sleep_between))

                logging.info("正在生成综合总结…")
                summary = _safe_summary(vc, model, secs, stop)

            items = [(t, t, p, analyses[i]) for i, (t, p) in enumerate(frames)]

        else:  # scene
            logging.info("模式：镜头切点检测（阈值 %.2f）", st)
            dur = _ffprobe_duration(video)
            cuts = _merge_cuts(_scene_cut_times(video, st), mgap)
            segs = _to_segments(cuts, dur) or [(0.0, dur)]
            if mf > 0 and len(segs) > mf:
                segs = _reduce_segments(segs, mf)
            if len(segs) > max_sc:
                segs = _reduce_segments(segs, max_sc)
            logging.info("共 %d 个镜头片段。", len(segs))

            shot_dir = tmp_path / "shots"
            shot_dir.mkdir(parents=True, exist_ok=True)
            seg_data: list[tuple[float, float, Path, list[Path]]] = []

            for si, (t0, t1) in enumerate(segs):
                ts = _sample_times(t0, t1, fpc)
                fps: list[Path] = []
                for j, tt in enumerate(ts):
                    fp = shot_dir / f"seg_{si:03d}_{j:02d}.jpg"
                    _extract_frame(video, tt, fp)
                    fps.append(fp)
                seg_data.append((t0, t1, fps[len(fps) // 2], fps))

            analyses = []
            summary = None
            sub = "按镜头片段截取的画面与整段概括"

            if no_ai or vc is None:
                analyses = [None] * len(seg_data)
            else:
                secs = []
                stop = False
                for i, (t0, t1, _, fpaths) in enumerate(seg_data):
                    logging.info("分析镜头 %d/%d（%.2f–%.2f s）…",
                                 i + 1, len(seg_data), t0, t1)
                    if stop:
                        text = "【已跳过】前面出现不可恢复错误。"
                    else:
                        try:
                            text = _analyze_clip(vc, model, fpaths, t0, t1)
                        except Exception as e:
                            logging.exception("镜头分析失败")
                            text = _fmt_err(e)
                            if "额度不足" in text:
                                stop = True
                    analyses.append(text)
                    secs.append(f"片段 {_rlabel(t0, t1)}：\n{text}")
                    time.sleep(max(0.0, sleep_between))

                logging.info("正在生成综合总结…")
                summary = _safe_summary(vc, model, secs, stop)

            items = [
                (t0, t1, cov, analyses[i])
                for i, (t0, t1, cov, _) in enumerate(seg_data)
            ]

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        logging.info("正在写入 PDF: %s", out_pdf)
        build_pdf(
            out_pdf, video.name, items,
            None if no_ai else summary,
            font_log, section_subtitle=sub,
            provider_info=provider_label,
        )
        for line in font_log:
            logging.warning("%s", line)


def _safe_summary(vc: VisionClient, model: str, sections: list[str], stop: bool) -> str:
    if stop:
        return "【服务端错误】综合总结无法生成。"
    try:
        return _summarize(vc, model, sections)
    except Exception as e:
        logging.exception("总结失败")
        return _fmt_err(e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from vision_client import provider_ids

    ap = argparse.ArgumentParser(
        description="视频截图 + 视觉模型分析 → PDF 报告",
    )
    ap.add_argument("--video", required=True, help="输入视频路径")
    ap.add_argument("--mode", choices=("scene", "interval"), default="scene")
    ap.add_argument("--interval", type=float, default=30.0, help="间隔秒数")
    ap.add_argument("--out", default=None, help="输出 PDF（省略则与视频同名）")
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--scene-threshold", type=float, default=0.32)
    ap.add_argument("--merge-gap", type=float, default=0.12)
    ap.add_argument("--max-scenes", type=int, default=48)
    ap.add_argument("--frames-per-clip", type=int, default=3)
    ap.add_argument("--provider", choices=provider_ids(), default="ollama",
                    help="模型提供方")
    ap.add_argument("--model", default="qwen3-vl:30b", help="模型名称")
    ap.add_argument("--api-key", default="", help="API 密钥（也可用环境变量）")
    ap.add_argument("--api-base", default="", help="自定义 Base URL")
    ap.add_argument("--no-ai", action="store_true", help="仅截帧不调模型")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    video = Path(args.video).resolve()
    if not video.is_file():
        logging.error("找不到视频: %s", video)
        sys.exit(1)

    out_pdf = Path(args.out).resolve() if args.out else \
        (video.parent / f"{video.stem or 'video'}.pdf").resolve()

    try:
        run_vision_report(
            video, out_pdf,
            mode=args.mode,
            interval_sec=max(1.0, args.interval),
            max_frames=max(0, args.max_frames),
            scene_threshold=args.scene_threshold,
            merge_gap_sec=args.merge_gap,
            max_scenes=args.max_scenes,
            frames_per_clip=args.frames_per_clip,
            no_ai=args.no_ai,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
            sleep_between=args.sleep,
        )
    except RuntimeError as e:
        logging.error("%s", e)
        sys.exit(2)

    logging.info("完成: %s", out_pdf)


if __name__ == "__main__":
    main()
