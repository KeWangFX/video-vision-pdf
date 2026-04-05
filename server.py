"""
FastAPI Web 版：上传视频 → 截帧 + 视觉模型分析 → PDF 下载/预览。
支持 OpenAI / Gemini / Claude / Ollama 等多种提供方。
运行: uvicorn server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from video_vision_pdf import run_vision_report
from vision_client import provider_ids

app = FastAPI(title="Video Vision PDF")

_STATIC = Path(__file__).resolve().parent / "static"


@app.exception_handler(Exception)
async def _err(request: Request, exc: Exception) -> Response:
    if not request.url.path.startswith("/api/"):
        raise exc
    if isinstance(exc, HTTPException):
        raise exc
    logging.exception("Unhandled on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/")
async def index() -> FileResponse:
    html = _STATIC / "index.html"
    if not html.is_file():
        raise HTTPException(404, "static/index.html missing")
    return FileResponse(html, media_type="text/html; charset=utf-8")


@app.get("/api/providers")
async def get_providers() -> dict:
    from vision_client import PROVIDERS
    return {"providers": PROVIDERS}


@app.get("/api/video-vision")
async def info() -> dict:
    return {"ok": True, "service": "video-vision",
            "providers": provider_ids()}


@app.post("/api/video-vision")
@app.post("/api/video_vision")
async def video_vision(
    file: UploadFile = File(...),
    mode: str = Form("scene"),
    interval: float = Form(30.0),
    max_frames: int = Form(24),
    scene_threshold: float = Form(0.32),
    merge_gap_sec: float = Form(0.12),
    max_scenes: int = Form(48),
    frames_per_clip: int = Form(3),
    provider: str = Form("ollama"),
    model: str = Form("qwen3-vl:30b"),
    api_key: str = Form(""),
    api_base: str = Form(""),
) -> Response:
    try:
        mode = (mode or "scene").strip().lower()
        if mode not in ("scene", "interval"):
            mode = "scene"
        interval = max(1.0, float(interval or 30.0))
        max_frames = max(0, int(max_frames if max_frames is not None else 24))
        scene_threshold = float(scene_threshold if scene_threshold is not None else 0.32)
        merge_gap_sec = max(0.0, float(merge_gap_sec if merge_gap_sec is not None else 0.12))
        max_scenes = max(1, int(max_scenes if max_scenes is not None else 48))
        frames_per_clip = max(1, min(6, int(frames_per_clip if frames_per_clip is not None else 3)))
        model = (model or "qwen3-vl:30b").strip()
        prov = (provider or "ollama").strip().lower()
        api_key_clean = (api_key or "").strip()
        api_base_clean = (api_base or "").strip()

        data = await file.read()
        if not data:
            raise HTTPException(400, "上传文件为空")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            name = file.filename or "video.mp4"
            if not name.lower().endswith((".mp4", ".m4v", ".webm", ".mov", ".mkv")):
                name += ".mp4"
            vid = tmp_path / Path(name).name
            vid.write_bytes(data)
            stem = Path(name).stem or "video"
            out_pdf = tmp_path / f"{stem}.pdf"

            try:
                await run_in_threadpool(
                    run_vision_report, vid, out_pdf,
                    mode=mode, interval_sec=interval, max_frames=max_frames,
                    scene_threshold=scene_threshold, merge_gap_sec=merge_gap_sec,
                    max_scenes=max_scenes, frames_per_clip=frames_per_clip,
                    no_ai=False, provider=prov, model=model,
                    api_key=api_key_clean, api_base=api_base_clean,
                )
            except RuntimeError as e:
                raise HTTPException(400, str(e)) from e
            except Exception as e:
                logging.exception("POST /api/video-vision failed")
                return JSONResponse(status_code=500,
                                    content={"detail": f"{type(e).__name__}: {e}"})

            pdf_bytes = out_pdf.read_bytes()

        fn = f"{stem}.pdf"
        disp = "attachment; filename*=UTF-8''" + quote(fn, safe="")
        return Response(content=pdf_bytes, media_type="application/pdf",
                        headers={"Content-Disposition": disp})
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("video-vision outer failure")
        return JSONResponse(status_code=500,
                            content={"detail": f"{type(exc).__name__}: {exc}"})
