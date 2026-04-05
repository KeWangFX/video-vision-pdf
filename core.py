"""通用工具：Unicode 字体自动查找（Windows / macOS / Linux）。"""

from __future__ import annotations

import os
from pathlib import Path

_FONT_CANDIDATES: tuple[str, ...] = (
    # Windows 中文字体（优先黑体，其次微软雅黑）
    "simhei.ttf",
    "msyh.ttc",
    "arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
)


def default_font_path() -> Path | None:
    """返回系统中第一个可用的 Unicode 字体路径，找不到返回 None。"""
    win_fonts = None
    windir = os.environ.get("WINDIR")
    if windir:
        win_fonts = Path(windir) / "Fonts"

    for name in _FONT_CANDIDATES:
        if os.sep in name or "/" in name:
            p = Path(name)
        elif win_fonts:
            p = win_fonts / name
        else:
            continue
        if p.is_file():
            return p
    return None
