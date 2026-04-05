# 视频画面分析 → PDF

将视频自动截帧，通过视觉 AI 模型分析每段画面内容，生成结构化 PDF 报告，辅助快速理解视频内容。

## 支持的模型

| 提供方 | 模型示例 | 密钥环境变量 | 费用 |
|--------|----------|-------------|------|
| **OpenAI** | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` | 付费 |
| **Google Gemini** | gemini-2.5-flash, gemini-2.5-pro | `GEMINI_API_KEY` | 免费额度 / 付费 |
| **Anthropic Claude** | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` | 付费 |
| **Ollama（本地）** | qwen3-vl, llava, minicpm-v | 无需 | 免费 |
| **OpenAI 兼容** | 自定义 | 可选 | 取决于服务 |

## 截帧模式

- **镜头模式（推荐）**：FFmpeg 自动检测镜头切换点，每段抽多帧，模型整体概括该段内容
- **间隔模式**：按固定秒数截帧，逐帧分析

## 依赖

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html)（需加入 PATH）
- [Ollama](https://ollama.com)（使用本地模型时）

## 安装

```bash
pip install -r requirements.txt
```

按需安装模型 SDK：
```bash
pip install google-genai    # 用 Gemini 时
pip install anthropic       # 用 Claude 时
```

## 使用方式

### 桌面 GUI

```bash
python app.py
```

### 命令行

```bash
# Ollama 本地
python video_vision_pdf.py --video video.mp4 --provider ollama --model qwen3-vl:30b

# OpenAI
python video_vision_pdf.py --video video.mp4 --provider openai --model gpt-4o-mini

# Gemini
python video_vision_pdf.py --video video.mp4 --provider gemini --model gemini-2.5-flash
```

输出 PDF 默认与视频同名（`video.mp4` → `video.pdf`），可用 `--out` 指定路径。

### Web 版

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。

## 项目结构

```
├── core.py              # 通用工具（字体查找）
├── vision_client.py     # 多提供方统一视觉客户端
├── video_vision_pdf.py  # 主引擎：截帧 + 分析 + PDF + CLI
├── app.py               # 桌面 GUI（tkinter）
├── server.py            # Web 后端（FastAPI）
├── static/index.html    # Web 前端
└── requirements.txt     # 依赖
```
