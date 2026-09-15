"""语音模型下载。

不用 huggingface_hub —— 它在这台机器上有两个坑：
  1. 默认走 Xet 传输协议，在 hf-mirror 上会卡死（实测 16 分钟只下了 17KB）
  2. Windows 上创建软链接失败（WinError 14007），导致小文件落地成 0 字节
改成直接 HTTP 拉文件到项目内的 models/ 目录，顺带能给用户真实的下载进度。
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# 必须存在的文件；vocabulary 各版本命名不同，单独处理
_REQUIRED = ("config.json", "model.bin", "tokenizer.json")
_VOCAB_CANDIDATES = ("vocabulary.txt", "vocabulary.json")
_EXTRA = ("preprocessor_config.json",)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) interview-copilot/0.1"


class DownloadError(RuntimeError):
    pass


def model_dir(name: str) -> Path:
    return MODELS_DIR / f"faster-whisper-{name}"


def _is_valid(directory: Path) -> bool:
    """目录里文件齐不齐、内容是不是真的（0 字节或坏 JSON 都算无效）。"""
    if not directory.is_dir():
        return False
    for filename in _REQUIRED:
        path = directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            return False
    if not any((directory / v).is_file() and (directory / v).stat().st_size > 0 for v in _VOCAB_CANDIDATES):
        return False
    try:
        json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


def _download(url: str, dst: Path, on_progress=None, label: str = "") -> None:
    """流式下载到 .part 再改名，避免半截文件被当成完整文件。"""
    part = dst.with_suffix(dst.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last_report = 0.0
        with open(part, "wb") as fh:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                now = time.time()
                if on_progress and (now - last_report > 0.4 or done == total):
                    last_report = now
                    on_progress(label, done, total)
    if part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        raise DownloadError(f"{label} 下载下来是空文件")
    os.replace(part, dst)


def ensure_model(
    name: str,
    endpoint: str = "https://hf-mirror.com",
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """确保模型在本地，返回可直接喂给 WhisperModel 的目录路径。"""
    # 用户直接给了本地目录/路径
    candidate = Path(name)
    if candidate.is_dir() or os.sep in name or "/" in name:
        if not candidate.is_dir():
            raise DownloadError(f"指定的模型目录不存在：{name}")
        return candidate

    directory = model_dir(name)
    if _is_valid(directory):
        return directory

    directory.mkdir(parents=True, exist_ok=True)
    endpoint = (endpoint or "https://hf-mirror.com").rstrip("/")
    repo = f"{endpoint}/Systran/faster-whisper-{name}/resolve/main"

    if on_status:
        on_status(f"正在下载语音模型 {name}（约 {_approx_size(name)}），只需一次…")

    # 先探 config.json，能立刻判断模型名对不对、网络通不通
    try:
        _download(f"{repo}/config.json", directory / "config.json", on_progress, "config.json")
    except urllib.error.HTTPError as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise DownloadError(
            f"下载 config.json 失败（HTTP {exc.code}）。请确认模型名「{name}」正确，"
            f"且能访问 {endpoint}。"
        ) from exc
    except Exception as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise DownloadError(f"连接 {endpoint} 失败：{exc}") from exc

    targets = list(_REQUIRED[1:]) + list(_VOCAB_CANDIDATES) + list(_EXTRA)
    for filename in targets:
        path = directory / filename
        if path.is_file() and path.stat().st_size > 0:
            continue
        try:
            _download(f"{repo}/{filename}", path, on_progress, filename)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and filename not in _REQUIRED:
                continue  # vocabulary / preprocessor 各版本可选
            raise DownloadError(f"下载 {filename} 失败（HTTP {exc.code}）") from exc
        except Exception as exc:
            raise DownloadError(f"下载 {filename} 失败：{exc}") from exc

    if not _is_valid(directory):
        raise DownloadError(f"模型下载不完整，请删除 {directory} 后重试。")
    if on_status:
        on_status(f"语音模型 {name} 已就绪")
    return directory


def _approx_size(name: str) -> str:
    return {
        "tiny": "75MB",
        "base": "145MB",
        "small": "484MB",
        "medium": "1.5GB",
        "large-v3": "3GB",
    }.get(name, "数百 MB")


def list_local() -> list[str]:
    if not MODELS_DIR.is_dir():
        return []
    return sorted(p.name.replace("faster-whisper-", "") for p in MODELS_DIR.iterdir() if _is_valid(p))
