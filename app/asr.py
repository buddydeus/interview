"""faster-whisper 本地语音识别封装（CPU / int8）。"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from . import model_store

# whisper 在静音或音乐片段上常见的幻觉输出，直接丢弃
_HALLUCINATIONS = (
    "谢谢观看", "谢谢大家观看", "感谢观看", "感谢收看", "谢谢收看",
    "字幕由", "字幕组", "请不吝点赞", "订阅", "打赏", "下期再见",
    "明镜与点点", "由 Amara.org", "MING PAO", "中文字幕",
    "点赞、订阅", "点讚", "轉發",
)


def _looks_like_hallucination(text: str) -> bool:
    if any(token in text for token in _HALLUCINATIONS):
        return True
    # 极低信息量的重复串，例如 "哈哈哈哈" "。。。。。"
    stripped = text.replace(" ", "")
    if len(stripped) >= 4 and len(set(stripped)) <= 2:
        return True
    return False


class Transcriber:
    """惰性加载模型；load() 可在后台线程里调，避免卡住界面。"""

    def __init__(self, cfg: dict):
        self.model_name = str(cfg.get("model") or "small")
        self.language = str(cfg.get("language") or "zh")
        self.compute_type = str(cfg.get("compute_type") or "int8")
        self.beam_size = int(cfg.get("beam_size") or 1)
        self.cpu_threads = int(cfg.get("cpu_threads") or 0)
        self.initial_prompt = str(cfg.get("initial_prompt") or "").strip()
        self.endpoint = str(cfg.get("hf_endpoint") or "https://hf-mirror.com").strip()
        self.model_path = ""
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(
        self,
        on_status: Callable[[str], None] | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            directory = model_store.ensure_model(
                self.model_name, self.endpoint, on_status, on_progress
            )
            self.model_path = str(directory)
            if on_status:
                on_status("正在加载语音模型到内存…")

            from faster_whisper import WhisperModel

            kwargs = {"device": "cpu", "compute_type": self.compute_type}
            if self.cpu_threads > 0:
                kwargs["cpu_threads"] = self.cpu_threads
            self._model = WhisperModel(self.model_path, **kwargs)

    def transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            self.load()
        if audio.size < 16000 * 0.2:  # 不足 200ms 不值得识别
            return ""

        segments, _info = self._model.transcribe(
            audio,
            language=None if self.language in ("auto", "") else self.language,
            beam_size=self.beam_size,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,  # 防止错误上下文滚雪球
            initial_prompt=self.initial_prompt or None,
            vad_filter=False,  # 上游 Segmenter 已经做过切分
            without_timestamps=True,
        )
        text = "".join(seg.text for seg in segments).strip()
        if not text or _looks_like_hallucination(text):
            return ""
        return text


# 所有识别后端都要满足这套接口：
#   load(on_status=None, on_progress=None) -> None
#   transcribe(audio: float32 16k 单声道) -> str
#   loaded: bool    model_path: str
def create_transcriber(cfg: dict):
    """按 asr.provider 选后端。

    auto（默认）—— 配了讯飞凭证就用讯飞，否则回退本地 whisper。
    这样加上凭证就自动切过去，凭证没配好也不会把应用弄成不可用。
    """
    provider = str(cfg.get("provider") or "auto").strip().lower()
    if provider in ("xfyun", "iflytek", "xunfei", "讯飞"):
        from .asr_xfyun import XfyunTranscriber

        return XfyunTranscriber(cfg)
    if provider == "auto":
        x = cfg.get("xfyun") or {}
        has_credentials = all(
            str(x.get(key) or "").strip() for key in ("app_id", "api_key", "api_secret")
        )
        if has_credentials:
            from .asr_xfyun import XfyunTranscriber

            return XfyunTranscriber(cfg)
    return Transcriber(cfg)


def provider_label(transcriber) -> str:
    """给界面显示用。"""
    from .asr_xfyun import XfyunTranscriber

    if isinstance(transcriber, XfyunTranscriber):
        return "讯飞语音听写（云端）"
    return f"本地 whisper {getattr(transcriber, 'model_name', '')}"
