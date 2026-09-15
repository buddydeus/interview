"""「按住说话」语音输入：录麦克风 -> 识别 -> 填回输入框。

和主链路的区别：主链路采 **loopback**（系统输出，也就是面试官的声音），
这里采 **麦克风**（你自己的声音）。两者走不同设备，互不干扰，可以同时开。

事件通过队列交回界面（和 Pipeline 一样），避免跨线程直接操作 UI。
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np
import pythoncom

from .audio import TARGET_SR, AudioError, resample_to_16k, resolve_microphone

# 按住说话的最长时间。超过自动停：讯飞单次会话上限 58 秒，
# 而且这种情况多半是用户忘了松手，没必要真录下去。
MAX_SECONDS = 30.0

# 太短基本是误触，没有识别价值（本地 whisper 也有 200ms 的下限）
MIN_SECONDS = 0.3

# 采样率候选，和设备协商用
_CANDIDATE_RATES = (16000, 48000, 44100, 32000)

_BLOCK_MS = 30


class VoiceInput:
    """按住说话。start() 开始录，stop() 停止并识别，结果从 drain() 取。

    所有耗时操作都在后台线程里，UI 不阻塞。
    """

    def __init__(self, cfg: dict, get_transcriber: Callable[[], object]):
        self.cfg = cfg
        self._get_transcriber = get_transcriber
        self._events: queue.Queue[dict] = queue.Queue()
        self._stop: threading.Event | None = None
        self._worker: threading.Thread | None = None
        self._listening = False

    # ---------- 事件 ----------

    def _emit(self, event: dict) -> None:
        self._events.put(event)

    def drain(self) -> list[dict]:
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    # ---------- 状态 ----------

    @property
    def listening(self) -> bool:
        """正在录。"""
        return self._listening

    @property
    def busy(self) -> bool:
        """还在录或还在识别。"""
        return self._worker is not None and self._worker.is_alive()

    # ---------- 控制 ----------

    def start(self) -> bool:
        """开始录音。已经在忙就返回 False（不重复起线程）。"""
        if self.busy:
            return False
        stop = threading.Event()
        self._stop = stop
        self._listening = True
        self._worker = threading.Thread(
            target=self._run, args=(stop,), name="voice-input", daemon=True
        )
        self._worker.start()
        return True

    def stop(self) -> None:
        """松开按钮：停止录音，线程会接着做识别。

        立刻把 listening 置 False（不等线程自己收尾）——调用方松开手之后
        就该认为"不录了"，否则界面要等一个采集块的时间才知道状态变了。
        """
        self._listening = False
        if self._stop is not None:
            self._stop.set()

    # ---------- 线程体 ----------

    def _run(self, stop: threading.Event) -> None:
        # WASAPI 基于 COM，且 COM 初始化只对当前线程有效。每次录音都会创建
        # 新线程，因此每个工作线程都必须独立初始化并成对释放 COM。
        pythoncom.CoInitialize()
        try:
            audio = np.zeros(0, dtype=np.float32)
            try:
                audio = self._record(stop)
            except AudioError as exc:
                self._emit({"type": "error", "text": str(exc)})
            except Exception as exc:
                self._emit({"type": "error", "text": f"录音失败：{exc}"})
            finally:
                self._listening = False

            seconds = len(audio) / TARGET_SR
            if seconds < MIN_SECONDS:
                self._emit(
                    {
                        "type": "error",
                        "text": "按住说话的时间太短了——按住别放，说完再松开。",
                    }
                )
                self._emit({"type": "done"})
                return

            self._emit({"type": "status", "text": f"正在识别 {seconds:.1f} 秒语音…"})
            try:
                transcriber = self._get_transcriber()
                text = transcriber.transcribe(audio)
            except Exception as exc:
                self._emit({"type": "error", "text": f"语音识别失败：{exc}"})
                self._emit({"type": "done"})
                return

            text = (text or "").strip()
            if text:
                self._emit({"type": "text", "text": text})
            else:
                self._emit({"type": "error", "text": "没听清，再说一次试试。"})
            self._emit({"type": "done"})
        finally:
            pythoncom.CoUninitialize()

    def _record(self, stop: threading.Event) -> np.ndarray:
        mic = resolve_microphone(str(self.cfg["audio"].get("mic_device") or ""))
        self._emit({"type": "status", "text": f"正在听…（{mic.name}）说完松开"})

        last_err: Exception | None = None
        for sr in _CANDIDATE_RATES:
            if stop.is_set():
                return np.zeros(0, dtype=np.float32)
            frames = max(1, int(sr * _BLOCK_MS / 1000))
            try:
                chunks: list[np.ndarray] = []
                with mic.recorder(samplerate=sr, channels=1, blocksize=frames) as recorder:
                    deadline = time.time() + MAX_SECONDS
                    while not stop.is_set() and time.time() < deadline:
                        chunk = recorder.record(numframes=frames)
                        if chunk is None or len(chunk) == 0:
                            continue
                        mono = np.asarray(chunk, dtype=np.float32)
                        if mono.ndim > 1:
                            mono = mono.mean(axis=1)
                        chunks.append(resample_to_16k(mono, sr))
                if chunks:
                    return np.concatenate(chunks).astype(np.float32)
                return np.zeros(0, dtype=np.float32)
            except Exception as exc:
                # 采样率不支持 / 设备被独占 / 麦克风被拔，换下一个候选
                if stop.is_set():
                    return np.zeros(0, dtype=np.float32)
                last_err = exc
                continue

        raise AudioError(
            f"打开麦克风失败：{last_err}。"
            "可能是被其他程序占用，或在设置里换一个输入设备。"
        )
