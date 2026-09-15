"""能量型 VAD：把连续音频流切成一句一句话。

不依赖任何模型，纯 RMS + 自适应噪声底，CPU 开销可忽略。
设计目标：宁可多切一句，也不要漏掉面试官的提问。
"""

from __future__ import annotations

from collections import deque

import numpy as np


class Segmenter:
    def __init__(
        self,
        sr: int = 16000,
        block_ms: int = 30,
        noise_ratio: float = 2.6,
        min_rms: float = 0.0035,
        start_ms: int = 120,
        end_silence_ms: int = 600,
        min_speech_ms: int = 400,
        max_speech_ms: int = 20000,
        preroll_ms: int = 200,
    ):
        self.sr = sr
        self.block_ms = block_ms
        self.noise_ratio = noise_ratio
        self.min_rms = min_rms
        self.start_blocks = max(1, int(start_ms // block_ms))
        self.end_blocks = max(1, int(end_silence_ms // block_ms))
        self.min_speech_blocks = max(1, int(min_speech_ms // block_ms))
        self.max_blocks = max(1, int(max_speech_ms // block_ms))
        self.preroll_blocks = max(1, int(preroll_ms // block_ms))

        self.noise: float | None = None
        self.last_rms = 0.0
        self._seen = 0
        self._preroll: deque[np.ndarray] = deque(maxlen=self.preroll_blocks)
        self._cur: list[np.ndarray] = []
        self._voiced_run = 0
        self._silent_run = 0
        self._active = False

    def reset(self) -> None:
        self.noise = None
        self._seen = 0
        self._preroll.clear()
        self._cur = []
        self._voiced_run = 0
        self._silent_run = 0
        self._active = False

    @property
    def speaking(self) -> bool:
        return self._active

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        """喂一块音频，返回本次新切出的完整句子（通常 0 或 1 段）。"""
        if block.size == 0:
            return []

        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
        self.last_rms = rms
        self._seen += 1

        # 头 8 块（约 240ms）先摸清环境噪声底，不判人声
        if self.noise is None:
            self.noise = max(rms, 1e-4)
        if self._seen <= 8:
            self.noise = max(self.noise, rms)
            return []

        threshold = max(self.noise * self.noise_ratio, self.min_rms)
        voiced = rms > threshold
        finished: list[np.ndarray] = []

        if not self._active:
            if voiced:
                self._voiced_run += 1
                self._cur.append(block)
                if self._voiced_run >= self.start_blocks:
                    # 正式进入说话态，把预滚缓冲补到句首
                    self._active = True
                    self._cur = list(self._preroll) + self._cur
                    self._silent_run = 0
            else:
                self._voiced_run = 0
                self._cur.clear()
                self._preroll.append(block)
                # 空闲时缓慢跟踪噪声底
                self.noise = 0.97 * self.noise + 0.03 * rms
            return finished

        # 说话中
        self._cur.append(block)
        if voiced:
            self._silent_run = 0
        else:
            self._silent_run += 1

        if self._silent_run >= self.end_blocks or len(self._cur) >= self.max_blocks:
            speech_blocks = len(self._cur) - self._silent_run
            if speech_blocks >= self.min_speech_blocks:
                # 尾部静音只留 3 块，省掉后续识别开销
                tail = max(0, self._silent_run - 3)
                body = self._cur[: len(self._cur) - tail] if tail else self._cur
                finished.append(np.concatenate(body))
            self._reset_utterance()

        return finished

    def flush(self) -> np.ndarray | None:
        """停止采集时把没说完的半句吐出来。"""
        if not self._active or not self._cur:
            self._reset_utterance()
            return None
        speech_blocks = len(self._cur) - self._silent_run
        body = np.concatenate(self._cur)
        self._reset_utterance()
        if speech_blocks < self.min_speech_blocks:
            return None
        return body

    def _reset_utterance(self) -> None:
        self._cur = []
        self._preroll.clear()
        self._active = False
        self._voiced_run = 0
        self._silent_run = 0
