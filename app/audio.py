"""采集系统正在播放的声音（Windows WASAPI loopback）。

腾讯会议里对方（面试官）的声音会从扬声器输出，loopback 设备能原样抓到，
不需要虚拟声卡，也不需要开立体声混音。
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Iterator

import numpy as np

TARGET_SR = 16000
_CANDIDATE_RATES = (16000, 48000, 44100, 32000)

# 判定"这个设备确实在出声"的音量门槛
AUDIBLE_RMS = 0.002


class AudioError(RuntimeError):
    pass


def _soundcard():
    try:
        import soundcard as sc
    except Exception as exc:  # pragma: no cover - 依赖缺失
        raise AudioError("缺少 soundcard 库，无法采集系统声音。请先 pip install soundcard") from exc
    return sc


def list_devices() -> list[dict]:
    """列出所有可用的采集设备，标出哪些是 loopback。"""
    sc = _soundcard()
    result = []
    for mic in sc.all_microphones(include_loopback=True):
        result.append(
            {
                "id": mic.id,
                "name": mic.name,
                "loopback": bool(getattr(mic, "isloopback", False)),
            }
        )
    return result


def resolve_device(device_id: str = ""):
    """按 id 找设备；id 为空时取默认扬声器对应的 loopback。"""
    sc = _soundcard()

    if device_id:
        for mic in sc.all_microphones(include_loopback=True):
            if mic.id == device_id or mic.name == device_id:
                return mic
        raise AudioError(f"找不到音频设备：{device_id}")

    speaker = sc.default_speaker()
    if speaker is None:
        raise AudioError("系统里找不到默认扬声器，请先确认有可用的播放设备。")

    # 官方推荐用法：拿扬声器名字去取它的 loopback 副本
    try:
        mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        if mic is not None:
            return mic
    except Exception:
        pass

    # 退化：挑名字跟默认扬声器最像的 loopback 设备
    loopbacks = [m for m in sc.all_microphones(include_loopback=True) if getattr(m, "isloopback", False)]
    for mic in loopbacks:
        if speaker.name in mic.name or mic.name.startswith(speaker.name):
            return mic
    if loopbacks:
        return loopbacks[0]
    raise AudioError("找不到 loopback 录音设备。请确认声卡驱动正常，或改用 --device 指定设备。")


def list_microphones() -> list[dict]:
    """列出真正的输入设备（不含 loopback）。

    和 list_devices() 的区别：那个是给"听面试官说话"用的（loopback，
    采的是系统输出）；这个是给"按住说话"用的（麦克风，采的是你自己的声音）。
    """
    sc = _soundcard()
    return [
        {"id": mic.id, "name": mic.name}
        for mic in sc.all_microphones(include_loopback=False)
    ]


def resolve_microphone(device_id: str = ""):
    """按 id 找麦克风；id 为空时用系统默认输入设备。"""
    sc = _soundcard()

    if device_id:
        for mic in sc.all_microphones(include_loopback=False):
            if mic.id == device_id or mic.name == device_id:
                return mic
        raise AudioError(
            f"找不到麦克风：{device_id}。请在设置里重新选一个输入设备。"
        )

    mic = sc.default_microphone()
    if mic is None:
        raise AudioError("系统里找不到默认麦克风，请在设置里指定一个输入设备。")
    return mic


def probe_loopback(duration: float = 0.4) -> list[tuple[object, float]]:
    """逐个试听 loopback 设备，返回 [(设备, 峰值音量)]，按音量降序。

    用来绕开一个很常见的坑：默认扬声器可能是显示器（HDMI/DP 音频），
    而人实际戴着耳机听，这时抓默认扬声器永远抓不到声音。
    """
    sc = _soundcard()
    results: list[tuple[object, float]] = []
    frames = 1600  # 0.1s @ 16k
    for mic in sc.all_microphones(include_loopback=True):
        if not getattr(mic, "isloopback", False):
            continue
        try:
            with mic.recorder(samplerate=TARGET_SR, channels=1, blocksize=frames) as recorder:
                peak = 0.0
                deadline = time.time() + duration
                while time.time() < deadline:
                    chunk = recorder.record(numframes=frames)
                    if chunk is None or len(chunk) == 0:
                        continue
                    mono = np.asarray(chunk, dtype=np.float32)
                    if mono.ndim > 1:
                        mono = mono.mean(axis=1)
                    peak = max(peak, float(np.sqrt(np.mean(np.square(mono)))))
                results.append((mic, peak))
        except Exception:
            continue
    results.sort(key=lambda item: item[1], reverse=True)
    return results


def resample_to_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    """重采样到 16kHz 单声道 float32。"""
    if samples.size == 0 or sr == TARGET_SR:
        return samples.astype(np.float32, copy=False)

    if sr % TARGET_SR == 0:
        # 整数倍降采样：先做块平均当抗混叠，再抽取
        factor = sr // TARGET_SR
        usable = (samples.size // factor) * factor
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        return samples[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)

    n_out = int(round(samples.size * TARGET_SR / sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src = np.linspace(0.0, samples.size - 1, n_out)
    return np.interp(src, np.arange(samples.size), samples).astype(np.float32)


class LoopbackRecorder:
    """按固定块长产出 16kHz 单声道音频块，可随时 stop()。"""

    def __init__(
        self,
        device_id: str = "",
        samplerate: int = 0,
        block_ms: int = 30,
        auto_pick: bool = True,
        on_status: Callable[[str], None] | None = None,
    ):
        self.device_id = device_id
        self.requested_sr = int(samplerate or 0)
        self.block_ms = block_ms
        self.auto_pick = auto_pick
        self.on_status = on_status
        self.actual_sr = 0
        self.device_name = ""
        self._stop = threading.Event()

    def _status(self, text: str) -> None:
        if self.on_status:
            try:
                self.on_status(text)
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()

    def _pick_device(self):
        """选定采集设备。device 留空时自动找出真正在出声的那个输出。"""
        if self.device_id:
            mic = resolve_device(self.device_id)
            self.device_name = mic.name
            return mic

        if self.auto_pick:
            self._status("正在自动识别哪个输出设备在出声…")
            try:
                ranked = probe_loopback(0.4)
            except Exception:
                ranked = []
            if ranked and ranked[0][1] > AUDIBLE_RMS:
                mic, _level = ranked[0]
                self.device_name = mic.name
                self._status(f"已锁定输出设备：{mic.name}")
                return mic
            self._status("当前没有设备在出声，暂用默认扬声器（开始播放后请留意电平条）")

        mic = resolve_device("")
        self.device_name = mic.name
        return mic

    def _rates(self) -> list[int]:
        rates = []
        if self.requested_sr:
            rates.append(self.requested_sr)
        for rate in _CANDIDATE_RATES:
            if rate not in rates:
                rates.append(rate)
        return rates

    def blocks(self) -> Iterator[np.ndarray]:
        mic = self._pick_device()
        last_err: Exception | None = None

        for sr in self._rates():
            if self._stop.is_set():
                return
            frames = max(1, int(sr * self.block_ms / 1000))
            try:
                with mic.recorder(samplerate=sr, channels=1, blocksize=frames) as recorder:
                    self.actual_sr = sr
                    while not self._stop.is_set():
                        chunk = recorder.record(numframes=frames)
                        if chunk is None or len(chunk) == 0:
                            continue
                        mono = np.asarray(chunk, dtype=np.float32)
                        if mono.ndim > 1:
                            mono = mono.mean(axis=1)
                        yield resample_to_16k(mono, sr)
                return
            except Exception as exc:  # 设备忙 / 采样率不支持 / 设备被拔
                if self._stop.is_set():
                    return
                last_err = exc
                time.sleep(0.4)
                continue

        raise AudioError(f"打开音频设备失败：{last_err}")
