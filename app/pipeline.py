"""编排：音频采集 -> 断句 -> 识别 -> 问题判定 -> 大模型作答。

三条常驻线程，用队列解耦：
  audio 线程：只负责抓声音，绝不阻塞
  asr   线程：识别 + 判问题
  answer线程：调大模型。新问题到来会取消正在生成的旧回答（面试官不会等你）
事件统一塞进一个队列，由界面定时取走，避免跨线程操作 UI。
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any

from . import question as qmod
from .asr import create_transcriber
from .audio import AUDIBLE_RMS, TARGET_SR, LoopbackRecorder
from .llm import Answerer
from .segmenter import Segmenter


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._events: queue.Queue[dict] = queue.Queue()
        self._seg_q: queue.Queue = queue.Queue(maxsize=6)
        self._ask_q: queue.Queue = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._running = False

        self._recorder: LoopbackRecorder | None = None
        self._transcriber: Transcriber | None = None
        self._transcriber_lock = threading.Lock()
        self._answerer: Answerer | None = None
        self._answerer_lock = threading.Lock()
        self._prewarm_thread: threading.Thread | None = None
        self._answer_worker: threading.Thread | None = None
        self._answer_stop: threading.Event | None = None

        self._pending_text = ""
        self._pending_ts = 0.0
        self._history: list[dict] = []
        self._recent: deque[str] = deque(maxlen=8)
        self._last_level_ts = 0.0

    # ---------- 事件 ----------

    def _emit(self, event: dict) -> None:
        self._events.put(event)

    def _emit_generation(self, stop: threading.Event, event: dict) -> None:
        if self._stop is stop and not stop.is_set():
            self._emit(event)

    def drain(self) -> list[dict]:
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    # ---------- 生命周期 ----------

    @property
    def running(self) -> bool:
        return self._running

    def _ensure_answerer(self) -> Answerer:
        """拿到（必要时创建）Answerer。

        构造本身几乎不耗时——重活（import openai、建客户端、建连）全在 warmup 里，
        所以这里可以放心同步调用。
        """
        with self._answerer_lock:
            if self._answerer is None:
                self._answerer = Answerer(self.cfg)
            return self._answerer

    def ensure_transcriber(self) -> Transcriber:
        """拿到（必要时创建）识别器。

        「按住说话」也走这里：本地 whisper 加载一次要几百 MB 内存，
        为了偶尔的语音输入再开一份太浪费，优先复用采集链路上那个。
        没在监听时现建一个——讯飞是纯网络的，建了不占内存。
        """
        with self._transcriber_lock:
            if self._transcriber is None:
                self._transcriber = create_transcriber(self.cfg["asr"])
            return self._transcriber

    def prewarm(self) -> None:
        """在后台线程里把大模型链路热好。

        import openai 1.7s + 建客户端 0.7s + 首次 DNS/TLS 1.8s ≈ 4.2 秒。
        这些不能放在 UI 线程上（点「开始监听」会卡住界面），
        也不该留到第一次提问时才做（用户点完按钮要干等好几秒）。
        放后台，用户还在拖窗口、选设备的时候就已经热好了。
        """
        if self._prewarm_thread is not None and self._prewarm_thread.is_alive():
            return
        self._prewarm_thread = threading.Thread(
            target=lambda: self._ensure_answerer().warmup(),
            name="prewarm",
            daemon=True,
        )
        self._prewarm_thread.start()

    def start(self) -> None:
        if self._running:
            return
        if any(thread.is_alive() for thread in self._threads):
            self.stop()

        stop = threading.Event()
        self._stop = stop
        audio_cfg = self.cfg["audio"]
        self._recorder = LoopbackRecorder(
            device_id=str(audio_cfg.get("device") or ""),
            samplerate=int(audio_cfg.get("samplerate") or 0),
            block_ms=int(audio_cfg.get("block_ms") or 30),
            on_status=lambda text: self._emit_generation(
                stop, {"type": "status", "text": text, "state": "starting"}
            ),
        )
        self._transcriber = self.ensure_transcriber()
        # 复用已经预热好的实例。这里如果重新 new 一个，warmup 白做了。
        self._ensure_answerer()
        self._history = []
        self._running = True

        # 每次启动换一个全新的 Event，并把引用捕获进线程局部变量。
        # 否则重启时 clear() 会把上一代还没退出的线程"复活"，出现两个识别线程抢音频。
        self._threads = [
            threading.Thread(target=self._audio_loop, args=(stop,), name="audio", daemon=True),
            threading.Thread(target=self._asr_loop, args=(stop,), name="asr", daemon=True),
        ]
        # 作答线程独立于采集链路，手动提问在没点「开始监听」时也要能用
        self._ensure_answer_worker()
        self._emit({"type": "status", "text": "正在连接音频…", "state": "starting"})
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        if not self._running and not any(thread.is_alive() for thread in self._threads):
            return
        self._stop.set()
        if self._recorder:
            self._recorder.stop()
        if self._answerer:
            self._answerer.cancel()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads = []
        self._running = False
        self._emit({"type": "status", "text": "已停止", "state": "idle"})

    def shutdown(self) -> None:
        """窗口退出时停止全部后台线程。"""
        self.stop()
        if self._answer_stop:
            self._answer_stop.set()
        if self._answerer:
            self._answerer.cancel()
        if self._answer_worker:
            self._answer_worker.join(timeout=2.0)

    def _ensure_answer_worker(self) -> None:
        """保证作答线程活着。

        作答完全不依赖声卡，所以手动提问（包括还没点「开始监听」、
        或者已经点过「停止监听」的情况）都必须能用。少了这一步，
        手动提问会以 `'NoneType' object has no attribute 'stream'` 收场——
        因为 Answerer 原本只在 start() 里创建。

        每次重建都换一个全新的 Event 并捕获进线程局部变量，理由同 start()。

        注意这里**不做同步 warmup**：本方法是 UI 线程调的，而 warmup 冷启动
        要好几秒，会把界面卡死。预热统一交给 prewarm() 在后台做。
        """
        if self._answer_worker is not None and self._answer_worker.is_alive():
            return
        self._ensure_answerer()
        stop = threading.Event()
        self._answer_stop = stop
        self._answer_worker = threading.Thread(
            target=self._answer_loop, args=(stop,), name="answer", daemon=True
        )
        self._answer_worker.start()

    def _fail_generation(self, stop: threading.Event, text: str) -> None:
        """终止当前采集代；旧线程的迟到错误不得影响新一代。"""
        if self._stop is not stop:
            return
        stop.set()
        self._running = False
        if self._recorder:
            self._recorder.stop()
        self._emit({"type": "error", "text": text})

    def apply_config(self, cfg: dict) -> None:
        """配置变更：重启采集链路，并让模型缓存失效。"""
        was_running = self._running
        if was_running:
            self.stop()
        self.cfg = cfg
        # 丢掉旧的 Answerer（它缓存着旧配置的客户端和 system prompt），
        # 重建后重新预热。旧的预热线程可能还在跑，但它操作的是被丢弃的对象，无害。
        self._answerer = None
        self._prewarm_thread = None
        # 识别器同理：配置变了（比如换了引擎）必须重建，否则还用着旧的
        self._transcriber = None
        self._ensure_answerer()
        self.prewarm()
        if was_running:
            self.start()

    # ---------- 手动触发 ----------

    def ask(self, question: str) -> None:
        text = qmod.clean(question)
        if not text:
            return
        # 手动提问可能发生在没点「开始监听」的时候，先把作答线程拉起来
        self._ensure_answer_worker()
        self._emit({"type": "question", "text": text})
        self._put_latest(self._ask_q, text)

    def ask_recent(self) -> None:
        """把最近的实时字幕拼起来当问题，用于规则漏判时人工兜底。"""
        count = max(1, int(self.cfg["question"].get("recent_count", 3)))
        text = " ".join(list(self._recent)[-count:]).strip()
        if not text:
            self._emit(
                {
                    "type": "error",
                    "text": "还没有捕捉到任何语音内容——先点「开始监听」；"
                    "已有字幕的话，直接点某一句旁边的「提问此句」。",
                }
            )
            return
        self.ask(text)

    def ask_image(self, image_url: str) -> None:
        """把区域截图直接交给多模态模型，并立即生成回答。"""
        if not image_url:
            return
        prompt = (
            "这是我主动选择的截图，请直接读取并解答截图主体，不要判断它是否像问句。"
            "如果包含多个问题，优先回答最完整、最靠下的问题；"
            "如果是代码或图表，请结合截图中的具体内容作答。"
        )
        self._ensure_answer_worker()
        self._emit({"type": "question", "text": "截图提问"})
        self._put_latest(self._ask_q, (prompt, image_url))

    def clear_history(self) -> None:
        self._history = []
        self._recent.clear()
        self._pending_text = ""

    def cancel_answer(self) -> None:
        """取消正在生成和排队等待的回答。

        界面上的「清空」必须调这个：否则清空之后，那个还在流式返回的
        回答会继续往回答栏里塞增量，用户会以为清空按钮坏了。
        """
        while True:
            try:
                self._ask_q.get_nowait()
            except queue.Empty:
                break
        if self._answerer:
            self._answerer.cancel()

    # ---------- 线程体 ----------

    def _audio_loop(self, stop: threading.Event) -> None:
        a = self.cfg["audio"]
        recorder = self._recorder
        if recorder is None:
            self._fail_generation(stop, "音频采集失败：录音器未初始化")
            return
        seg = Segmenter(
            sr=TARGET_SR,
            block_ms=int(a.get("block_ms") or 30),
            noise_ratio=float(a.get("noise_ratio") or 2.6),
            min_rms=float(a.get("min_rms") or 0.0035),
            start_ms=int(a.get("start_ms") or 120),
            end_silence_ms=int(a.get("end_silence_ms") or 600),
            min_speech_ms=int(a.get("min_speech_ms") or 400),
            max_speech_ms=int(a.get("max_speech_ms") or 20000),
            preroll_ms=int(a.get("preroll_ms") or 200),
        )
        try:
            started = time.time()
            peak = 0.0
            warned = False
            for block in recorder.blocks():
                if stop.is_set():
                    break
                for sentence in seg.push(block):
                    self._put_latest(self._seg_q, sentence)

                peak = max(peak, seg.last_rms)
                now = time.time()
                if now - self._last_level_ts > 0.08:
                    self._last_level_ts = now
                    self._emit_generation(
                        stop,
                        {
                            "type": "level",
                            "value": min(1.0, seg.last_rms * 9.0),
                            "speaking": seg.speaking,
                        }
                    )
                # 跑了 12 秒还是一点声音都没有，多半是选错设备了
                if not warned and now - started > 12 and peak < AUDIBLE_RMS:
                    warned = True
                    self._emit_generation(
                        stop,
                        {
                            "type": "error",
                            "text": (
                                f"已运行 12 秒没抓到任何声音（设备：{recorder.device_name}）。"
                                "请确认腾讯会议的声音是从这个设备播出来的，"
                                "或在设置里手动指定采集设备。"
                            ),
                        }
                    )
            tail = seg.flush()
            if tail is not None:
                self._put_latest(self._seg_q, tail)
            if not stop.is_set():
                self._fail_generation(stop, "音频采集意外结束，请重新开始监听。")
        except Exception as exc:
            self._fail_generation(stop, f"音频采集失败：{exc}")

    def _asr_loop(self, stop: threading.Event) -> None:
        transcriber = self._transcriber
        if transcriber is None:
            self._fail_generation(stop, "语音识别初始化失败：识别器未初始化")
            return
        try:
            transcriber.load(
                on_status=lambda text: self._emit_generation(
                    stop, {"type": "status", "text": text, "state": "loading"}
                ),
                on_progress=lambda name, done, total: self._emit_generation(
                    stop, {"type": "download", "file": name, "done": done, "total": total}
                ),
            )
        except Exception as exc:
            self._fail_generation(stop, f"语音识别初始化失败：{exc}")
            return
        if stop.is_set():
            return
        self._emit_generation(
            stop, {"type": "status", "text": "正在监听面试官发言", "state": "running"}
        )

        while not stop.is_set():
            try:
                sentence = self._seg_q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                text = transcriber.transcribe(sentence)
            except Exception as exc:
                self._emit({"type": "error", "text": f"语音识别出错：{exc}"})
                continue
            if stop.is_set():
                break
            if not text:
                continue
            self._recent.append(text)
            self._emit_generation(stop, {"type": "transcript", "text": text})
            self._maybe_question(text)

    def _maybe_question(self, text: str) -> None:
        qcfg = self.cfg["question"]
        now = time.time()
        gap_ms = (now - self._pending_ts) * 1000 if self._pending_text else 0.0

        if self._pending_text and gap_ms <= float(qcfg.get("merge_gap_ms") or 1200):
            self._pending_text = f"{self._pending_text} {text}".strip()
        else:
            self._pending_text = text.strip()
        self._pending_ts = now

        # 面试官一句话可能很长，超出上限就只保留尾部
        if len(self._pending_text) > 300:
            self._pending_text = self._pending_text[-300:]

        if not qcfg.get("auto_answer", True):
            return
        if qmod.is_question(
            self._pending_text,
            min_chars=int(qcfg.get("min_chars") or 5),
            threshold=float(qcfg.get("threshold") or 3.0),
        ):
            pending, self._pending_text = self._pending_text, ""
            self.ask(pending)

    def _answer_loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                request = self._ask_q.get(timeout=0.3)
            except queue.Empty:
                continue
            # 队列里已有更新的问题，直接丢掉这条
            if not self._ask_q.empty():
                continue

            if isinstance(request, tuple):
                question, image_url = request
            else:
                question, image_url = request, ""
            self._emit({"type": "answer_start", "question": question})
            chunks: list[str] = []
            cancelled = False
            try:
                stream = (
                    self._answerer.stream(question, self._history, image_url=image_url)
                    if image_url
                    else self._answerer.stream(question, self._history)
                )
                for piece in stream:
                    if stop.is_set():
                        cancelled = True
                        break
                    if not self._ask_q.empty():
                        self._answerer.cancel()
                        cancelled = True
                        break
                    chunks.append(piece)
                    self._emit({"type": "answer_delta", "text": piece})
            except Exception as exc:
                self._emit({"type": "error", "text": f"生成回答失败：{exc}"})
            else:
                answer = "".join(chunks).strip()
                if answer and not cancelled:
                    self._history.append({"q": question, "a": answer})
                    keep = int(self.cfg["llm"].get("history_rounds") or 4)
                    self._history = self._history[-keep:]
            self._emit({"type": "answer_end"})

    # ---------- 工具 ----------

    @staticmethod
    def _put_latest(target: queue.Queue, item: Any) -> None:
        """队列满时丢掉最旧的，保证永远处理最新音频，不堆积。"""
        while True:
            try:
                target.put_nowait(item)
                return
            except queue.Full:
                try:
                    target.get_nowait()
                except queue.Empty:
                    return
