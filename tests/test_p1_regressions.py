from __future__ import annotations

import copy
import os
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app.config import DEFAULTS
from app.pipeline import Pipeline


class _BlockingAnswerer:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.new_answered = threading.Event()
        self.cancelled = threading.Event()

    def stream(self, question: str, history=None):
        self.cancelled.clear()
        if question == "old":
            self.started.set()
            self.release.wait(timeout=2.0)
            if self.cancelled.is_set():
                return
        else:
            self.new_answered.set()
        yield question

    def cancel(self) -> None:
        self.cancelled.set()


class _BrokenRecorder:
    device_name = "broken"

    def blocks(self):
        raise RuntimeError("device lost")

    def stop(self) -> None:
        pass


class PipelineRegressionTests(unittest.TestCase):
    def test_stop_keeps_answer_worker_available(self) -> None:
        pipeline = Pipeline(copy.deepcopy(DEFAULTS))
        answerer = _BlockingAnswerer()
        pipeline._answerer = answerer
        pipeline._running = True
        pipeline._ensure_answer_worker()
        pipeline._ask_q.put("old")
        self.assertTrue(answerer.started.wait(timeout=1.0))

        worker = pipeline._answer_worker
        pipeline.stop()
        self.assertIs(pipeline._answer_worker, worker)
        self.assertTrue(worker.is_alive())

        answerer.release.set()
        pipeline.ask("new")
        self.assertTrue(answerer.new_answered.wait(timeout=1.0))
        pipeline.shutdown()
        self.assertFalse(worker.is_alive())

    def test_audio_failure_has_one_terminal_error(self) -> None:
        pipeline = Pipeline(copy.deepcopy(DEFAULTS))
        stop = threading.Event()
        pipeline._stop = stop
        pipeline._running = True
        pipeline._recorder = _BrokenRecorder()

        pipeline._audio_loop(stop)

        events = pipeline.drain()
        self.assertFalse(pipeline.running)
        self.assertTrue(stop.is_set())
        self.assertEqual([event["type"] for event in events], ["error"])
        self.assertIn("device lost", events[0]["text"])


class UiRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _wait_until(self, predicate, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def _dialog(self):
        from app.ui import SettingsDialog

        with patch("app.ui.audio.list_devices", return_value=[]):
            return SettingsDialog(copy.deepcopy(DEFAULTS))

    def test_llm_connection_test_does_not_block_gui(self) -> None:
        dialog = self._dialog()
        started = threading.Event()
        release = threading.Event()

        def ping(_answerer) -> str:
            started.set()
            release.wait(timeout=2.0)
            return "就绪"

        with patch("app.llm.Answerer.ping", ping):
            before = time.monotonic()
            dialog._test_llm()
            elapsed = time.monotonic() - before
            self.assertLess(elapsed, 0.2)
            self.assertTrue(started.wait(timeout=1.0))
            self.assertFalse(dialog.btn_test.isEnabled())
            release.set()
            self.assertTrue(self._wait_until(dialog.btn_test.isEnabled))

        self.assertIn("连接成功", dialog.lbl_test.text())
        dialog.close()

    def test_xfyun_connection_test_does_not_block_gui(self) -> None:
        dialog = self._dialog()
        started = threading.Event()
        release = threading.Event()

        def verify(_transcriber) -> str:
            started.set()
            release.wait(timeout=2.0)
            return "就绪"

        with patch("app.asr_xfyun.XfyunTranscriber.verify", verify):
            before = time.monotonic()
            dialog._test_xfyun()
            elapsed = time.monotonic() - before
            self.assertLess(elapsed, 0.2)
            self.assertTrue(started.wait(timeout=1.0))
            self.assertFalse(dialog.btn_xf_test.isEnabled())
            release.set()
            self.assertTrue(self._wait_until(dialog.btn_xf_test.isEnabled))

        self.assertIn("连接成功", dialog.lbl_xf_test.text())
        dialog.close()

    def test_hotkey_signal_runs_slot_on_gui_thread(self) -> None:
        from app.ui import OverlayWindow

        called = threading.Event()
        callback_thread: list[int] = []

        class ProbeWindow(OverlayWindow):
            def _toggle_visible(self) -> None:
                callback_thread.append(threading.get_ident())
                called.set()

        with (
            patch("app.ui.Pipeline.prewarm"),
            patch.object(ProbeWindow, "_setup_hotkeys"),
        ):
            window = ProbeWindow(copy.deepcopy(DEFAULTS))

        thread = threading.Thread(target=window._hotkey_toggle_requested.emit)
        thread.start()
        thread.join(timeout=1.0)

        self.assertTrue(self._wait_until(called.is_set))
        self.assertEqual(callback_thread, [threading.get_ident()])
        with patch("app.ui.cfgmod.save_ui_position"):
            window.close()


if __name__ == "__main__":
    unittest.main()
