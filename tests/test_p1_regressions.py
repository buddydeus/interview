from __future__ import annotations

import copy
import os
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app.config import DEFAULTS
from app.pipeline import Pipeline
from app.voice_input import VoiceInput


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
    def test_ask_recent_uses_configured_subtitle_count(self) -> None:
        cfg = copy.deepcopy(DEFAULTS)
        cfg["question"]["recent_count"] = 2
        pipeline = Pipeline(cfg)
        pipeline._recent.extend(["第一句", "第二句", "第三句", "第四句"])

        with patch.object(pipeline, "ask") as ask:
            pipeline.ask_recent()

        ask.assert_called_once_with("第三句 第四句")

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


class VoiceInputRegressionTests(unittest.TestCase):
    def test_each_recording_thread_balances_com_lifecycle(self) -> None:
        class Transcriber:
            @staticmethod
            def transcribe(_audio) -> str:
                return "测试"

        voice = VoiceInput(copy.deepcopy(DEFAULTS), lambda: Transcriber())
        samples = np.zeros(8000, dtype=np.float32)

        with (
            patch.object(voice, "_record", return_value=samples),
            patch("pythoncom.CoInitialize") as initialize,
            patch("pythoncom.CoUninitialize") as uninitialize,
        ):
            for _ in range(3):
                self.assertTrue(voice.start())
                voice._worker.join(timeout=1.0)
                self.assertFalse(voice.busy)
                self.assertIn("done", [event["type"] for event in voice.drain()])

        self.assertEqual(initialize.call_count, 3)
        self.assertEqual(uninitialize.call_count, 3)


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

    def test_hotkey_settings_are_editable(self) -> None:
        dialog = self._dialog()
        self.assertEqual(dialog.sp_recent_count.value(), 3)
        self.assertEqual(dialog.ed_hotkey_answer.text(), "n")
        self.assertEqual(dialog.ed_hotkey_clear.text(), "m")
        self.assertEqual(dialog.ed_hotkey_talk.text(), "space")
        self.assertEqual(dialog.ed_hotkey_listen.text(), "b")

        dialog.sp_recent_count.setValue(5)
        dialog.ed_hotkey_answer.setText("f6")
        dialog.ed_hotkey_clear.setText("f7")
        dialog.ed_hotkey_talk.setText("f8")
        dialog.ed_hotkey_listen.setText("f9")
        cfg = dialog.result_config()

        self.assertEqual(cfg["question"]["recent_count"], 5)
        self.assertEqual(cfg["ui"]["hotkey_answer"], "f6")
        self.assertEqual(cfg["ui"]["hotkey_clear"], "f7")
        self.assertEqual(cfg["ui"]["hotkey_talk"], "f8")
        self.assertEqual(cfg["ui"]["hotkey_listen"], "f9")
        dialog.close()

    def test_single_key_hotkeys_ignore_key_repeat(self) -> None:
        from app.ui import OverlayWindow

        callbacks = {}
        actions: list[tuple[str, int]] = []

        def hook_key(key, callback, **_kwargs):
            callbacks[key] = callback
            return object()

        class KeyEvent:
            def __init__(self, event_type: str):
                self.event_type = event_type

        class ProbeWindow(OverlayWindow):
            def _start_voice(self) -> bool:
                actions.append(("talk_start", threading.get_ident()))
                return True

            def _stop_voice(self) -> None:
                actions.append(("talk_stop", threading.get_ident()))

            def _toggle_running(self) -> None:
                actions.append(("listen", threading.get_ident()))

            def _ask_recent_from_hotkey(self) -> None:
                actions.append(("ask_recent", threading.get_ident()))

            def _clear(self) -> None:
                actions.append(("clear", threading.get_ident()))

        with (
            patch("app.ui.Pipeline.prewarm"),
            patch("keyboard.add_hotkey", return_value=object()),
            patch("keyboard.hook_key", side_effect=hook_key),
            patch("keyboard.unhook"),
        ):
            window = ProbeWindow(copy.deepcopy(DEFAULTS))

            def press_keys() -> None:
                callbacks["space"](KeyEvent("down"))
                callbacks["space"](KeyEvent("down"))
                callbacks["space"](KeyEvent("up"))
                callbacks["b"](KeyEvent("down"))
                callbacks["b"](KeyEvent("down"))
                callbacks["b"](KeyEvent("up"))
                callbacks["n"](KeyEvent("down"))
                callbacks["n"](KeyEvent("down"))
                callbacks["n"](KeyEvent("up"))
                callbacks["m"](KeyEvent("down"))
                callbacks["m"](KeyEvent("down"))
                callbacks["m"](KeyEvent("up"))

            thread = threading.Thread(target=press_keys)
            thread.start()
            thread.join(timeout=1.0)

            self.assertTrue(self._wait_until(lambda: len(actions) == 5))
            self.assertEqual([name for name, _thread_id in actions], [
                "talk_start", "talk_stop", "listen", "ask_recent", "clear"
            ])
            self.assertEqual(
                [thread_id for _name, thread_id in actions],
                [threading.get_ident()] * 5,
            )
            with patch("app.ui.cfgmod.save_ui_position"):
                window.close()

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

    def test_hotkeys_are_suspended_while_question_input_is_active(self) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest

        from app.ui import OverlayWindow

        actions: list[str] = []

        class ProbeWindow(OverlayWindow):
            def _toggle_visible(self) -> None:
                actions.append("toggle")

            def _start_voice(self) -> bool:
                actions.append("talk_start")
                return True

            def _stop_voice(self) -> None:
                actions.append("talk_stop")

            def _toggle_running(self) -> None:
                actions.append("listen")

            def _clear(self) -> None:
                actions.append("clear")

        with (
            patch("app.ui.Pipeline.prewarm"),
            patch.object(ProbeWindow, "_setup_hotkeys"),
        ):
            window = ProbeWindow(copy.deepcopy(DEFAULTS))

        window.show()
        window.activateWindow()
        window.ed_ask.setFocus()
        self.app.processEvents()
        self.assertTrue(window._hotkeys_suspended())

        with patch.object(
            window.pipeline, "ask_recent", side_effect=lambda: actions.append("ask_recent")
        ):
            window._hotkey_toggle_requested.emit()
            window._hotkey_answer_requested.emit()
            window._hotkey_clear_requested.emit()
            window._hotkey_talk_pressed.emit()
            window._hotkey_talk_released.emit()
            window._hotkey_listen_requested.emit()

            self.assertEqual(actions, [])

            QTest.keyClick(window.ed_ask, Qt.Key.Key_Escape)
            self.app.processEvents()
            self.assertFalse(window._hotkeys_suspended())

            window._hotkey_toggle_requested.emit()
            window._hotkey_answer_requested.emit()
            window._hotkey_clear_requested.emit()
            window._hotkey_talk_pressed.emit()
            window._hotkey_talk_released.emit()
            window._hotkey_listen_requested.emit()

        self.assertEqual(
            actions,
            ["toggle", "ask_recent", "clear", "talk_start", "talk_stop", "listen"],
        )
        with patch("app.ui.cfgmod.save_ui_position"):
            window.close()

    def test_transcript_follows_delayed_scroll_range_changes(self) -> None:
        from app.ui import OverlayWindow

        with (
            patch("app.ui.Pipeline.prewarm"),
            patch.object(OverlayWindow, "_setup_hotkeys"),
        ):
            window = OverlayWindow(copy.deepcopy(DEFAULTS))

        bar = window.scroll_transcript.verticalScrollBar()
        bar.setRange(0, 50)
        bar.setValue(0)
        bar.setRange(0, 100)

        self.assertEqual(bar.value(), bar.maximum())
        with patch("app.ui.cfgmod.save_ui_position"):
            window.close()


if __name__ == "__main__":
    unittest.main()
