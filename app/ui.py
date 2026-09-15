"""PyQt6 悬浮置顶窗：实时字幕 + 面试官提问 + 流式回答。

关键点：用 SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) 让本窗口
在共享屏幕 / 录屏里不可见，这是面试场景的刚需。
"""

from __future__ import annotations

import copy
import sys
import threading
from string import Template

from PyQt6.QtCore import Qt, QRect, QTimer, QPoint, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QGuiApplication, QKeySequence, QShortcut, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import audio, config as cfgmod
from .pipeline import Pipeline
from .voice_input import VoiceInput

DARK = {
    "card": "rgba(17, 19, 26, 0.95)",
    "border": "rgba(255, 255, 255, 0.10)",
    "text": "#E9ECF4",
    "muted": "#8B92A6",
    "accent": "#5B8DEF",
    "accent_hover": "#7AA2F5",
    "accent_text": "#FFFFFF",
    "question": "#FFC46B",
    "panel": "rgba(255, 255, 255, 0.05)",
    "panel_border": "rgba(255, 255, 255, 0.09)",
    "ok": "#4ADE80",
    "warn": "#FBBF24",
    "err": "#F87171",
}

LIGHT = {
    "card": "rgba(252, 253, 255, 0.97)",
    "border": "rgba(15, 20, 35, 0.12)",
    "text": "#1D2331",
    "muted": "#6B7280",
    "accent": "#2F6FEB",
    "accent_hover": "#1D5FDB",
    "accent_text": "#FFFFFF",
    "question": "#A85F06",
    "panel": "rgba(15, 20, 35, 0.04)",
    "panel_border": "rgba(15, 20, 35, 0.10)",
    "ok": "#16A34A",
    "warn": "#D97706",
    "err": "#DC2626",
}

STYLE = Template(
    """
QFrame#card {
    background: $card;
    border: 1px solid $border;
    border-radius: 14px;
}
QLabel { color: $text; background: transparent; }
QLabel#title { font-size: 15px; font-weight: 600; }
QLabel#status { color: $muted; font-size: 12px; }
QLabel#section { color: $muted; font-size: 11px; font-weight: 600; letter-spacing: 1px; }
QLabel#question {
    color: $question;
    font-size: 15px;
    font-weight: 600;
    background: $panel;
    border: 1px solid $panel_border;
    border-radius: 9px;
    padding: 9px 11px;
}
QPlainTextEdit, QTextBrowser {
    background: $panel;
    border: 1px solid $panel_border;
    border-radius: 9px;
    color: $text;
    padding: 8px 10px;
    selection-background-color: $accent;
}
QTextBrowser#answer { font-size: 14px; line-height: 150%; }
QScrollArea#transcript {
    background: $panel;
    border: 1px solid $panel_border;
    border-radius: 9px;
}
QScrollArea#transcript QWidget#transcriptHost { background: transparent; }
QLabel#segtext { color: $text; }
QLabel#seghint { color: $muted; }
QPushButton#segask {
    padding: 1px 8px;
    font-size: 11px;
    border-radius: 6px;
    color: $muted;
}
QPushButton#segask:hover { border-color: $accent; color: $accent; }
QScrollArea#transcript QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 2px;
}
QScrollArea#transcript QScrollBar::handle:vertical {
    background: $panel_border;
    border-radius: 4px;
    min-height: 20px;
}
QScrollArea#transcript QScrollBar::add-line:vertical,
QScrollArea#transcript QScrollBar::sub-line:vertical { height: 0; }
QScrollArea#transcript QScrollBar::add-page:vertical,
QScrollArea#transcript QScrollBar::sub-page:vertical { background: transparent; }
QPushButton {
    background: $panel;
    border: 1px solid $panel_border;
    border-radius: 8px;
    color: $text;
    padding: 6px 12px;
    font-size: 13px;
}
QPushButton:hover { border-color: $accent; color: $accent; }
QPushButton:disabled { color: $muted; border-color: $panel_border; }
QPushButton#primary {
    background: $accent;
    border: 1px solid $accent;
    color: $accent_text;
    font-weight: 600;
}
QPushButton#primary:hover { background: $accent_hover; border-color: $accent_hover; color: $accent_text; }
QPushButton#primary[recording="true"] { background: $err; border-color: $err; color: #FFFFFF; }
QPushButton#icon { padding: 4px 9px; }
QPushButton#send {
    background: $accent;
    border: 1px solid $accent;
    color: $accent_text;
    font-weight: 600;
    padding: 5px 10px;
}
QPushButton#send:hover { background: $accent_hover; border-color: $accent_hover; color: $accent_text; }
QPushButton#send:disabled { background: $panel; border-color: $panel_border; color: $muted; }
QPushButton#talk { padding: 5px 10px; }
QPushButton#talk:hover { border-color: $accent; color: $accent; }
QPushButton#talk[recording="true"] {
    background: $err;
    border-color: $err;
    color: #FFFFFF;
    font-weight: 600;
}
QLineEdit#ask:focus { border-color: $accent; }
QProgressBar {
    background: $panel;
    border: 1px solid $panel_border;
    border-radius: 3px;
    height: 6px;
}
QProgressBar::chunk { background: $ok; border-radius: 3px; }
QDialog { background: $card; }
QTabWidget::pane { border: 1px solid $panel_border; border-radius: 8px; }
QTabBar::tab {
    background: transparent; color: $muted; padding: 7px 14px;
    border: none; font-size: 13px;
}
QTabBar::tab:selected { color: $accent; font-weight: 600; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: $panel; border: 1px solid $panel_border; border-radius: 7px;
    color: $text; padding: 5px 8px; min-height: 20px;
}
QComboBox QAbstractItemView {
    background: $card; color: $text; selection-background-color: $accent;
}
QCheckBox { color: $text; font-size: 13px; }
QSlider::groove:horizontal { height: 4px; background: $panel_border; border-radius: 2px; }
QSlider::handle:horizontal {
    background: $accent; width: 12px; margin: -5px 0; border-radius: 6px;
}
"""
)


def apply_capture_exclusion(widget: QWidget, enabled: bool) -> bool:
    """让窗口在屏幕共享/录屏中不可见。返回是否成功。"""
    if not enabled or sys.platform != "win32":
        return False
    try:
        import ctypes

        hwnd = int(widget.winId())
        user32 = ctypes.windll.user32
        # 0x11 = WDA_EXCLUDEFROMCAPTURE（Win10 2004+），0x01 = WDA_MONITOR（旧系统，截到全黑）
        for flag in (0x00000011, 0x00000001):
            if user32.SetWindowDisplayAffinity(hwnd, flag):
                return True
    except Exception:
        pass
    return False


def apply_always_on_top(widget: QWidget, enabled: bool) -> None:
    widget.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
    widget.show()


_APP_MUTEX = None


def _claim_single_instance() -> bool:
    """Windows 命名互斥量：避免多个实例同时注册并处理全局热键。"""
    if sys.platform != "win32":
        return True

    import ctypes

    global _APP_MUTEX
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, "Local\\InterviewCopilot")
    if not handle:
        return True
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle(handle)
        return False
    _APP_MUTEX = (kernel32, handle)
    return True


# --------------------------------------------------------------------------- #
# 设置对话框
# --------------------------------------------------------------------------- #


class SettingsDialog(QDialog):
    _llm_test_finished = pyqtSignal(bool, str)
    _xfyun_test_finished = pyqtSignal(bool, str)

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("设置")
        self.setMinimumWidth(520)
        self.setStyleSheet(STYLE.substitute(parent_theme(cfg)))

        tabs = QTabWidget(self)
        tabs.addTab(self._tab_llm(), "大模型")
        tabs.addTab(self._tab_asr(), "语音识别")
        tabs.addTab(self._tab_ui(), "界面")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

        self._llm_test_finished.connect(self._finish_llm_test)
        self._xfyun_test_finished.connect(self._finish_xfyun_test)

    # --- 各标签页 ---

    def _tab_llm(self) -> QWidget:
        llm = self.cfg["llm"]
        page = QWidget()
        form = QFormLayout(page)

        self.ed_key = QLineEdit(llm.get("api_key", ""))
        self.ed_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.ed_key.setPlaceholderText("sk-...  也可用环境变量 OPENAI_API_KEY")
        self.ed_base = QLineEdit(llm.get("base_url", ""))

        # Key 旁边放一个测试按钮：配完当场知道通不通，不用退出去跑命令行
        self.btn_test = QPushButton("测试连接")
        self.btn_test.clicked.connect(self._test_llm)
        key_row = QWidget()
        key_layout = QHBoxLayout(key_row)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.addWidget(self.ed_key, 1)
        key_layout.addWidget(self.btn_test)

        self.lbl_test = QLabel("")
        self.lbl_test.setWordWrap(True)
        self.cb_model = QComboBox()
        self.cb_model.setEditable(True)
        self.cb_model.addItems([
            "deepseek-chat",
            "deepseek-reasoner",
        ])
        self.cb_model.setCurrentText(str(llm.get("model", "deepseek-chat")))
        self.sp_chars = QSpinBox()
        self.sp_chars.setRange(80, 1200)
        self.sp_chars.setSingleStep(20)
        self.sp_chars.setValue(int(llm.get("max_chars", 220)))
        self.sp_rounds = QSpinBox()
        self.sp_rounds.setRange(0, 12)
        self.sp_rounds.setValue(int(llm.get("history_rounds", 4)))
        # 思考模式：deepseek-v4 这类模型默认先吐几百到几千字思维链再给正文，
        # 面试场景纯属白等。默认关闭，实测总耗时能降到三分之一左右。
        from .llm import THINKING_CHOICES

        self.cb_thinking = QComboBox()
        for value, label in THINKING_CHOICES:
            self.cb_thinking.addItem(label, value)
        mode = str(llm.get("thinking") or "off").strip().lower()
        if mode == "disabled":
            mode = "off"
        index = self.cb_thinking.findData(mode)
        self.cb_thinking.setCurrentIndex(max(0, index))
        self.cb_thinking.setToolTip(
            "关闭思考后模型直接给答案，首字和总耗时都明显更快。\n"
            "开着思考时思维链会占用「回答字数」额度，偶尔会出现正文被挤空的情况。"
        )
        self.ed_style = QPlainTextEdit(str(llm.get("style", "")))
        self.ed_style.setFixedHeight(60)
        self.ed_resume = QPlainTextEdit(str(llm.get("resume", "")))
        self.ed_resume.setPlaceholderText("粘贴你的简历或自我介绍，回答会更贴合你的真实经历")
        self.ed_resume.setFixedHeight(110)
        self.ed_jd = QPlainTextEdit(str(llm.get("jd", "")))
        self.ed_jd.setPlaceholderText("粘贴目标岗位 JD，用来对齐回答重点")
        self.ed_jd.setFixedHeight(90)

        form.addRow("API Key", key_row)
        form.addRow("接口地址", self.ed_base)
        form.addRow("", self.lbl_test)
        form.addRow("模型", self.cb_model)
        form.addRow("思考模式", self.cb_thinking)
        form.addRow("回答字数上限", self.sp_chars)
        form.addRow("携带历史轮数", self.sp_rounds)
        form.addRow("回答风格", self.ed_style)
        form.addRow("我的简历", self.ed_resume)
        form.addRow("岗位 JD", self.ed_jd)
        return page

    def _tab_asr(self) -> QWidget:
        asr = self.cfg["asr"]
        xf = asr.get("xfyun") or {}
        qcfg = self.cfg["question"]
        page = QWidget()
        form = QFormLayout(page)

        # --- 识别引擎 ---
        self.cb_provider = QComboBox()
        self.cb_provider.addItem("自动（配了讯飞凭证就用讯飞）", "auto")
        self.cb_provider.addItem("讯飞语音听写（云端）", "xfyun")
        self.cb_provider.addItem("本地 whisper（离线）", "local")
        index = self.cb_provider.findData(str(asr.get("provider") or "auto"))
        self.cb_provider.setCurrentIndex(max(0, index))

        # --- 讯飞凭证 ---
        self.ed_xf_appid = QLineEdit(str(xf.get("app_id") or ""))
        self.ed_xf_appid.setPlaceholderText("在讯飞控制台创建应用后获取")
        self.ed_xf_key = QLineEdit(str(xf.get("api_key") or ""))
        self.ed_xf_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.ed_xf_secret = QLineEdit(str(xf.get("api_secret") or ""))
        self.ed_xf_secret.setEchoMode(QLineEdit.EchoMode.Password)

        self.btn_xf_test = QPushButton("测试连接")
        self.btn_xf_test.clicked.connect(self._test_xfyun)
        xf_test_row = QWidget()
        xf_test_layout = QHBoxLayout(xf_test_row)
        xf_test_layout.setContentsMargins(0, 0, 0, 0)
        xf_test_layout.addWidget(self.btn_xf_test, 1)
        self.lbl_xf_test = QLabel("")
        self.lbl_xf_test.setWordWrap(True)
        xf_test_layout.addWidget(self.lbl_xf_test, 3)

        # --- 本地 whisper ---
        self.cb_asr = QComboBox()
        self.cb_asr.setEditable(True)
        self.cb_asr.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.cb_asr.setCurrentText(str(asr.get("model", "small")))
        self.cb_lang = QComboBox()
        self.cb_lang.addItems(["zh", "en", "auto"])
        self.cb_lang.setCurrentText(str(asr.get("language", "zh")))

        # --- 通用 ---
        self.cb_device = QComboBox()
        self._fill_devices(str(self.cfg["audio"].get("device", "")))
        btn_refresh = QPushButton("刷新设备")
        btn_refresh.clicked.connect(lambda: self._fill_devices(self.cb_device.currentData() or ""))
        device_row = QWidget()
        device_layout = QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.addWidget(self.cb_device, 1)
        device_layout.addWidget(btn_refresh)

        # 「按住说话」用的麦克风。和上面的采集设备是两回事：上面那个听的是
        # 系统输出（面试官），这个听的是你自己的声音。
        self.cb_mic = QComboBox()
        self._fill_microphones(str(self.cfg["audio"].get("mic_device", "")))
        btn_mic_refresh = QPushButton("刷新")
        btn_mic_refresh.clicked.connect(
            lambda: self._fill_microphones(self.cb_mic.currentData() or "")
        )
        mic_row = QWidget()
        mic_layout = QHBoxLayout(mic_row)
        mic_layout.setContentsMargins(0, 0, 0, 0)
        mic_layout.addWidget(self.cb_mic, 1)
        mic_layout.addWidget(btn_mic_refresh)

        self.chk_auto = QCheckBox("识别到提问就自动生成回答")
        self.chk_auto.setChecked(bool(qcfg.get("auto_answer", True)))
        self.sp_threshold = QDoubleSpinBox()
        self.sp_threshold.setRange(0.0, 12.0)
        self.sp_threshold.setSingleStep(0.5)
        self.sp_threshold.setValue(float(qcfg.get("threshold", 3.0)))
        self.sp_recent_count = QSpinBox()
        self.sp_recent_count.setRange(1, 8)
        self.sp_recent_count.setSuffix(" 条")
        self.sp_recent_count.setValue(int(qcfg.get("recent_count", 3)))
        self.sp_silence = QSpinBox()
        self.sp_silence.setRange(200, 3000)
        self.sp_silence.setSingleStep(100)
        self.sp_silence.setSuffix(" ms")
        self.sp_silence.setValue(int(self.cfg["audio"].get("end_silence_ms", 600)))

        form.addRow("识别引擎", self.cb_provider)
        form.addRow(self._section("讯飞语音听写（云端）"))
        form.addRow("APPID", self.ed_xf_appid)
        form.addRow("APIKey", self.ed_xf_key)
        form.addRow("APISecret", self.ed_xf_secret)
        form.addRow("", xf_test_row)
        form.addRow(self._section("本地 whisper（离线）"))
        form.addRow("语音模型", self.cb_asr)
        form.addRow("语言", self.cb_lang)
        form.addRow(self._section("通用"))
        form.addRow("采集设备", device_row)
        form.addRow("麦克风（按住说话）", mic_row)
        form.addRow("", self.chk_auto)
        form.addRow("提问判定阈值", self.sp_threshold)
        form.addRow("问最近字幕数", self.sp_recent_count)
        form.addRow("断句静音时长", self.sp_silence)
        return page

    @staticmethod
    def _section(text: str) -> QLabel:
        return QLabel(text, objectName="section")

    def _tab_ui(self) -> QWidget:
        ui = self.cfg["ui"]
        page = QWidget()
        form = QFormLayout(page)

        self.cb_theme = QComboBox()
        self.cb_theme.addItems(["dark", "light"])
        self.cb_theme.setCurrentText(str(ui.get("theme", "dark")))
        self.sl_opacity = QSlider(Qt.Orientation.Horizontal)
        self.sl_opacity.setRange(30, 100)
        self.sl_opacity.setValue(int(float(ui.get("opacity", 0.95)) * 100))
        self.sp_font = QSpinBox()
        self.sp_font.setRange(10, 24)
        self.sp_font.setValue(int(ui.get("font_size", 14)))
        self.chk_capture = QCheckBox("屏幕共享时隐藏本窗口（防截屏）")
        self.chk_capture.setChecked(bool(ui.get("exclude_from_capture", True)))
        self.chk_top = QCheckBox("窗口始终置顶")
        self.chk_top.setChecked(bool(ui.get("always_on_top", True)))
        self.ed_hotkey_toggle = QLineEdit(str(ui.get("hotkey_toggle", "")))
        self.ed_hotkey_answer = QLineEdit(str(ui.get("hotkey_answer", "n")))
        self.ed_hotkey_clear = QLineEdit(str(ui.get("hotkey_clear", "m")))
        self.ed_hotkey_talk = QLineEdit(str(ui.get("hotkey_talk", "space")))
        self.ed_hotkey_listen = QLineEdit(str(ui.get("hotkey_listen", "b")))

        form.addRow("主题", self.cb_theme)
        form.addRow("不透明度", self.sl_opacity)
        form.addRow("正文字号", self.sp_font)
        form.addRow("", self.chk_capture)
        form.addRow("", self.chk_top)
        form.addRow("显示/隐藏热键", self.ed_hotkey_toggle)
        form.addRow("问最近按键", self.ed_hotkey_answer)
        form.addRow("清空按键", self.ed_hotkey_clear)
        form.addRow("按住说话按键", self.ed_hotkey_talk)
        form.addRow("监听开关按键", self.ed_hotkey_listen)
        return page

    def _test_llm(self) -> None:
        """用当前表单里的 Key/地址/模型真实发一次请求。不改动已保存的配置。"""
        from .llm import Answerer

        theme = parent_theme(self.cfg)
        probe = copy.deepcopy(self.cfg)
        cfgmod.set_(probe, "llm.api_key", self.ed_key.text().strip())
        cfgmod.set_(probe, "llm.base_url", self.ed_base.text().strip())
        cfgmod.set_(probe, "llm.model", self.cb_model.currentText().strip())
        cfgmod.set_(probe, "llm.thinking", self.cb_thinking.currentData() or "off")

        self.btn_test.setEnabled(False)
        self.lbl_test.setText("正在测试…")
        self.lbl_test.setStyleSheet(f"color: {theme['muted']};")

        def run_test() -> None:
            try:
                reply = Answerer(probe).ping()
            except Exception as exc:
                self._llm_test_finished.emit(False, f"连接失败：{exc}")
            else:
                self._llm_test_finished.emit(True, f"连接成功，模型回复：{reply}")

        threading.Thread(target=run_test, name="test-llm", daemon=True).start()

    @pyqtSlot(bool, str)
    def _finish_llm_test(self, ok: bool, text: str) -> None:
        theme = parent_theme(self.cfg)
        self.lbl_test.setText(text)
        self.lbl_test.setStyleSheet(f"color: {theme['ok' if ok else 'err']};")
        self.btn_test.setEnabled(True)

    def _test_xfyun(self) -> None:
        """用当前表单里的讯飞凭证真连一次，验证 APPID/Key/Secret 和服务开通状态。"""
        from .asr_xfyun import XfyunTranscriber

        theme = parent_theme(self.cfg)
        probe = copy.deepcopy(self.cfg)
        cfgmod.set_(probe, "asr.xfyun.app_id", self.ed_xf_appid.text().strip())
        cfgmod.set_(probe, "asr.xfyun.api_key", self.ed_xf_key.text().strip())
        cfgmod.set_(probe, "asr.xfyun.api_secret", self.ed_xf_secret.text().strip())

        self.btn_xf_test.setEnabled(False)
        self.lbl_xf_test.setText("正在测试…")
        self.lbl_xf_test.setStyleSheet(f"color: {theme['muted']};")

        def run_test() -> None:
            try:
                reply = XfyunTranscriber(probe["asr"]).verify()
            except Exception as exc:
                self._xfyun_test_finished.emit(False, f"失败：{exc}")
            else:
                self._xfyun_test_finished.emit(True, f"连接成功：{reply}")

        threading.Thread(target=run_test, name="test-xfyun", daemon=True).start()

    @pyqtSlot(bool, str)
    def _finish_xfyun_test(self, ok: bool, text: str) -> None:
        theme = parent_theme(self.cfg)
        self.lbl_xf_test.setText(text)
        self.lbl_xf_test.setStyleSheet(f"color: {theme['ok' if ok else 'err']};")
        self.btn_xf_test.setEnabled(True)

    def _fill_devices(self, current: str) -> None:
        self.cb_device.clear()
        self.cb_device.addItem("默认扬声器（自动）", "")
        try:
            for dev in audio.list_devices():
                if dev["loopback"]:
                    self.cb_device.addItem(f"{dev['name']}  [loopback]", dev["id"])
        except Exception as exc:
            self.cb_device.addItem(f"设备枚举失败：{exc}", "")
        index = self.cb_device.findData(current)
        self.cb_device.setCurrentIndex(max(0, index))

    def _fill_microphones(self, current: str) -> None:
        self.cb_mic.clear()
        self.cb_mic.addItem("系统默认麦克风", "")
        try:
            for mic in audio.list_microphones():
                self.cb_mic.addItem(mic["name"], mic["id"])
        except Exception as exc:
            self.cb_mic.addItem(f"设备枚举失败：{exc}", "")
        index = self.cb_mic.findData(current)
        self.cb_mic.setCurrentIndex(max(0, index))

    # --- 收集结果 ---

    def result_config(self) -> dict:
        cfgmod.set_(self.cfg, "llm.api_key", self.ed_key.text().strip())
        cfgmod.set_(self.cfg, "llm.base_url", self.ed_base.text().strip())
        cfgmod.set_(self.cfg, "llm.model", self.cb_model.currentText().strip())
        cfgmod.set_(self.cfg, "llm.thinking", self.cb_thinking.currentData() or "off")
        cfgmod.set_(self.cfg, "llm.max_chars", self.sp_chars.value())
        cfgmod.set_(self.cfg, "llm.history_rounds", self.sp_rounds.value())
        cfgmod.set_(self.cfg, "llm.style", self.ed_style.toPlainText().strip())
        cfgmod.set_(self.cfg, "llm.resume", self.ed_resume.toPlainText().strip())
        cfgmod.set_(self.cfg, "llm.jd", self.ed_jd.toPlainText().strip())

        cfgmod.set_(self.cfg, "asr.provider", self.cb_provider.currentData() or "auto")
        cfgmod.set_(self.cfg, "asr.xfyun.app_id", self.ed_xf_appid.text().strip())
        cfgmod.set_(self.cfg, "asr.xfyun.api_key", self.ed_xf_key.text().strip())
        cfgmod.set_(self.cfg, "asr.xfyun.api_secret", self.ed_xf_secret.text().strip())
        cfgmod.set_(self.cfg, "asr.model", self.cb_asr.currentText().strip())
        cfgmod.set_(self.cfg, "asr.language", self.cb_lang.currentText())
        cfgmod.set_(self.cfg, "audio.device", self.cb_device.currentData() or "")
        cfgmod.set_(self.cfg, "audio.mic_device", self.cb_mic.currentData() or "")
        cfgmod.set_(self.cfg, "question.auto_answer", self.chk_auto.isChecked())
        cfgmod.set_(self.cfg, "question.threshold", round(self.sp_threshold.value(), 1))
        cfgmod.set_(self.cfg, "question.recent_count", self.sp_recent_count.value())
        cfgmod.set_(self.cfg, "audio.end_silence_ms", self.sp_silence.value())

        cfgmod.set_(self.cfg, "ui.theme", self.cb_theme.currentText())
        cfgmod.set_(self.cfg, "ui.opacity", self.sl_opacity.value() / 100.0)
        cfgmod.set_(self.cfg, "ui.font_size", self.sp_font.value())
        cfgmod.set_(self.cfg, "ui.exclude_from_capture", self.chk_capture.isChecked())
        cfgmod.set_(self.cfg, "ui.always_on_top", self.chk_top.isChecked())
        cfgmod.set_(self.cfg, "ui.hotkey_toggle", self.ed_hotkey_toggle.text().strip())
        cfgmod.set_(self.cfg, "ui.hotkey_answer", self.ed_hotkey_answer.text().strip())
        cfgmod.set_(self.cfg, "ui.hotkey_clear", self.ed_hotkey_clear.text().strip())
        cfgmod.set_(self.cfg, "ui.hotkey_talk", self.ed_hotkey_talk.text().strip())
        cfgmod.set_(self.cfg, "ui.hotkey_listen", self.ed_hotkey_listen.text().strip())
        return self.cfg


def parent_theme(cfg: dict) -> dict:
    return LIGHT if str(cfg["ui"].get("theme", "dark")) == "light" else DARK


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #


class OverlayWindow(QWidget):
    _hotkey_toggle_requested = pyqtSignal()
    _hotkey_answer_requested = pyqtSignal()
    _hotkey_clear_requested = pyqtSignal()
    _hotkey_talk_pressed = pyqtSignal()
    _hotkey_talk_released = pyqtSignal()
    _hotkey_listen_requested = pyqtSignal()

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.pipeline = Pipeline(cfg)
        self._hotkey_toggle_requested.connect(self._toggle_visible_from_hotkey)
        self._hotkey_answer_requested.connect(self._ask_recent_from_hotkey)
        self._hotkey_clear_requested.connect(self._clear_from_hotkey)
        self._hotkey_talk_pressed.connect(self._start_voice_from_hotkey)
        self._hotkey_talk_released.connect(self._stop_voice_from_hotkey)
        self._hotkey_listen_requested.connect(self._toggle_running_from_hotkey)
        # 立刻在后台把大模型链路热起来（import openai + 建客户端 + 建连约 4 秒）。
        # 等用户点「开始监听」时通常已经热好了，第一次提问不再干等。
        self.pipeline.prewarm()
        # 「按住说话」。识别器复用采集链路上那个（本地 whisper 加载一次几百 MB，
        # 为了偶尔用一下语音输入再开一份太浪费），所以传的是取用函数而不是实例。
        self.voice = VoiceInput(cfg, get_transcriber=self.pipeline.ensure_transcriber)
        self._drag_from: QPoint | None = None
        self._answer_buf = ""
        # 点「清空」后置 True，用来丢掉那些已经排进事件队列、还没来得及渲染的
        # 回答增量。否则清空完屏幕又会被填满，看起来像按钮坏了。
        self._answer_muted = False
        self._capture_ok = False
        self._hotkeys: list = []
        self._voice_hotkey_started = False
        # 实时字幕的逐句行：(行容器, 文本标签, 提问按钮)，用于改字号和裁剪
        self._seg_rows: list[tuple[QWidget, QLabel, QPushButton]] = []
        self._max_seg_rows = 60          # 超过就丢最旧的，防止窗口无限增长
        self._seg_font = QFont(self.font().family(), 11)
        self._seg_font_ask = QFont(self.font().family(), 10)
        self.lbl_seg_hint: QLabel | None = None

        self._build_window()
        self._build_ui()
        self._apply_theme()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pump_events)
        self._timer.start(40)

        self._setup_hotkeys()

    # --- 窗口骨架 ---

    def _build_window(self) -> None:
        ui = self.cfg["ui"]
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if ui.get("always_on_top", True):
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("面试副驾")
        self.resize(int(ui.get("width", 480)), int(ui.get("height", 620)))
        self.setWindowOpacity(float(ui.get("opacity", 0.95)))
        self._restore_position()

    def _restore_position(self) -> None:
        """回到上次拖到的位置。显示器拔了/分辨率变了就放弃，避免窗口跑到看不见的地方。"""
        ui = self.cfg["ui"]
        x, y = int(ui.get("x", -1)), int(ui.get("y", -1))
        if x < 0 or y < 0:
            return
        for screen in QGuiApplication.screens():
            if screen.availableGeometry().intersects(QRect(x, y, 80, 40)):
                self.move(x, y)
                return

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.card = QFrame(objectName="card")
        root.addWidget(self.card)
        lay = QVBoxLayout(self.card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        # 标题栏
        header = QHBoxLayout()
        header.setSpacing(7)
        self.dot = QLabel("●")
        self.dot.setFixedWidth(12)
        self.title = QLabel("面试副驾", objectName="title")
        header.addWidget(self.dot)
        header.addWidget(self.title)
        header.addStretch(1)
        for text, slot, tip in (
            ("设置", self._open_settings, "配置大模型 Key、简历、语音模型"),
            ("—", self.showMinimized, "最小化"),
            ("×", self.close, "退出"),
        ):
            btn = QPushButton(text, objectName="icon")
            btn.setToolTip(tip)
            btn.setFixedWidth(34 if text != "设置" else 50)
            btn.clicked.connect(slot)
            header.addWidget(btn)
        lay.addLayout(header)

        self.lbl_status = QLabel("未开始", objectName="status")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        # 实时字幕：一句一行，行尾带「提问此句」
        lay.addWidget(QLabel("实时字幕（每句可单独提问）", objectName="section"))
        self.scroll_transcript = QScrollArea(objectName="transcript")
        self.scroll_transcript.setWidgetResizable(True)
        self.scroll_transcript.setFixedHeight(126)
        self.scroll_transcript.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.transcript_host = QWidget(objectName="transcriptHost")
        self.transcript_layout = QVBoxLayout(self.transcript_host)
        self.transcript_layout.setContentsMargins(8, 6, 8, 6)
        self.transcript_layout.setSpacing(3)
        self.lbl_seg_hint = QLabel(
            "开始后，面试官说的话会逐句出现在这里", objectName="seghint"
        )
        self.lbl_seg_hint.setWordWrap(True)
        self.transcript_layout.addWidget(self.lbl_seg_hint)
        self.transcript_layout.addStretch(1)
        self.scroll_transcript.setWidget(self.transcript_host)
        self.scroll_transcript.verticalScrollBar().rangeChanged.connect(
            lambda _minimum, _maximum: self._scroll_transcript_to_bottom()
        )
        lay.addWidget(self.scroll_transcript)

        # 提问
        lay.addWidget(QLabel("面试官提问", objectName="section"))
        self.lbl_question = QLabel("—", objectName="question")
        self.lbl_question.setWordWrap(True)
        self.lbl_question.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.lbl_question)

        # 自己打字提问。规则漏判、或者想追问某个细节时用，
        # 不依赖麦克风，也不依赖前面有没有识别到字幕。
        ask_row = QHBoxLayout()
        ask_row.setSpacing(6)
        self.ed_ask = QLineEdit(objectName="ask")
        self.ed_ask.setPlaceholderText("也可以自己打字提问，回车发送")
        self.ed_ask.setClearButtonEnabled(True)
        self.ed_ask.setToolTip("在这里输入任意问题，回车或点「提问」直接发给大模型")
        self.ed_ask.returnPressed.connect(self._ask_typed)
        self._ask_escape_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Escape), self.ed_ask
        )
        self._ask_escape_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._ask_escape_shortcut.activated.connect(self.ed_ask.clearFocus)
        # 按住说话：和微信一样，按住录、松开就发。用麦克风（不是采集链路的
        # loopback），所以面试官的声音不会被录进来。
        self.btn_talk = QPushButton("按住说话", objectName="talk")
        self.btn_talk.setFixedWidth(76)
        self.btn_talk.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_talk.setToolTip(
            "按住说话，松开后自动识别并提问。\n"
            "想先看一眼识别结果再发：用左边的输入框打字。"
        )
        self.btn_talk.pressed.connect(self._start_voice)
        self.btn_talk.released.connect(self._stop_voice)
        self.btn_send = QPushButton("提问", objectName="send")
        self.btn_send.setFixedWidth(52)
        self.btn_send.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_send.setToolTip("把输入框里的问题发给大模型（等价于回车）")
        self.btn_send.clicked.connect(self._ask_typed)
        ask_row.addWidget(self.ed_ask, 1)
        ask_row.addWidget(self.btn_talk, 0)
        ask_row.addWidget(self.btn_send, 0)
        lay.addLayout(ask_row)
        # 一开始就把 recording 属性定下来，别让它停在 None——
        # 样式表靠这个属性切换"录音中"的红色，属性没设过就不会生效。
        self._set_talk_state("idle")

        # 回答
        lay.addWidget(QLabel("建议回答", objectName="section"))
        self.txt_answer = QTextBrowser(objectName="answer")
        self.txt_answer.setPlaceholderText("识别到提问后，这里会流式给出可以直接照读的回答")
        lay.addWidget(self.txt_answer, 1)

        # 工具栏
        footer = QHBoxLayout()
        footer.setSpacing(7)
        self.btn_start = QPushButton("开始监听", objectName="primary")
        self.btn_start.setProperty("recording", "false")
        self.btn_start.clicked.connect(self._toggle_running)

        # 加了输入框之后，"手动提问"这个名字会和它撞车——这个按钮其实只问
        # 最近几句字幕，改成"问最近"，并在提示里说清三个入口的分工。
        self.btn_ask = QPushButton("问最近")
        self.btn_ask.setToolTip(
            "把最近几句实时字幕拼起来提问。\n"
            "只想问某一句：点那句话旁边的「提问此句」；\n"
            "想自己打字问：用上面的输入框。"
        )
        self.btn_ask.clicked.connect(self.pipeline.ask_recent)

        self.btn_copy = QPushButton("复制")
        self.btn_copy.setToolTip("复制当前回答")
        self.btn_copy.clicked.connect(self._copy_answer)

        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self._clear)

        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setValue(0)
        self.level.setTextVisible(False)
        self.level.setFixedWidth(56)
        self.level.setFixedHeight(6)

        footer.addWidget(self.btn_start)
        footer.addWidget(self.btn_ask)
        footer.addWidget(self.btn_copy)
        footer.addWidget(self.btn_clear)
        footer.addStretch(1)
        footer.addWidget(self.level)
        footer.addWidget(QSizeGrip(self))
        lay.addLayout(footer)

    def _apply_theme(self) -> None:
        theme = parent_theme(self.cfg)
        self.setStyleSheet(STYLE.substitute(theme))
        self.dot.setStyleSheet(f"color: {theme['muted']};")
        size = int(self.cfg["ui"].get("font_size", 14))
        self.txt_answer.setFont(QFont(self.font().family(), size))
        self.lbl_question.setFont(QFont(self.font().family(), size))
        self.ed_ask.setFont(QFont(self.font().family(), size))
        self.btn_talk.setFont(QFont(self.font().family(), max(11, size - 2)))
        self.btn_send.setFont(QFont(self.font().family(), max(11, size - 2)))
        self._seg_font = QFont(self.font().family(), max(10, size - 3))
        self._seg_font_ask = QFont(self.font().family(), max(9, size - 4))
        for row, label, _btn in self._seg_rows:
            label.setFont(self._seg_font)

    # --- 交互 ---

    def mousePressEvent(self, event):  # noqa: N802 - Qt 命名
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < 46:
            self._drag_from = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_from is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_from)
            event.accept()

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._drag_from = None

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if not self._capture_ok and self.cfg["ui"].get("exclude_from_capture", True):
            self._capture_ok = apply_capture_exclusion(self, True)
            if self._capture_ok:
                self.lbl_status.setText("防截屏已开启，共享屏幕时对方看不到本窗口")

    def closeEvent(self, event):  # noqa: N802
        # 按住说话时被关窗：得先把麦克风放掉，否则设备要等进程退出才释放
        self.voice.stop()
        self.pipeline.shutdown()
        self._teardown_hotkeys()
        try:
            # 只写窗口位置。不要 save(self.cfg)——那会把启动时那份旧配置整份写回，
            # 冲掉用户在程序运行期间手改的内容（换模型、填 Key 都会被抹掉）。
            cfgmod.save_ui_position(self.width(), self.height(), self.x(), self.y())
        except Exception:
            pass
        super().closeEvent(event)

    def _toggle_running(self) -> None:
        if self.pipeline.running:
            self.pipeline.stop()
            self._set_button_state(False)
        else:
            self.pipeline.start()
            self._set_button_state(True)

    def _set_button_state(self, running: bool) -> None:
        self.btn_start.setText("停止监听" if running else "开始监听")
        self.btn_start.setProperty("recording", "true" if running else "false")
        self.btn_start.style().unpolish(self.btn_start)
        self.btn_start.style().polish(self.btn_start)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.cfg, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_cfg = dialog.result_config()
            cfgmod.save(new_cfg)
            self.cfg = new_cfg
            # VoiceInput 只在录音那一刻读 cfg["audio"]["mic_device"]，
            # 但 cfg 是个 dict 引用——不换掉的话改完麦克风还用旧的。
            self.voice.cfg = new_cfg
            self._apply_theme()
            self.setWindowOpacity(float(new_cfg["ui"].get("opacity", 0.95)))
            apply_always_on_top(self, bool(new_cfg["ui"].get("always_on_top", True)))
            self._capture_ok = apply_capture_exclusion(self, bool(new_cfg["ui"].get("exclude_from_capture", True)))
            self.pipeline.apply_config(new_cfg)
            self._setup_hotkeys()
            self._flash("设置已保存")

    def _copy_answer(self) -> None:
        QGuiApplication.clipboard().setText(self.txt_answer.toPlainText())

    def _clear(self) -> None:
        self.pipeline.clear_history()
        self._clear_transcript()
        self.txt_answer.clear()
        self.lbl_question.setText("—")
        self.ed_ask.clear()
        self._answer_buf = ""
        # 正在生成的那条也要掐掉，并把它的残余增量挡在门外
        self._answer_muted = True
        self.pipeline.cancel_answer()

    def _flash(self, text: str) -> None:
        self.lbl_status.setText(text)

    # --- 实时字幕：逐句行 ---

    def _add_transcript_row(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if self.lbl_seg_hint is not None:
            self.transcript_layout.removeWidget(self.lbl_seg_hint)
            self.lbl_seg_hint.deleteLater()
            self.lbl_seg_hint = None

        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(6)

        label = QLabel(text, objectName="segtext")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setFont(self._seg_font)

        btn = QPushButton("提问此句", objectName="segask")
        btn.setFont(self._seg_font_ask)
        btn.setFixedWidth(62)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip("把这一句单独发给大模型")
        # 用默认参数把 text 绑进闭包，避免所有按钮都指向最后一句
        btn.clicked.connect(lambda _checked=False, sentence=text: self._ask_sentence(sentence))

        line.addWidget(label, 1)
        line.addWidget(btn, 0, Qt.AlignmentFlag.AlignTop)
        self.transcript_layout.insertWidget(self.transcript_layout.count() - 1, row)
        self._seg_rows.append((row, label, btn))

        self._trim_transcript_rows()

    def _scroll_transcript_to_bottom(self) -> None:
        bar = self.scroll_transcript.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _trim_transcript_rows(self) -> None:
        while len(self._seg_rows) > self._max_seg_rows:
            row, _label, _btn = self._seg_rows.pop(0)
            self.transcript_layout.removeWidget(row)
            row.deleteLater()

    def _clear_transcript(self) -> None:
        for row, _label, _btn in self._seg_rows:
            self.transcript_layout.removeWidget(row)
            row.deleteLater()
        self._seg_rows.clear()

    def _ask_sentence(self, sentence: str) -> None:
        self.pipeline.ask(sentence)
        self._flash(f"已提交提问：{sentence[:24]}{'…' if len(sentence) > 24 else ''}")

    def _ask_typed(self) -> None:
        """把输入框里的内容当成问题发出去。

        空输入要有明确反馈——直接 return 的话，用户点了按钮什么都没发生，
        和之前"手动提问没反应"是同一类体验问题。
        """
        text = self.ed_ask.text().strip()
        if not text:
            self._flash("先输入要问的内容，再回车或点「提问」")
            self.ed_ask.setFocus()
            return
        self.ed_ask.clear()
        self._ask_sentence(text)

    # --- 按住说话 ---

    def _start_voice(self) -> bool:
        """按下按钮。真正的录音在后台线程里，这里立刻返回。"""
        if not self.voice.start():
            # 上一次还在识别。按住不放没用，明确说一句，别让人以为按坏了。
            self._flash("上一段还在识别，稍等一下再按")
            return False
        self._set_talk_state("recording")
        self.lbl_status.setText("正在听…说完松开按钮")
        return True

    def _stop_voice(self) -> None:
        """松开按钮：停录，线程接着做识别，结果从事件队列回来。

        判据用 busy 而不是 listening：录音满了 30 秒会自动停，那时已经不在录、
        但还在识别——这种"手还按着、录已经停了"的情况也得能正常收尾。
        """
        if not self.voice.busy:
            return
        self.voice.stop()
        self._set_talk_state("working")

    def _set_talk_state(self, state: str) -> None:
        text = {"idle": "按住说话", "recording": "松开结束", "working": "识别中…"}[state]
        self.btn_talk.setText(text)
        self.btn_talk.setProperty("recording", "true" if state == "recording" else "false")
        self.btn_talk.style().unpolish(self.btn_talk)
        self.btn_talk.style().polish(self.btn_talk)

    def _submit_voice_text(self, text: str) -> None:
        """识别结果：先填进输入框（让人看见识别成了什么），再直接提问。

        不做二次确认——「按住说话」的价值就是少点一下；识别错了重按一次
        比每次都要再确认一遍划算。
        """
        self.ed_ask.setText(text)
        self._ask_sentence(text)

    def _show_answer_placeholder(self) -> None:
        """提问后立刻给点动静。空白回答区会让人以为"点了没反应"。"""
        theme = parent_theme(self.cfg)
        self.txt_answer.setHtml(f'<span style="color:{theme["muted"]}">正在生成…</span>')

    # --- 事件泵 ---

    def _pump_events(self) -> None:
        self._pump_voice_events()
        for event in self.pipeline.drain():
            kind = event.get("type")
            if kind == "status":
                self.lbl_status.setText(event.get("text", ""))
                self._color_dot(event.get("state", "idle"))
                if event.get("state") == "idle" and self.pipeline.running is False:
                    self._set_button_state(False)
            elif kind == "level":
                self.level.setValue(int(float(event.get("value", 0)) * 100))
            elif kind == "download":
                name = event.get("file", "")
                done = int(event.get("done", 0))
                total = int(event.get("total", 0))
                if total > 0:
                    self.lbl_status.setText(
                        f"正在下载 {name}  {done / 1e6:.0f}/{total / 1e6:.0f} MB（{done * 100 // total}%）"
                    )
                    self.level.setValue(done * 100 // total)
                else:
                    self.lbl_status.setText(f"正在下载 {name}  {done / 1e6:.0f} MB")
            elif kind == "transcript":
                self._add_transcript_row(event.get("text", ""))
            elif kind == "question":
                # 只有这个事件代表"用户/规则真的提了一个新问题"——它是 ask() 在
                # UI 线程同步发出的。所以只有它能解除静音。
                self.lbl_question.setText(event.get("text", ""))
                self._answer_buf = ""
                self._answer_muted = False
                self._show_answer_placeholder()
            elif kind == "answer_start":
                # answer_start 是作答线程异步发的，可能晚于「清空」才到达。
                # 它属于被取消的那条回答，不能再解除静音、也不能再显示占位。
                if self._answer_muted:
                    continue
                self._answer_buf = ""
                self._show_answer_placeholder()
            elif kind == "answer_delta":
                # 刚点过「清空」：这是在途回答的残余增量，丢掉
                if self._answer_muted:
                    continue
                self._answer_buf += event.get("text", "")
                self.txt_answer.setPlainText(self._answer_buf)
                self.txt_answer.moveCursor(QTextCursor.MoveOperation.End)
                self.txt_answer.ensureCursorVisible()
            elif kind == "answer_end":
                if self._answer_muted:
                    continue
                # 一个字都没出（被新问题取消 / 报错），别把"正在生成…"留在屏幕上
                if not self._answer_buf.strip():
                    self.txt_answer.clear()
            elif kind == "error":
                self.lbl_status.setText(event.get("text", ""))
                self._color_dot("error")
                if self.pipeline.running is False:
                    self._set_button_state(False)

    def _pump_voice_events(self) -> None:
        """把「按住说话」的事件渲染出来。和主链路同一个泵，同一个线程。"""
        for event in self.voice.drain():
            kind = event.get("type")
            if kind == "status":
                self.lbl_status.setText(event.get("text", ""))
            elif kind == "text":
                self._submit_voice_text(event.get("text", ""))
            elif kind == "error":
                self._flash(event.get("text", ""))
                self._color_dot("error")
            elif kind == "done":
                # 录完了、也识别完了（或失败了）：按钮回到初始样子
                self._set_talk_state("idle")

    def _color_dot(self, state: str) -> None:
        theme = parent_theme(self.cfg)
        color = {
            "running": theme["ok"],
            "loading": theme["warn"],
            "starting": theme["warn"],
            "error": theme["err"],
        }.get(state, theme["muted"])
        self.dot.setStyleSheet(f"color: {color};")

    # --- 全局热键 ---

    def _hotkeys_suspended(self) -> bool:
        return self.isActiveWindow() and self.ed_ask.hasFocus()

    def _setup_hotkeys(self) -> None:
        self._teardown_hotkeys()
        ui = self.cfg["ui"]
        try:
            import keyboard  # 可选依赖
        except Exception:
            return

        def bind(combo: str, action) -> None:
            if not combo:
                return
            try:
                handle = keyboard.add_hotkey(combo, action, suppress=False)
            except Exception:
                return
            self._hotkeys.append(lambda handle=handle: keyboard.remove_hotkey(handle))

        def bind_press_cycle(key: str, pressed, released=None) -> None:
            """同一次物理按键只触发一次，避免长按产生的自动重复。"""
            if not key:
                return
            lock = threading.Lock()
            down = False

            def on_event(event) -> None:
                nonlocal down
                is_down = event.event_type == keyboard.KEY_DOWN
                with lock:
                    if down == is_down:
                        return
                    down = is_down
                if is_down:
                    pressed()
                elif released is not None:
                    released()

            try:
                handle = keyboard.hook_key(key, on_event, suppress=False)
            except Exception:
                return
            self._hotkeys.append(lambda handle=handle: keyboard.unhook(handle))

        def bind_action(combo: str, action) -> None:
            if "+" in combo:
                bind(combo, action)
            else:
                bind_press_cycle(combo, action)

        bind(str(ui.get("hotkey_toggle") or ""), self._hotkey_toggle_requested.emit)
        bind_action(
            str(ui.get("hotkey_answer") or ""), self._hotkey_answer_requested.emit
        )
        bind_action(
            str(ui.get("hotkey_clear") or ""), self._hotkey_clear_requested.emit
        )
        bind_press_cycle(
            str(ui.get("hotkey_talk") or ""),
            self._hotkey_talk_pressed.emit,
            self._hotkey_talk_released.emit,
        )
        bind_action(
            str(ui.get("hotkey_listen") or ""), self._hotkey_listen_requested.emit
        )

    def _teardown_hotkeys(self) -> None:
        if not self._hotkeys:
            return
        try:
            import keyboard

            for unbind in self._hotkeys:
                try:
                    unbind()
                except Exception:
                    pass
        except Exception:
            pass
        self._hotkeys = []

    @pyqtSlot()
    def _start_voice_from_hotkey(self) -> None:
        if self._hotkeys_suspended():
            return
        if not self._voice_hotkey_started:
            self._voice_hotkey_started = self._start_voice()

    @pyqtSlot()
    def _stop_voice_from_hotkey(self) -> None:
        if not self._voice_hotkey_started:
            return
        self._voice_hotkey_started = False
        self._stop_voice()

    @pyqtSlot()
    def _toggle_visible_from_hotkey(self) -> None:
        if self._hotkeys_suspended():
            return
        self._toggle_visible()

    def _toggle_visible(self) -> None:
        self.setVisible(not self.isVisible())

    @pyqtSlot()
    def _ask_recent_from_hotkey(self) -> None:
        if self._hotkeys_suspended():
            return
        self.pipeline.ask_recent()

    @pyqtSlot()
    def _clear_from_hotkey(self) -> None:
        if self._hotkeys_suspended():
            return
        self._clear()

    @pyqtSlot()
    def _toggle_running_from_hotkey(self) -> None:
        if self._hotkeys_suspended():
            return
        self._toggle_running()


def run(cfg: dict) -> int:
    if not _claim_single_instance():
        return 0
    app = QApplication(sys.argv)
    app.setApplicationName("面试副驾")
    window = OverlayWindow(cfg)
    window.show()
    return app.exec()
