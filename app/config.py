"""配置：默认值 <- config.yaml <- 环境变量，逐层覆盖。"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
EXAMPLE_PATH = ROOT / "config.example.yaml"

DEFAULTS: dict[str, Any] = {
    "audio": {
        # 采集设备：听面试官说话用，采的是系统输出（loopback）
        "device": "",
        # 麦克风：「按住说话」用，采的是你自己的声音。留空 = 系统默认输入设备
        "mic_device": "",
        "samplerate": 0,
        "block_ms": 30,
        "noise_ratio": 2.6,
        "min_rms": 0.0035,
        "start_ms": 120,
        "end_silence_ms": 600,
        "min_speech_ms": 400,
        "max_speech_ms": 20000,
        "preroll_ms": 200,
    },
    "asr": {
        # auto = 配了讯飞凭证就用讯飞，否则用本地 whisper
        # xfyun = 强制讯飞云端；local = 强制本地 whisper
        "provider": "auto",
        # ---- 本地 whisper（provider=local 时生效）----
        "model": "small",
        "language": "zh",
        "compute_type": "int8",
        "beam_size": 1,
        "cpu_threads": 0,
        "initial_prompt": "以下是中文技术面试的对话，涉及编程、算法、系统设计、项目经历、职业规划。",
        "hf_endpoint": "https://hf-mirror.com",
        # ---- 讯飞语音听写（provider=xfyun 时生效）----
        "xfyun": {
            "app_id": "",
            "api_key": "",
            "api_secret": "",
            "endpoint": "wss://iat-api.xfyun.cn/v2/iat",
            "language": "zh_cn",   # zh_cn / en_us
            "domain": "iat",       # iat=日常用语；xfime-mianqie=方言免切
            "accent": "mandarin",
            "dwa": "",             # 填 wpgs 开启动态修正
            "ptt": 1,              # 1=加标点
            # 每帧之间等多久再发下一帧。0 = 不等待、全速发送（推荐）。
            # 我们拿到的是完整音频段而非实时流，按 40ms 实时速率发纯属白等：
            # 实测 26s 音频，0ms 用 1.2s 识别完，40ms 要 27s。
            "frame_interval_ms": 0,
            "timeout": 15,
        },
    },
    "question": {
        "min_chars": 5,
        "threshold": 3.0,
        "merge_gap_ms": 1200,
        "recent_count": 3,
        "auto_answer": True,
    },
    "llm": {
        "base_url": "https://api.deepseek.com",
        "api_key": "",
        "model": "deepseek-chat",
        "temperature": 0.4,
        "max_tokens": 2000,
        "max_chars": 260,
        "timeout": 60,
        # 思考模式。deepseek-v4 系列默认开着思考，会先吐几百到几千字的
        # reasoning_content 再给正文——面试场景纯属浪费，实测关掉后
        # 总耗时从 3.8~9.0 秒降到 1.3~1.8 秒，答案还更贴合字数上限。
        #   off    = 关闭思考（最快，面试推荐）
        #   auto   = 不干预，用服务端默认（默认开启，强度 high）
        #   low / medium / high / max = 开启思考并指定强度
        "thinking": "off",
        "style": "使用第一人称口语回答，表达沉稳、具体、简洁。围绕目标岗位突出可迁移经验，但不虚构未做过的技术或项目。",
        "resume": "",
        "jd": "",
        "history_rounds": 4,
    },
    "ui": {
        "width": 480,
        "height": 680,   # 比早期版本高 60px：加了「自己打字提问」输入框
        "x": -1,          # -1 = 自动定位；拖动过窗口后会记住位置
        "y": -1,
        "opacity": 0.95,
        "font_size": 14,
        "theme": "dark",
        "exclude_from_capture": True,
        "always_on_top": True,
        "hotkey_toggle": "ctrl+alt+h",
        "hotkey_answer": "n",
        "hotkey_clear": "m",
        "hotkey_talk": "space",
        "hotkey_listen": "b",
    },
}

# 环境变量 -> 配置路径。按顺序应用，后面的覆盖前面的（即优先级更高）。
_ENV_OVERRIDES = (
    ("INTERVIEW_LLM_BASE_URL", "llm.base_url"),
    ("INTERVIEW_LLM_MODEL", "llm.model"),
    ("INTERVIEW_ASR_MODEL", "asr.model"),
    ("HF_ENDPOINT", "asr.hf_endpoint"),
    ("XFYUN_APP_ID", "asr.xfyun.app_id"),
    ("XFYUN_API_KEY", "asr.xfyun.api_key"),
    ("XFYUN_API_SECRET", "asr.xfyun.api_secret"),
    ("DEEPSEEK_API_KEY", "llm.api_key"),
    ("OPENAI_API_KEY", "llm.api_key"),  # 标准变量名，与上一个同时存在时以它为准
)


def _deep_merge(base: dict, patch: dict) -> dict:
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def get(cfg: dict, path: str, default: Any = None) -> Any:
    """按 "llm.api_key" 形式取值。"""
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_(cfg: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load(path: Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    target = path or CONFIG_PATH
    if target.exists():
        with open(target, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        if not isinstance(user_cfg, dict):
            raise ValueError(f"{target} 格式不对，顶层应该是键值对。")
        _deep_merge(cfg, user_cfg)

    for env_key, cfg_path in _ENV_OVERRIDES:
        value = os.environ.get(env_key)
        if value:
            set_(cfg, cfg_path, value)
    return cfg


def save(cfg: dict, path: Path | None = None) -> None:
    target = path or CONFIG_PATH
    with open(target, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False, indent=2)


def save_ui_position(width: int, height: int, x: int, y: int, path: Path | None = None) -> None:
    """只把窗口位置/尺寸写回文件。

    必须重新读一遍磁盘上的文件再改，不能拿内存里的整份配置覆盖：
    程序启动后用户可能在文件里手改了配置（换模型、填 Key），
    用启动时那份旧副本整份写回会把这些改动悄悄冲掉。
    """
    target = path or CONFIG_PATH
    data: dict = {}
    if target.exists():
        with open(target, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if isinstance(loaded, dict):
            data = loaded

    ui = data.get("ui")
    if not isinstance(ui, dict):
        ui = {}
        data["ui"] = ui
    ui["width"] = int(width)
    ui["height"] = int(height)
    ui["x"] = int(x)
    ui["y"] = int(y)

    with open(target, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False, indent=2)


def ensure_config_file() -> Path:
    """首次运行时从示例复制一份 config.yaml。"""
    if not CONFIG_PATH.exists() and EXAMPLE_PATH.exists():
        CONFIG_PATH.write_text(EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return CONFIG_PATH
