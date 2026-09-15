"""入口：参数解析、环境自检、启动界面。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Windows 控制台默认 GBK，中文输出会炸，先强制 UTF-8
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _banner(text: str) -> None:
    print(f"\n=== {text} ===")


def cmd_list_devices() -> int:
    from . import audio

    _banner("可用的采集设备（带 [loopback] 的才能抓到腾讯会议的声音）")
    try:
        for dev in audio.list_devices():
            mark = "  <- 默认扬声器 loopback" if dev["loopback"] else ""
            print(f"[{'loopback' if dev['loopback'] else '  mic   '}] {dev['name']}{mark}")
            print(f"           id: {dev['id']}")
    except Exception as exc:
        print(f"枚举失败：{exc}")
        return 1
    return 0


def cmd_check(cfg: dict) -> int:
    """面试前跑一遍，确认每一环都是通的。"""
    _banner("环境自检")
    ok = True

    print(f"Python           {sys.version.split()[0]}")

    for name, module in (("faster-whisper", "faster_whisper"), ("soundcard", "soundcard"),
                         ("PyQt6", "PyQt6"), ("openai", "openai"), ("numpy", "numpy")):
        try:
            __import__(module)
            print(f"[OK]   {name}")
        except Exception as exc:
            ok = False
            print(f"[FAIL] {name} 未安装：{exc}")

    try:
        from . import audio

        loopback = [d for d in audio.list_devices() if d["loopback"]]
        if loopback:
            print(f"[OK]   找到 {len(loopback)} 个 loopback 设备，启动时会自动锁定在出声的那个：")
            for dev in loopback:
                print(f"         - {dev['name']}")
        else:
            ok = False
            print("[FAIL] 没找到 loopback 设备，无法采集系统声音")
    except Exception as exc:
        ok = False
        print(f"[FAIL] 音频设备检查失败：{exc}")

    key = str(
        cfg["llm"].get("api_key")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or ""
    )
    if key:
        print(f"[OK]   大模型 Key 已配置（{key[:6]}…{key[-4:]}）")
        try:
            from .llm import Answerer

            reply = Answerer(cfg).ping()
            print(f"[OK]   大模型实际调用成功，模型回复：{reply!r}")
        except Exception as exc:
            ok = False
            print(f"[FAIL] 大模型调用失败：{exc}")
    else:
        ok = False
        print("[FAIL] 大模型 Key 未配置，界面里点「设置」填写，"
              "或设置环境变量 OPENAI_API_KEY")

    if str(cfg["llm"].get("resume") or "").strip():
        print("[OK]   简历已填写，回答会贴合你的经历")
    else:
        print("[WARN] 简历为空，回答会比较通用（设置里可以补）")

    from . import model_store
    from .asr import create_transcriber, provider_label
    from .asr_xfyun import XfyunTranscriber

    transcriber = create_transcriber(cfg["asr"])
    print(f"[OK]   识别引擎：{provider_label(transcriber)}")

    if isinstance(transcriber, XfyunTranscriber):
        try:
            reply = transcriber.verify()
            print(f"[OK]   讯飞连接与鉴权正常（{reply}）")
        except Exception as exc:
            ok = False
            print(f"[FAIL] 讯飞调用失败：{exc}")
    else:
        want = str(cfg["asr"]["model"])
        local = model_store.list_local()
        if want in local:
            size = sum(f.stat().st_size for f in model_store.model_dir(want).rglob("*") if f.is_file())
            print(f"[OK]   本地语音模型 {want} 已就绪（{size / 1e6:.0f}MB）")
        else:
            print(f"[WARN] 本地语音模型 {want} 尚未下载，首次点「开始监听」会自动下载"
                  f"（约 {model_store._approx_size(want)}）")
        if local:
            print(f"       本地已有模型：{', '.join(local)}")

    print(f"\n防截屏           {'开启' if cfg['ui'].get('exclude_from_capture') else '关闭'}")
    # 「按住说话」用的麦克风。和上面那些 loopback 设备是两码事：那些是"听面试官"，
    # 这个是"听你自己"。这一项不通，界面上的「按住说话」按下去就没反应。
    try:
        from . import audio as audio_mod

        mic = audio_mod.resolve_microphone(str(cfg["audio"].get("mic_device") or ""))
        print(f"[OK]   麦克风（按住说话）：{mic.name}")
    except Exception as exc:
        print(f"[FAIL] 麦克风不可用，「按住说话」会失败：{exc}")
        ok = False
    print(f"大模型           {cfg['llm']['model']} @ {cfg['llm']['base_url']}")
    from .llm import Answerer

    print(f"思考模式         {Answerer(cfg).thinking_label()}")
    print(f"\n结论：{'全部就绪，可以开始' if ok else '有未通过项，请按上面提示处理'}")
    return 0 if ok else 1


def cmd_test_audio(seconds: float = 3.0) -> int:
    """对着每个 loopback 设备听几秒，告诉你到底哪个在出声。"""
    import time

    from . import audio

    _banner(f"逐个试听 loopback 设备，每台 {seconds:.0f} 秒")
    print("请现在让电脑播放声音（比如放一段腾讯会议的语音），否则测不出结果。\n")
    try:
        ranked = audio.probe_loopback(seconds)
    except Exception as exc:
        print(f"探测失败：{exc}")
        return 1

    if not ranked:
        print("没有找到任何 loopback 设备。")
        return 1

    best = ranked[0]
    for mic, peak in ranked:
        mark = "  <-- 声音在这里" if peak > audio.AUDIBLE_RMS and mic is best[0] else ""
        print(f"音量 {peak:.5f}  {'[有声]' if peak > audio.AUDIBLE_RMS else '[无声]'}  {mic.name}{mark}")

    print()
    if best[1] > audio.AUDIBLE_RMS:
        print(f"结论：声音来自「{best[0].name}」。")
        print("把它的 id 填到 config.yaml 的 audio.device，或留空让程序每次自动识别：")
        print(f"  audio.device: \"{best[0].id}\"")
    else:
        print("结论：所有设备都没有声音。请确认电脑确实在播放音频，然后重试。")
    return 0


def cmd_ask(question: str, cfg: dict) -> int:
    """不开界面，命令行直接试答一个问题。用来验证 Key 和回答风格。"""
    from .llm import Answerer

    _banner("试答")
    print(f"问题：{question}\n")
    answerer = Answerer(cfg)
    pieces: list[str] = []
    try:
        for piece in answerer.stream(question):
            pieces.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
    except Exception as exc:
        print(f"\n\n生成失败：{exc}")
        return 1
    print(f"\n\n（共 {len(''.join(pieces))} 字）")
    return 0


def cmd_selftest() -> int:
    """规则层单元自检：断句 + 提问判定。"""
    import numpy as np

    from . import question as qmod
    from .segmenter import Segmenter

    _banner("提问判定")
    cases = [
        ("请介绍一下你最有挑战的一个项目", True),
        ("你刚才说的那个方案，如果并发量再涨十倍你会怎么优化？", True),
        ("说说你对微服务的理解", True),
        ("MySQL 的索引为什么用 B+ 树", True),
        ("你的职业规划是什么", True),
        ("如果线上出现内存泄漏你会如何排查", True),
        ("Redis 的持久化机制是怎么实现的", True),
        ("今天天气不错", False),
        ("嗯，好的", False),
        ("我们团队用的是 Java 和 Spring Cloud", False),
        ("我知道什么是对的", False),
    ]
    failed = 0
    for text, expected in cases:
        total, hits = qmod.score(text)
        got = qmod.is_question(text)
        flag = "OK  " if got == expected else "FAIL"
        if got != expected:
            failed += 1
        print(f"[{flag}] {got!s:5} (score={total:.1f}) {text}   {'; '.join(hits)}")

    _banner("音频断句")
    sr = 16000
    block = int(sr * 0.03)
    rng = np.random.default_rng(0)
    seg = Segmenter(sr=sr)
    silence = rng.normal(0, 0.0008, block).astype(np.float32)
    speech = (rng.normal(0, 0.08, block) * np.hanning(block)).astype(np.float32)

    got = []
    for _ in range(30):
        got += seg.push(silence)
    for _ in range(30):          # 0.9s 人声
        got += seg.push(speech)
    for _ in range(40):          # 1.2s 静音
        got += seg.push(silence)
    if got:
        print(f"[OK  ] 切出 {len(got)} 段，长度 {got[0].size / sr:.2f}s")
    else:
        failed += 1
        print("[FAIL] 没有切出任何语音段")

    print(f"\n自检结果：{'通过' if failed == 0 else f'{failed} 项失败'}")
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="interview-copilot", description="面试副驾：实时识别提问并生成可照读的回答")
    parser.add_argument("--config", type=Path, default=None, help="指定配置文件路径")
    parser.add_argument("--list-devices", action="store_true", help="列出音频采集设备")
    parser.add_argument("--test-audio", type=float, nargs="?", const=3.0, default=None,
                        metavar="秒数", help="逐个试听 loopback 设备，找出真正在出声的那个")
    parser.add_argument("--check", action="store_true", help="运行环境自检")
    parser.add_argument("--selftest", action="store_true", help="运行规则层自检")
    parser.add_argument("--ask", metavar="问题", default=None,
                        help="不开界面，直接试答一个问题（验证 Key 和回答风格）")
    args = parser.parse_args(argv)

    from . import config as cfgmod

    if args.list_devices:
        return cmd_list_devices()
    if args.test_audio is not None:
        return cmd_test_audio(args.test_audio)

    cfgmod.ensure_config_file()
    cfg = cfgmod.load(args.config)

    if args.check:
        return cmd_check(cfg)
    if args.selftest:
        return cmd_selftest()
    if args.ask:
        return cmd_ask(args.ask, cfg)

    from . import ui

    return ui.run(cfg)
