"""判定一段话是不是面试官提出的问题。

纯规则打分，零依赖、零延迟。规则来自中文面试里面试官的实际措辞习惯：
疑问词、祈使句（"介绍一下"）、句尾语气词、问号。
"""

from __future__ import annotations

import re

# 强疑问词：出现即可基本断定是提问（ASR 经常丢掉句末问号，不能只靠标点）
_ZH_MARKERS_STRONG = (
    "为什么", "为啥", "怎么", "怎样", "如何", "请问", "请教", "想问", "问一下", "问个问题",
    "是否", "能否", "能不能", "可不可以", "是不是", "有没有", "对不对", "行不行", "好不好",
    "哪个", "哪些", "哪里", "哪儿", "多少", "多久", "几年", "考考你",
)

# 弱疑问词：可能出现在陈述句里（"我知道什么是对的"），单独出现不判为提问
_ZH_MARKERS_WEAK = (
    "什么", "谁",
)

# 面试官常用的祈使式提问（不带问号也是提问）
_IMPERATIVES = (
    "介绍一下", "介绍下", "介绍介绍", "说说", "说一下", "讲讲", "讲一下", "谈谈", "聊一聊",
    "举个例子", "举例说明", "举个例子说明", "描述一下", "阐述一下", "解释一下", "说明一下",
    "展开说说", "展开讲讲", "详细说说", "详细讲讲", "你来说说", "你怎么看", "你认为", "你觉得",
    "你的理解", "你的看法", "你的思路", "你的想法", "你会怎么做", "你会怎么处理", "你打算",
    "你是怎么", "你是如何", "你负责", "你遇到过", "你做过", "你的项目", "你的经历",
    "评价一下", "分析一下", "对比一下", "推荐一下", "设计一个", "实现一个", "写一个",
)

# 句尾语气词（出现在末尾时是疑问句的强信号）
_TAIL_PARTICLES = ("吗", "呢", "吧", "么", "嘛")

_EN_OPENERS = (
    "what", "why", "how", "when", "where", "which", "who", "whose", "whom",
    "do you", "did you", "can you", "could you", "would you", "will you",
    "are you", "is it", "have you", "has it", "tell me", "describe", "explain",
    "walk me through", "give me an example", "talk about",
)

_TRAILING_PUNCT = re.compile(r"[\s，。、；：,.;:~…!！]+$")
_QUESTION_TAIL = re.compile(r"[?？]$")


def clean(text: str) -> str:
    """去掉首尾空白和句尾多余标点（问号除外）。"""
    text = re.sub(r"\s+", " ", (text or "").strip())
    while _TRAILING_PUNCT.search(text):
        text = _TRAILING_PUNCT.sub("", text)
    return text.strip()


def score(text: str) -> tuple[float, list[str]]:
    """返回 (得分, 命中的规则说明)。得分越高越像提问。"""
    raw = (text or "").strip()
    if not raw:
        return 0.0, []

    hits: list[str] = []
    total = 0.0

    if _QUESTION_TAIL.search(raw):
        total += 4.0
        hits.append("问号结尾")

    lowered = raw.lower()
    # 先查强疑问词，再查弱疑问词；命中一个就停，避免"为什么…什么"被重复计分
    for markers, weight, label in ((_ZH_MARKERS_STRONG, 3.0, "强疑问词"), (_ZH_MARKERS_WEAK, 2.0, "弱疑问词")):
        for marker in markers:
            if marker in raw:
                total += weight
                hits.append(f"{label}「{marker}」")
                break
        else:
            continue
        break

    if raw.endswith(_TAIL_PARTICLES):
        total += 2.5
        hits.append("句尾语气词")

    for word in _IMPERATIVES:
        if word in raw:
            total += 3.0
            hits.append(f"祈使式「{word}」")
            break

    for opener in _EN_OPENERS:
        if lowered.startswith(opener) or f" {opener}" in lowered:
            total += 3.0
            hits.append(f"英文疑问「{opener}」")
            break

    # 出现两个以上不同疑问信号，基本可以确定是提问
    if len(hits) >= 2:
        total += 1.0
        hits.append("多重疑问信号")

    return total, hits


def is_question(text: str, min_chars: int = 5, threshold: float = 3.0) -> bool:
    candidate = clean(text)
    if len(candidate) < min_chars:
        return False
    total, _ = score(candidate)
    return total >= threshold
