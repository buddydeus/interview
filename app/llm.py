"""大模型作答：任何 OpenAI 兼容接口，流式返回。

注意：部分自建网关会拦截 OpenAI SDK 的默认 User-Agent（返回 403 "Your request was blocked."），
所以下面统一覆盖成自己的 UA。
"""

from __future__ import annotations

import os
import threading
from typing import Iterator

USER_AGENT = "interview-copilot/0.1"

# 思考模式 -> (thinking 字段, reasoning_effort)
#
# 为什么关思考要用 `thinking` 而不是 `reasoning_effort`：
# DeepSeek 官方把 `reasoning_effort` 的取值定义为 low/medium/high/max，
# 没有"关闭"这一档；Responses API 才用 reasoning.effort="none" 关。
# 关思考的官方写法是 thinking.type=disabled，且必须走 extra_body
# —— 直接当顶层参数塞给 SDK 会被丢掉（SDK 不认识这个字段）。
_THINKING_MODES: dict[str, tuple[dict | None, str | None]] = {
    "off": ({"type": "disabled"}, None),
    "disabled": ({"type": "disabled"}, None),
    "auto": (None, None),          # 不干预，服务端默认（开启 / high）
    "low": ({"type": "enabled"}, "low"),
    "medium": ({"type": "enabled"}, "medium"),
    "high": ({"type": "enabled"}, "high"),
    "max": ({"type": "enabled"}, "max"),
}

# 界面下拉与 --check 共用同一份选项，避免两处各写一份标签对不上
THINKING_CHOICES: tuple[tuple[str, str], ...] = (
    ("off", "关闭（最快）"),
    ("auto", "服务端默认"),
    ("low", "开启 · 低"),
    ("medium", "开启 · 中"),
    ("high", "开启 · 高"),
    ("max", "开启 · 最高"),
)

_THINKING_LABELS = dict(THINKING_CHOICES)
_THINKING_LABELS["disabled"] = _THINKING_LABELS["off"]


def _explain_error(exc: Exception, thinking_sent: bool) -> Exception:
    """把接口返回的原始 400 翻译成能直接照做的中文。

    最常见的一种：换了第三方网关，对方不认 `thinking` 字段。原始报错是
    一长串 JSON，看不出该改哪里，这里直接点出配置项。
    """
    text = str(exc)
    if not thinking_sent or "400" not in text:
        return exc
    if "thinking" not in text.lower():
        return exc
    return RuntimeError(
        "当前接口不认识 `thinking` 参数（这是 DeepSeek 的原生字段，"
        "部分第三方网关不支持）。请把设置里的「思考模式」改成"
        "「服务端默认」，或直接改 config.yaml 的 `llm.thinking: auto`。\n"
        f"原始信息：{text[:300]}"
    )

SYSTEM_TEMPLATE = """你是候选人的「实时面试副驾」。候选人正在参加一场中文面试，\
你看到的是面试官刚刚说出的一句话。请立刻给出一份候选人可以照着念的回答。

硬性要求：
1. 第一句直接给结论或核心观点，不要寒暄、不要复述问题、不要写"好的"。
2. 然后用 2~4 个短要点展开，每个要点单独一行，一行不超过 30 字。
3. 如果问题涉及项目经历，优先使用【候选人简历】里的真实信息；简历里没有的细节，\
用「（这里补一个你项目里的具体例子）」这样的占位提示，绝对不要编造具体数字、公司名或时间。
4. 如果这句话其实不是提问（例如面试官在陈述或闲聊），只回一行：\
`（未识别到明确问题，可点"手动"重新提问）`。
5. 纯口语，像正常说话。不要用 Markdown 标题、加粗、代码块、编号列表以外的格式。
6. 回答总长度控制在 {max_chars} 字以内。

回答风格：{style}"""


class Answerer:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._client = None
        self._client_sig: tuple | None = None
        self._system_cache: str | None = None
        self._cancel = threading.Event()

    # ---------- 配置 ----------

    @property
    def _llm(self) -> dict:
        return self._cfg.get("llm", {})

    def _api_key(self) -> str:
        return (
            self._llm.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or ""
        ).strip()

    def _client_or_create(self):
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(
                "还没配置大模型 API Key。点右上角「设置」填写，"
                "或设置环境变量 OPENAI_API_KEY / DEEPSEEK_API_KEY。"
            )

        base_url = str(self._llm.get("base_url") or "").rstrip("/")
        if not base_url:
            raise RuntimeError("还没配置大模型接口地址（llm.base_url）。")
        timeout = float(self._llm.get("timeout") or 60)
        sig = (api_key, base_url, timeout)
        if self._client is None or self._client_sig != sig:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                # 自建网关常按 UA 拦截 OpenAI SDK，换成自己的标识
                default_headers={"User-Agent": USER_AGENT},
            )
            self._client_sig = sig
        return self._client

    def _system_prompt(self) -> str:
        if self._system_cache is not None:
            return self._system_cache
        llm = self._llm
        text = SYSTEM_TEMPLATE.format(
            max_chars=int(llm.get("max_chars") or 220),
            style=str(llm.get("style") or "口语化、条理清晰。"),
        )
        resume = str(llm.get("resume") or "").strip()
        if resume:
            text += f"\n\n【候选人简历 / 背景】\n{resume}"
        jd = str(llm.get("jd") or "").strip()
        if jd:
            text += f"\n\n【目标岗位 JD】\n{jd}"
        self._system_cache = text
        return text

    def invalidate(self) -> None:
        """配置变更后清掉缓存，下次调用重建。"""
        self._system_cache = None
        self._client_sig = None

    # ---------- 思考模式 ----------

    def thinking_mode(self) -> str:
        """当前生效的思考模式名。认不出的值按 auto 处理（不干预）。"""
        mode = str(self._llm.get("thinking") or "off").strip().lower()
        return mode if mode in _THINKING_MODES else "auto"

    def thinking_label(self) -> str:
        """给界面 / --check 用的人话描述。"""
        return _THINKING_LABELS[self.thinking_mode()]

    def _thinking_kwargs(self) -> dict:
        thinking, effort = _THINKING_MODES[self.thinking_mode()]
        kwargs: dict = {}
        if thinking is not None:
            kwargs["extra_body"] = {"thinking": thinking}
        if effort:
            kwargs["reasoning_effort"] = effort
        return kwargs

    # ---------- 生成 ----------

    def cancel(self) -> None:
        self._cancel.set()

    def warmup(self) -> None:
        """把首次提问前的固定开销提前做掉：import openai、建客户端、建立连接。

        只构造客户端是**不够的**——实测首个请求还要额外花约 1.8 秒做
        DNS+TLS+建连（第二个请求复用连接只要 0.25 秒）。所以这里真发一个
        最小请求，把连接也建起来。

        全程静默吞掉异常：没配 Key、断网都不该影响启动。
        真正的错误留到提问时再报，那里才有上下文。
        """
        try:
            client = self._client_or_create()
        except Exception:
            return
        try:
            client.chat.completions.create(
                model=str(self._llm.get("model") or "deepseek-chat"),
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                **self._thinking_kwargs(),
            )
        except Exception:
            pass

    def ping(self) -> str:
        """最小代价验证 Key / 地址 / 余额是否真的可用。面试前必跑。"""
        client = self._client_or_create()
        # 开着思考时思维链会吃掉大部分额度，给足才能拿到正文；
        # 关掉思考 256 绰绰有余。
        thinking_off = self.thinking_mode() in ("off", "disabled")
        extra = self._thinking_kwargs()
        try:
            response = client.chat.completions.create(
                model=str(self._llm.get("model") or "deepseek-chat"),
                messages=[{"role": "user", "content": "只回复两个字：就绪"}],
                # 推理模型（deepseek-flash / deepseek-reasoner 这类）会先花掉一部分
                # token 做思考。预算给太小，思考就把额度耗尽，content 是空的、
                # finish_reason=length —— 看起来像"调用成功但没回答"，其实是误报。
                max_tokens=256 if thinking_off else 2048,
                temperature=0,
                **extra,
            )
        except Exception as exc:
            raise _explain_error(exc, "extra_body" in extra) from None
        if not response.choices:
            raise RuntimeError("模型没有返回任何内容")
        text = (response.choices[0].message.content or "").strip()
        if not text:
            finish = response.choices[0].finish_reason
            raise RuntimeError(
                f"模型返回了空内容（finish_reason={finish}）。"
                "若开着思考模式，思维链会占用 llm.max_tokens——"
                "把 llm.thinking 设为 off，或把 llm.max_tokens 调大。"
            )
        return text

    def stream(self, question: str, history: list[dict] | None = None) -> Iterator[str]:
        """流式产出回答片段。调用方中途 break 即可停止生成。"""
        self._cancel.clear()
        client = self._client_or_create()

        messages: list[dict] = [{"role": "system", "content": self._system_prompt()}]
        for turn in (history or []):
            if turn.get("q"):
                messages.append({"role": "user", "content": turn["q"]})
            if turn.get("a"):
                messages.append({"role": "assistant", "content": turn["a"]})
        messages.append({"role": "user", "content": question})

        extra = self._thinking_kwargs()
        try:
            response = client.chat.completions.create(
                model=str(self._llm.get("model") or "deepseek-chat"),
                messages=messages,
                temperature=float(self._llm.get("temperature", 0.4)),
                max_tokens=int(self._llm.get("max_tokens") or 800),
                stream=True,
                **extra,
            )
        except Exception as exc:
            raise _explain_error(exc, "extra_body" in extra) from None
        for chunk in response:
            if self._cancel.is_set():
                break
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
