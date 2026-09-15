"""讯飞语音听写（流式版）WebAPI。

接口规范：https://www.xfyun.cn/doc/asr/voicedictation/API.html
  - wss://iat-api.xfyun.cn/v2/iat
  - 鉴权：HMAC-SHA256 对 "host/date/request-line" 签名，结果做 Base64 后拼进 query
  - 音频：16k / 16bit / 单声道 / raw PCM，每帧 1280 字节（40ms），需 Base64
  - 单次会话最长 60s，10s 未发数据会被服务端断开

用法上我们不做流式展示（界面只关心最终文本），所以每段语音单独开一条连接、
把整段音频发完再取最终结果。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from email.utils import formatdate
from urllib.parse import urlencode, urlparse

import numpy as np

DEFAULT_ENDPOINT = "wss://iat-api.xfyun.cn/v2/iat"

# 1280 字节 = 640 个 16bit 采样 = 40ms @16k，官方建议值
FRAME_BYTES = 1280

# 单帧 base64 后不能超过 13000 字节
_MAX_B64_FRAME = 13000

# 官方限制单次会话 60 秒，留一点余量
_MAX_SESSION_SECONDS = 58.0


class XfyunError(RuntimeError):
    pass


def _explain_handshake_error(exc: Exception) -> str:
    """把裸的握手异常翻译成人能看懂的原因。

    凭证填错时 websocket-client 只会抛一大段 "Handshake status 401 ..."，
    对用户毫无帮助；这里按服务端返回的关键字给出可操作的提示。
    """
    text = str(exc)
    lowered = text.lower()

    if "apikey not found" in lowered:
        reason = "服务端找不到这个 APIKey：请确认复制完整，且与 APPID 属于同一个应用"
    elif "signature cannot be verified" in lowered:
        reason = "签名校验失败：APISecret 填错了"
    elif "401" in text or "unauthorized" in lowered:
        reason = "鉴权失败：请检查 APPID / APIKey / APISecret 是否填写正确且属于同一应用"
    elif "403" in text or "forbidden" in lowered:
        reason = "权限不足：该服务可能未开通，或免费额度已用尽"
    elif "timed out" in lowered or "timeout" in lowered:
        reason = "连接超时：请检查网络或代理设置"
    else:
        reason = "连接失败"

    return f"连接讯飞失败：{reason}。（原始信息：{text}）"


def build_auth_url(endpoint: str, api_key: str, api_secret: str, date: str | None = None) -> str:
    """按官方规范生成带签名的 WebSocket URL。

    签名原文顺序固定：host -> date -> request-line，且 ":" 后必须有一个空格。
    date 用 RFC1123 GMT 格式，服务端允许 ±300 秒时钟偏差。
    """
    parsed = urlparse(endpoint)
    host = parsed.netloc
    path = parsed.path or "/"
    stamp = date or formatdate(timeval=None, localtime=False, usegmt=True)

    signature_origin = f"host: {host}\ndate: {stamp}\nGET {path} HTTP/1.1"
    digest = hmac.new(
        api_secret.encode("utf-8"), signature_origin.encode("utf-8"), hashlib.sha256
    ).digest()
    signature = base64.b64encode(digest).decode("utf-8")

    authorization_origin = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")

    query = urlencode({"authorization": authorization, "date": stamp, "host": host})
    return f"{endpoint}?{query}"


def _join_words(payload: dict) -> str:
    """把 data.result.ws[].cw[].w 拼成一句话。"""
    parts: list[str] = []
    for item in payload.get("ws") or []:
        for word in item.get("cw") or []:
            parts.append(str(word.get("w") or ""))
    return "".join(parts)


class _ResultMerger:
    """合并分片结果。

    不开 dwa 时服务端只做追加，直接拼即可；
    开了 wpgs 会出现 pgs=rpl（替换某几片）的情况，需要按 rg 范围回退。
    """

    def __init__(self) -> None:
        self._parts: dict[int, str] = {}

    def add(self, sn: int, text: str, pgs: str | None, rg: list | None) -> None:
        if pgs == "rpl" and rg and len(rg) == 2:
            start, end = int(rg[0]), int(rg[1])
            for index in range(start, end + 1):
                self._parts.pop(index, None)
        if text:
            self._parts[sn] = text

    def text(self) -> str:
        return "".join(self._parts[key] for key in sorted(self._parts))


class XfyunTranscriber:
    """与 Transcriber（本地 whisper）保持相同接口，便于在 pipeline 里替换。"""

    def __init__(self, cfg: dict):
        x = cfg.get("xfyun") or {}
        self.app_id = str(x.get("app_id") or "").strip()
        self.api_key = str(x.get("api_key") or "").strip()
        self.api_secret = str(x.get("api_secret") or "").strip()
        self.endpoint = str(x.get("endpoint") or DEFAULT_ENDPOINT).strip()
        self.language = str(x.get("language") or "zh_cn")
        self.domain = str(x.get("domain") or "iat")
        self.accent = str(x.get("accent") or "mandarin")
        self.dwa = str(x.get("dwa") or "").strip()
        self.ptt = int(x.get("ptt", 1))
        # 注意不能用 `or 40`：0 是合法值（= 不节流、全速发送），
        # 但 `0 or 40` 会得到 40，把"全速"这个选项悄悄吃掉。
        interval = x.get("frame_interval_ms")
        self.frame_interval_ms = 40.0 if interval is None else float(interval)
        timeout = x.get("timeout")
        self.timeout = 15.0 if timeout is None else float(timeout)
        self.model_path = self.endpoint  # 供界面显示

    # ---------- 凭证 ----------

    def _credentials(self) -> tuple[str, str, str]:
        missing = [
            name
            for name, value in (
                ("app_id", self.app_id),
                ("api_key", self.api_key),
                ("api_secret", self.api_secret),
            )
            if not value
        ]
        if missing:
            raise XfyunError(
                "讯飞凭证不全，缺少：" + "、".join(missing)
                + "。请在「设置 → 语音识别」里填写，或设置环境变量 "
                "XFYUN_APP_ID / XFYUN_API_KEY / XFYUN_API_SECRET。"
            )
        return self.app_id, self.api_key, self.api_secret

    # ---------- 与本地后端一致的接口 ----------

    @property
    def loaded(self) -> bool:
        return bool(self.app_id and self.api_key and self.api_secret)

    def load(self, on_status=None, on_progress=None) -> None:
        """讯飞是云端服务，没有模型要下载；这里只做凭证校验。"""
        self._credentials()
        if on_status:
            on_status("讯飞语音听写已就绪（云端识别，无需下载模型）")

    def verify(self) -> str:
        """真连一次，验证凭证和服务开通状态。给 --check 用。"""
        text = self.transcribe(np.zeros(int(16000 * 0.4), dtype=np.float32))
        return text or "（静音，无识别结果，但连接与鉴权正常）"

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""

        app_id, api_key, api_secret = self._credentials()

        seconds = audio.size / 16000.0
        if seconds > _MAX_SESSION_SECONDS:
            raise XfyunError(
                f"这段语音 {seconds:.0f} 秒，超过讯飞单次会话 60 秒上限。"
                "请把 config.yaml 里的 audio.max_speech_ms 调到 58000 以内。"
            )

        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        if not pcm:
            return ""

        try:
            import websocket  # websocket-client
        except Exception as exc:  # pragma: no cover
            raise XfyunError("缺少 websocket-client，请先 pip install websocket-client") from exc

        url = build_auth_url(self.endpoint, api_key, api_secret)
        try:
            ws = websocket.create_connection(url, timeout=self.timeout)
        except Exception as exc:
            raise XfyunError(_explain_handshake_error(exc)) from exc

        merger = _ResultMerger()
        try:
            self._send_audio(ws, pcm, app_id)
            return self._recv_result(ws, merger)
        finally:
            try:
                ws.close()
            except Exception:
                pass

    # ---------- 内部 ----------

    def _business(self) -> dict:
        business = {
            "language": self.language,
            "domain": self.domain,
            "accent": self.accent,
            "ptt": self.ptt,
            "vinfo": 1,
        }
        if self.dwa:
            # 注意：dwa 与 vinfo 同时设置时 vinfo 会失效
            business["dwa"] = self.dwa
            business.pop("vinfo", None)
        return business

    def _send_audio(self, ws, pcm: bytes, app_id: str) -> None:
        interval = max(0.0, self.frame_interval_ms / 1000.0)
        offset = 0
        status = 0
        total = len(pcm)

        while True:
            buf = pcm[offset : offset + FRAME_BYTES]
            offset += len(buf)
            if not buf:
                status = 2  # 数据结束标识

            frame: dict = {
                "data": {
                    "status": status,
                    "format": "audio/L16;rate=16000",
                    "encoding": "raw",
                    "audio": base64.b64encode(buf).decode("utf-8"),
                }
            }
            if status == 0:
                # common / business 只在第一帧上传
                frame["common"] = {"app_id": app_id}
                frame["business"] = self._business()

            payload = json.dumps(frame)
            if len(payload) > _MAX_B64_FRAME + 512:
                raise XfyunError("单帧音频过大，请调小 FRAME_BYTES")
            ws.send(payload)

            if status == 2:
                return
            status = 1
            if interval:
                time.sleep(interval)

    def _recv_result(self, ws, merger: _ResultMerger) -> str:
        while True:
            try:
                raw = ws.recv()
            except Exception as exc:
                raise XfyunError(f"接收讯飞结果失败：{exc}") from exc
            if not raw:
                break

            message = json.loads(raw)
            code = message.get("code")
            if code != 0:
                raise XfyunError(f"讯飞返回错误 code={code}：{message.get('message')}")

            data = message.get("data") or {}
            result = data.get("result") or {}
            if result:
                merger.add(
                    int(result.get("sn") or 0),
                    _join_words(result),
                    result.get("pgs"),
                    result.get("rg"),
                )
            if data.get("status") == 2:
                break

        return merger.text().strip()
