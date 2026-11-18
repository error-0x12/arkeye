"""Serve the trace viewer, with optional FV player linkage.

The viewer works standalone: open ``index.html`` and pick a ``trace.jsonl``.
This launcher adds two things:

1. ``?trace=`` auto-loading, so no file dialog is needed.
2. An FV proxy at ``/fv/*`` that lets the page drive / follow the FV player.

The FV HTTP API cannot be called from a browser page directly: it rejects any
request carrying an ``Origin`` header and serves no CORS, and its bearer token
must not be put in a URL. So the token stays in this process and the page only
talks to same-origin ``/fv/*`` routes. Only loopback endpoints are accepted and
only a fixed set of upstream paths is reachable.

    python tools/trace_viewer/serve.py output/1_8_hud_weighted_fusion/trace.jsonl

Start FV itself with ``--http`` for the linkage to work:

    FV.exe --http "1-8实战.mp4"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections import defaultdict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlsplit

VIEWER_DIR = Path(__file__).resolve().parent
TRACE_MOUNT = "/trace.jsonl"
VIDEO_MOUNT = "/video.mp4"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
FV_TIMEOUT = 3.0


def default_discovery_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return base / "FV" / "remote"


class FvError(RuntimeError):
    """Raised when the FV player cannot be reached or answered badly."""


@dataclass(frozen=True)
class FvEndpoint:
    pid: int
    base_url: str
    token: str


def load_endpoints(directory: Path) -> list[FvEndpoint]:
    """Read ``fv-<PID>.json`` discovery files, keeping usable loopback entries only."""

    endpoints: list[FvEndpoint] = []
    if not directory.is_dir():
        return endpoints
    for path in sorted(directory.glob("fv-*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or value.get("protocolVersion") != 1:
            continue
        base_url = value.get("baseUrl")
        token = value.get("token")
        if not isinstance(base_url, str) or not isinstance(token, str) or not token:
            continue
        if urlsplit(base_url).hostname not in LOOPBACK_HOSTS:
            continue
        try:
            pid = int(value.get("pid"))
        except (TypeError, ValueError):
            continue
        endpoints.append(FvEndpoint(pid, base_url if base_url.endswith("/") else f"{base_url}/", token))
    return endpoints


class FvProxy:
    """Talk to one FV instance, caching the discovered endpoint for a few seconds."""

    def __init__(self, discovery_dir: Path, *, timeout: float = FV_TIMEOUT, cache_seconds: float = 5.0) -> None:
        self.discovery_dir = discovery_dir
        self.timeout = timeout
        self.cache_seconds = cache_seconds
        self._lock = threading.Lock()
        self._endpoint: FvEndpoint | None = None
        self._checked_at = 0.0

    # -- discovery ---------------------------------------------------------

    def endpoint(self, *, refresh: bool = False) -> FvEndpoint:
        with self._lock:
            if not refresh and self._endpoint is not None and time.monotonic() - self._checked_at < self.cache_seconds:
                return self._endpoint
            candidates = load_endpoints(self.discovery_dir)
            if not candidates:
                self._endpoint = None
                self._checked_at = time.monotonic()
                raise FvError("未发现 FV 接口：请用 FV.exe --http 启动播放器")
            for candidate in reversed(candidates):
                try:
                    self._request(candidate, "GET", "v1/state")
                except FvError:
                    continue
                self._endpoint = candidate
                self._checked_at = time.monotonic()
                return candidate
            self._endpoint = None
            self._checked_at = time.monotonic()
            raise FvError("发现文件存在，但 FV 没有响应（可能已退出或换了端口）")

    # -- transport ---------------------------------------------------------

    def _request(self, endpoint: FvEndpoint, method: str, path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            endpoint.base_url + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {endpoint.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise FvError(f"FV 返回 HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise FvError(f"无法连接 FV: {exc}") from exc
        if not raw.strip():
            return {}
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FvError("FV 返回的不是 JSON") from exc
        return value if isinstance(value, dict) else {"value": value}

    # -- operations used by the page --------------------------------------

    def state(self) -> dict:
        return self._request(self.endpoint(), "GET", "v1/state")

    def clips(self) -> dict:
        return self._request(self.endpoint(), "GET", "v1/clips")

    def seek(self, position_seconds: float) -> dict:
        return self._request(self.endpoint(), "POST", "v1/playback/seek", {"positionSeconds": position_seconds})

    def set_playing(self, playing: bool) -> dict:
        return self._request(self.endpoint(), "PUT", "v1/playback", {"playing": playing})

    def info(self, *, refresh: bool = False) -> dict:
        """Connection summary for the page. Never contains the token."""

        try:
            endpoint = self.endpoint(refresh=refresh)
            state = self.state()
        except FvError as exc:
            return {"available": False, "error": str(exc)}
        return {
            "available": True,
            "pid": endpoint.pid,
            "baseUrl": endpoint.base_url,
            "state": state,
        }


LEVEL_DIR = Path(".cache/levels/obt/main")
CONFIG_DIR = Path("configs")
ENEMY_DB = Path(".cache/levels/enemydata/enemy_database.json")
# 同一关里最高血量超过第二名的这个倍数，才认为它是 Boss
BOSS_HP_RATIO = 1.8


def normalize_stage_id(name: object) -> str | None:
    """把 trace 里的 ``1-8`` 归一成 ``main_01-08``。"""

    if not isinstance(name, str) or not name.strip():
        return None
    text = name.strip()
    if text.startswith(("main_", "sub_")):
        return text
    parts = text.replace("_", "-").split("-")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return f"main_{int(parts[0]):02d}-{int(parts[1]):02d}"
    return None


def _route_map_points(route: dict) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def append(position: object) -> None:
        if not isinstance(position, dict) or "col" not in position or "row" not in position:
            return
        point = float(position["col"]), float(position["row"])
        if not points or point != points[-1]:
            points.append(point)

    append(route.get("startPosition"))
    for checkpoint in route.get("checkpoints") or []:
        if checkpoint.get("type") == "MOVE":
            append(checkpoint.get("position"))
    append(route.get("endPosition"))
    return points


def _class_max_hp(enemy_ids: set[str]) -> dict[str, float]:
    try:
        data = json.loads(ENEMY_DB.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    result: dict[str, float] = {}
    for entry in data.get("enemies") or []:
        key = str(entry.get("Key") or "")
        if key not in enemy_ids:
            continue
        for value in entry.get("Value") or []:
            attributes = (value.get("enemyData") or {}).get("attributes") or {}
            hp = (attributes.get("maxHp") or {}).get("m_value")
            if isinstance(hp, (int, float)):
                result[key] = max(result.get(key, 0.0), float(hp))
    return result


def pick_boss(hp: dict[str, float]) -> str | None:
    """血量明显离群的那一类当作 Boss，避免把角色名写死。"""

    if len(hp) < 2:
        return None
    ranked = sorted(hp.items(), key=lambda item: item[1], reverse=True)
    (top, top_hp), (_, second_hp) = ranked[0], ranked[1]
    if second_hp <= 0:
        return None
    return top if top_hp >= second_hp * BOSS_HP_RATIO else None


def build_route_payload(stage_id: str | None) -> dict:
    """把关卡路线投影到战斗帧坐标，附带普通/飞行/Boss 分类。"""

    if not stage_id:
        return {"available": False, "error": "trace 里没有关卡信息，无法加载路线"}
    level_path = LEVEL_DIR / f"level_{stage_id}.json"
    config_path = CONFIG_DIR / f"{stage_id}.json"
    if not level_path.is_file() or not config_path.is_file():
        return {"available": False, "error": f"缺少 {level_path.name} 或 {config_path.name}"}
    try:
        from arkeye.board import BoardCalibration
        from arkeye.timeline import build_stage_timeline
    except Exception as exc:  # noqa: BLE001 - 没装 arkeye 时退化成不可用
        return {"available": False, "error": f"无法导入 arkeye：{exc}"}

    level = json.loads(level_path.read_text(encoding="utf-8"))
    calibration = BoardCalibration.load(config_path)
    timeline = build_stage_timeline(level)

    used: dict[int, list[str]] = defaultdict(list)
    for enemy_id, indices in timeline.class_route_indices.items():
        for index in indices:
            used[index].append(enemy_id)

    boss = pick_boss(_class_max_hp(set(timeline.class_route_indices)))
    level_routes = level.get("routes") or []
    routes = []
    for index in sorted(used):
        if index >= len(level_routes):
            continue
        map_points = _route_map_points(level_routes[index])
        if len(map_points) < 2:
            continue
        projected = calibration.map_points_to_frame(map_points)
        mode = str(level_routes[index].get("motionMode") or "").upper()
        classes = sorted(used[index])
        kind = "boss" if boss is not None and boss in classes else ("fly" if mode == "FLY" else "normal")
        routes.append(
            {
                "index": index,
                "mode": mode,
                "kind": kind,
                "classes": classes,
                "points": [[round(float(x), 1), round(float(y), 1)] for x, y in projected],
            }
        )
    return {
        "available": True,
        "stage": stage_id,
        "frame": [1920, 1080],
        "boss": boss,
        "routes": routes,
    }


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    try:
        length = int(handler.headers.get("Content-Length") or 0)
    except ValueError as exc:
        raise FvError("无效的 Content-Length") from exc
    if length <= 0:
        raise FvError("缺少请求体")
    raw = handler.rfile.read(length)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FvError("请求体不是合法 JSON") from exc
    if not isinstance(value, dict):
        raise FvError("请求体必须是 JSON 对象")
    return value


def make_handler(
    trace_payload: bytes,
    fv: FvProxy | None,
    route_payload: dict | None = None,
    video_path: Path | None = None,
):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ArkEyeTraceViewer/2"

        # -- routing -------------------------------------------------------

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in {"/", "/index.html"}:
                self._send_bytes((VIEWER_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == TRACE_MOUNT:
                self._send_bytes(trace_payload, "application/x-ndjson; charset=utf-8")
            elif path == "/routes.json":
                self._send_json(route_payload or {"available": False, "error": "未加载路线数据"})
            elif path == VIDEO_MOUNT:
                if video_path is None:
                    self.send_error(404, "本服务未加载视频（加 --video <文件> 启动）")
                else:
                    self._send_video(video_path)
            elif path == "/fv/info":
                refresh = "refresh=1" in (urlsplit(self.path).query or "")
                self._run_tool(lambda: (fv.info(refresh=refresh) if fv else {"available": False, "error": "本服务以 --no-fv 启动"}))
            elif path == "/fv/state":
                self._run_tool(lambda: fv.state() if fv else self._no_fv())
            elif path == "/fv/clips":
                self._run_tool(lambda: fv.clips() if fv else self._no_fv())
            else:
                self.send_error(404, "only /, /trace.jsonl, /routes.json and /fv/* are served")

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path == "/fv/seek":
                self._run_tool(lambda: self._seek())
            elif path == "/fv/playback":
                self._run_tool(lambda: self._playback())
            else:
                self.send_error(404, "unknown endpoint")

        def do_PUT(self) -> None:
            if urlsplit(self.path).path == "/fv/playback":
                self._run_tool(lambda: self._playback())
            else:
                self.send_error(404, "unknown endpoint")

        # -- operations ----------------------------------------------------

        @staticmethod
        def _no_fv() -> dict:
            raise FvError("本服务以 --no-fv 启动，未连接 FV")

        def _seek(self) -> dict:
            if fv is None:
                return self._no_fv()
            body = _read_json_body(self)
            raw = body.get("positionSeconds")
            # 只接受真正的数字：bool 和 "12.5" 这类字符串都算非法参数
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise FvError("positionSeconds 必须是数字")
            position = float(raw)
            if not position >= 0 or not math.isfinite(position):
                raise FvError("positionSeconds 必须是有限的非负数")
            return fv.seek(position)

        def _playback(self) -> dict:
            if fv is None:
                return self._no_fv()
            body = _read_json_body(self)
            playing = body.get("playing")
            if not isinstance(playing, bool):
                raise FvError("playing 必须是布尔值")
            return fv.set_playing(playing)

        # -- response helpers ----------------------------------------------

        def _run_tool(self, action) -> None:
            try:
                payload = action()
            except FvError as exc:
                self._send_json({"error": str(exc)}, status=502)
                return
            self._send_json(payload)

        def _send_json(self, payload: dict, *, status: int = 200) -> None:
            self._send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def _send_video(self, path: Path) -> None:
            """发送视频，支持 Range。

            <video> 拖动进度时会发 Range 请求；一次性返回整段也能播，
            但 seek 会退化成重新下载，按需求返回 206 才流畅。
            """

            size = path.stat().st_size
            header = self.headers.get("Range")
            start, end = 0, size - 1
            status = 200
            if header and header.startswith("bytes="):
                raw = header[len("bytes=") :].split(",")[0].strip()
                first, _, last = raw.partition("-")
                try:
                    if first:
                        start = max(0, int(first))
                        end = min(size - 1, int(last)) if last else size - 1
                    elif last:
                        start = max(0, size - int(last))
                        end = size - 1
                except ValueError:
                    start, end = 0, size - 1
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = 206
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = stream.read(min(1024 * 256, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the ArkEye trace viewer with one trace file loaded")
    parser.add_argument("trace", type=Path, help="trace.jsonl produced by render_current.py --trace")
    parser.add_argument("--port", type=int, default=8712, help="listen port on 127.0.0.1 (default 8712)")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    parser.add_argument("--no-fv", action="store_true", help="disable the FV player proxy")
    parser.add_argument("--fv-discovery-dir", type=Path, help="override the FV discovery directory")
    parser.add_argument("--stage", help="stage id for route drawing, e.g. main_01-08 (default: read from the trace)")
    parser.add_argument("--video", type=Path, help="mp4 to serve at /video.mp4 so the page can draw real frames")
    args = parser.parse_args()

    trace_path = args.trace.resolve()
    if not trace_path.is_file():
        raise SystemExit(f"trace file not found: {trace_path}")

    payload = trace_path.read_bytes()

    stage_id = normalize_stage_id(args.stage)
    if stage_id is None:
        for raw in payload.splitlines()[:5]:
            if not raw.strip():
                continue
            try:
                stage_id = normalize_stage_id(json.loads(raw.decode("utf-8")).get("stage"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                stage_id = None
            break
    routes = build_route_payload(stage_id)

    fv = None if args.no_fv else FvProxy(args.fv_discovery_dir or default_discovery_dir())
    video_path = args.video.resolve() if args.video is not None else None
    if video_path is not None and not video_path.is_file():
        raise SystemExit(f"video file not found: {video_path}")
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(payload, fv, routes, video_path)
    )
    query = f"trace={quote(TRACE_MOUNT)}"
    if video_path is not None:
        query += f"&video={quote(VIDEO_MOUNT)}"
    url = f"http://127.0.0.1:{server.server_address[1]}/?{query}"
    size_mb = trace_path.stat().st_size / 1024 / 1024
    print(f"trace : {trace_path} ({size_mb:.1f} MB)")
    if video_path is None:
        print("video : 未提供（加 --video <mp4> 可在页面里显示真实画面）")
    else:
        print(f"video : {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")
    if fv is None:
        print("fv    : proxy disabled (--no-fv)")
    else:
        status = fv.info()
        if status.get("available"):
            print(f"fv    : connected  pid={status['pid']} {status['baseUrl']}")
        else:
            print(f"fv    : not connected - {status.get('error')}")
            print("        start the player with: FV.exe --http \"1-8实战.mp4\"")
    if routes.get("available"):
        kinds = {"normal": 0, "fly": 0, "boss": 0}
        for route in routes["routes"]:
            kinds[route["kind"]] = kinds.get(route["kind"], 0) + 1
        print(
            f"routes: {len(routes['routes'])} 条  普通 {kinds['normal']} / 飞行 {kinds['fly']} / Boss {kinds['boss']}"
            f"  (boss={routes.get('boss')})"
        )
    else:
        print(f"routes: 不可用 - {routes.get('error')}")
    print(f"viewer: {url}")
    print("Ctrl+C to stop")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
