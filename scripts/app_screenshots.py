"""Drive the web app in a real (headless) Chrome: an end-to-end check of the page and the screenshots of the report (Week 8).

Starts the app on a free port, starts Chrome headless with the remote-debugging protocol (no extra package: a minimal WebSocket
client is below), and for each language: loads the page, uploads a photo, presses Paint, and saves screenshots of the page
before, during and after the painting, of the before / after slider, of the replay, of the time-lapse video (Week 9: Chrome must
play the MP4, and the five download buttons must be enabled) and of a refused file. The English
overview is also saved with numbered markers on the controls (``app_overview_annotated.png``), and the result column of the
three English stages side by side (``app_flow.png``). Fails, with the step it was at, when the page does
not behave (no live preview, no result, no error message).

Usage: venv/Scripts/python.exe scripts/app_screenshots.py [--photo data/eval_set/apple.png] [--out report/figures] [--languages en vi]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CHROME_CANDIDATES = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                     r"C:\Program Files\Microsoft\Edge\Application\msedge.exe", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
WIDTH, HEIGHT = 1400, 1000


class WebSocket:
    """Just enough RFC 6455 for the DevTools protocol: text frames out (masked), text frames in (any length)."""

    def __init__(self, url: str) -> None:
        u = urllib.parse.urlparse(url)
        self.sock = socket.create_connection((u.hostname, u.port), timeout=120)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        reply = b""
        while b"\r\n\r\n" not in reply:
            reply += self.sock.recv(1024)
        if b" 101 " not in reply.split(b"\r\n", 1)[0]:
            raise ConnectionError(reply.decode(errors="replace"))

    def _read(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(min(n - len(buf), 1 << 20))
            if not chunk:
                raise ConnectionError("the browser closed the connection")
            buf += chunk
        return buf

    def send(self, text: str) -> None:
        data, mask = text.encode(), os.urandom(4)
        n = len(data)
        head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + n.to_bytes(2, "big") if n < 65536
                                 else bytes([0x80 | 127]) + n.to_bytes(8, "big"))
        self.sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def recv(self) -> str:
        message = b""
        while True:
            b0, b1 = self._read(2)
            n = b1 & 0x7F
            n = int.from_bytes(self._read(2 if n == 126 else 8), "big") if n >= 126 else n
            data = self._read(n)
            if b0 & 0x0F == 8:
                raise ConnectionError("closed")
            if b0 & 0x0F in (9, 10):  # ping / pong
                continue
            message += data
            if b0 & 0x80:
                return message.decode()


class Page:
    """The DevTools protocol on one page: calls, JavaScript, waits, screenshots."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws, self.next_id = ws, 0

    def call(self, method: str, **params):
        self.next_id += 1
        self.ws.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.next_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expression: str):
        result = self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            raise RuntimeError(f"{expression[:200]}: {details.get('exception', {}).get('description') or details.get('text')}")
        return result["result"].get("value")

    def wait(self, expression: str, what: str, timeout: float = 60.0):
        end = time.time() + timeout
        while time.time() < end:
            value = self.js(expression)
            if value:
                return value
            time.sleep(0.25)
        raise TimeoutError(f"waited {timeout:.0f} s for {what}")

    def screenshot(self, path: Path, full: bool = True) -> None:
        height = HEIGHT
        if full:
            height = max(HEIGHT, int(self.call("Page.getLayoutMetrics")["cssContentSize"]["height"]))
        self.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=height, deviceScaleFactor=1, mobile=False)
        time.sleep(0.5)
        path.write_bytes(base64.b64decode(self.call("Page.captureScreenshot", format="png")["data"]))
        self.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=1, mobile=False)

    def upload(self, selector: str, path: Path) -> None:
        root = self.call("DOM.getDocument", depth=-1, pierce=True)["root"]["nodeId"]
        node = self.call("DOM.querySelector", nodeId=root, selector=selector)["nodeId"]
        if not node:
            raise RuntimeError(f"no element {selector}")
        self.call("DOM.setFileInputFiles", files=[str(path.resolve())], nodeId=node)

    def click_text(self, tag: str, text: str) -> None:
        """Click the first ``tag`` element whose visible text is ``text`` (buttons, radio labels, tabs)."""
        point = self.js(f"""(() => {{ const el = [...document.querySelectorAll({json.dumps(tag)})]
            .find(e => e.innerText.trim() === {json.dumps(text)} && e.offsetParent !== null);
            if (!el) return null; el.scrollIntoView({{block: 'center'}}); const r = el.getBoundingClientRect();
            return [r.x + r.width / 2, r.y + r.height / 2]; }})()""")
        if not point:
            raise RuntimeError(f"no {tag} with the text {text!r}")
        for kind in ("mousePressed", "mouseReleased"):  # a real click, as a user's mouse makes it
            self.call("Input.dispatchMouseEvent", type=kind, x=point[0], y=point[1], button="left", clickCount=1)

    def box(self, expression: str):
        """The page box ``(x, y, w, h)`` of the element an expression selects."""
        return self.js(f"(() => {{ const e = {expression}; if (!e) return null; const r = e.getBoundingClientRect();"
                       " return [r.x + scrollX, r.y + scrollY, r.width, r.height]; })()")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 120.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{url} did not answer")


def start_app(port: int) -> subprocess.Popen:
    code = ("from neural_painter.app import gradio_app as a; a.disable_power_throttling(); "
            f"a.build_app().queue(default_concurrency_limit=1).launch(server_port={port}, max_file_size=a.MAX_UPLOAD, quiet=True)")
    return subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_chrome(language: str, profile: str) -> tuple[subprocess.Popen, int]:
    chrome = next((c for c in CHROME_CANDIDATES if Path(c).is_file()), None)
    if chrome is None:
        raise SystemExit("no Chrome or Edge found")
    port = free_port()
    proc = subprocess.Popen([chrome, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}", f"--lang={language}",
                             f"--accept-lang={language}", f"--window-size={WIDTH},{HEIGHT}", "--hide-scrollbars", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_http(f"http://127.0.0.1:{port}/json/version")
    return proc, port


def annotate(shot: Path, boxes: list[tuple[str, list[float]]], out: Path) -> None:
    """The screenshot with a numbered marker at the top left of each box (the caption explains the numbers)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img = plt.imread(shot)
    fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    for k, (_, (x, y, w, h)) in enumerate(boxes, start=1):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#d81b60", linewidth=2.5))
        ax.text(x - 4, y - 4, str(k), color="white", fontsize=15, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.3", facecolor="#d81b60", edgecolor="white"))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def flow_figure(out: Path, tag: str, titles: list[str]) -> None:
    """The result column of three screenshots side by side: the live preview, the before / after slider, the replay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shots = [plt.imread(out / f"app_{tag}_{name}.png") for name in ("painting", "compare", "replay")]
    crops = [img[90:800, 500:1315] for img in shots]  # the right-hand column of the 1400 px wide page
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, crop, title in zip(axes, crops, titles):
        ax.imshow(crop)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.savefig(out / "app_flow.png", dpi=130)
    plt.close(fig)


def run(language: str, photo: Path, out: Path, app_url: str, junk: Path) -> dict:
    from neural_painter.app.gradio_app import LANGUAGES, SAMPLES_LABEL, TEXT

    words = TEXT["vi" if language.startswith("vi") else "en"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        chrome, port = start_chrome(language, profile)
        try:
            target = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
            page = Page(WebSocket(next(t for t in target if t["type"] == "page")["webSocketDebuggerUrl"]))
            page.call("Page.enable")
            page.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=1, mobile=False)
            page.call("Page.navigate", url=app_url)
            page.wait(f"[...document.querySelectorAll('button')].some(b => b.innerText.trim() === {json.dumps(words['paint'])})",
                      "the page in the browser's language")
            tag = "en" if language.startswith("en") else "vi"
            page.screenshot(out / f"app_{tag}_start.png")
            names = {code: name for name, code in LANGUAGES}
            for lang_code in ("en" if tag == "vi" else "vi", tag):  # the language switch relabels the page, there and back
                page.click_text("label", names[lang_code])
                page.wait(f"[...document.querySelectorAll('button')].some(b => b.innerText.trim() === {json.dumps(TEXT[lang_code]['paint'])})",
                          f"the page in {names[lang_code]} after the switch", 30)
            page.upload("input[type=file]", photo)
            page.wait("document.querySelector('img') !== null && [...document.querySelectorAll('img')].some(i => i.src.includes('/file='))",
                      "the uploaded photo's preview")
            page.click_text("label", words["watercolor"])
            # a Gradio component is a .block whose text starts with its label
            block = "[...document.querySelectorAll('.block')].find(b => b.innerText.trim().split('\\n')[0].trim().startsWith({}))"
            marks = [("photo", words["photo"]), ("samples", SAMPLES_LABEL), ("brush", words["brush"]), ("strokes", words["strokes"]),
                     ("mode", words["mode"]), ("speed", words["speed"]), ("status", words["idle"].replace("**", "")[:12]),
                     ("live", words["live"])]
            boxes = [(name, page.box(block.format(json.dumps(text)))) for name, text in marks]
            paint_button = f"[...document.querySelectorAll('button')].find(e => e.innerText.trim() === {json.dumps(words['paint'])})"
            boxes.insert(6, ("paint", page.box(paint_button)))
            page.screenshot(out / f"app_{tag}_ready.png")
            page.click_text("button", words["paint"])
            running = words["running"].split(":")[0]
            page.wait(f"document.body.innerText.includes({json.dumps(running)})", "the live progress line", 120)
            time.sleep(4)
            page.screenshot(out / f"app_{tag}_painting.png")
            done = words["done"].split(":")[0]
            page.wait(f"document.body.innerText.includes({json.dumps(done + ':')})", "the finished painting", 600)
            selected = "[...document.querySelectorAll('button[role=tab]')].find(b => b.innerText.trim() === {})?.getAttribute('aria-selected') === 'true'"
            page.wait(selected.format(json.dumps(words["compare"])), "the page switching to the photo and the painting", 15)
            time.sleep(1.5)
            page.screenshot(out / f"app_{tag}_compare.png")
            page.click_text("button[role=tab]", words["replay"])
            time.sleep(1)
            point = page.js("""(() => { const r = [...document.querySelectorAll('input[type=range]')].filter(e => e.offsetParent !== null).pop();
                r.scrollIntoView({block: 'center'}); const b = r.getBoundingClientRect(); return [b.x + b.width * 0.35, b.y + b.height / 2]; })()""")
            for kind in ("mousePressed", "mouseReleased"):  # a third of the way along the replay slider
                page.call("Input.dispatchMouseEvent", type=kind, x=point[0], y=point[1], button="left", clickCount=1)
            time.sleep(1.5)
            page.screenshot(out / f"app_{tag}_replay.png")
            replay_at = page.js("[...document.querySelectorAll('input[type=range]')].filter(e => e.offsetParent !== null).pop().value")
            page.wait(f"!document.body.innerText.includes({json.dumps(words['making'][:15])})", "the time-lapse video", 120)
            page.click_text("button[role=tab]", words["timelapse"])
            page.wait("(() => { const v = document.querySelector('video'); return v !== null && v.readyState >= 2 && v.currentTime > 0.5; })()",
                      "the time-lapse playing in the browser", 30)
            time.sleep(2)
            page.screenshot(out / f"app_{tag}_timelapse.png")
            page.click_text("button[role=tab]", words["painting"])  # opened only now: Gradio once blanked an image in an unopened tab
            page.wait("[...document.querySelectorAll('img')].some(i => i.offsetParent !== null && i.naturalWidth >= 256 && i.clientWidth >= 300)",
                      "the painting in the Painting tab", 15)
            labels = [words[k] for k in ("download_png", "download_jpg", "download_mp4", "download_gif", "download_npz")]
            buttons = page.js(f"[...document.querySelectorAll('button')].filter(b => {json.dumps(labels)}.includes(b.innerText.trim()))"
                              ".map(b => b.innerText.trim() + (b.disabled ? ':off' : ':on'))")
            if sorted(buttons) != sorted(f"{text}:on" for text in labels):
                raise RuntimeError(f"the download buttons are not all enabled: {buttons}")
            status_text = page.js(f"document.body.innerText.split('\\n').find(l => l.startsWith({json.dumps(done)}))")
            page.upload("input[type=file]", junk)
            time.sleep(1.5)
            page.click_text("button", words["paint"])
            page.wait(f"document.body.innerText.includes({json.dumps(words['not_image'][:20])})", "the refusal of a damaged file", 60)
            time.sleep(0.5)
            page.screenshot(out / f"app_{tag}_error.png", full=False)
            return {"language": language, "boxes": [b for b in boxes if b[1]], "status": status_text, "replay_at": replay_at}
        finally:
            chrome.kill()
            chrome.wait(10)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--photo", type=Path, default=ROOT / "data" / "eval_set" / "apple.png")
    ap.add_argument("--out", type=Path, default=ROOT / "report" / "figures")
    ap.add_argument("--languages", nargs="+", default=["en-US", "vi-VN"])
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the Vietnamese status line, printed into a pipe on Windows
    args.out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    app = start_app(port)
    try:
        wait_http(f"http://127.0.0.1:{port}/")
        with tempfile.TemporaryDirectory() as tmp:
            junk = Path(tmp) / "damaged.png"
            junk.write_bytes(os.urandom(4000))
            for language in args.languages:
                result = run(language, args.photo, args.out, f"http://127.0.0.1:{port}/?__theme=light", junk)  # light pages print better
                print(f"{language}: {result['status']}", flush=True)
                if language.startswith("en"):
                    annotate(args.out / "app_en_ready.png", result["boxes"], args.out / "app_overview_annotated.png")
                    flow_figure(args.out, "en", ["(a) painting: the live preview", "(b) done: photo and painting",
                                                 f"(c) replay at {result['replay_at']} of 495 strokes"])
                    print("markers: " + ", ".join(f"{k} {name}" for k, (name, _) in enumerate(result["boxes"], start=1)))
    finally:
        app.terminate()
    print(f"wrote the screenshots to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
