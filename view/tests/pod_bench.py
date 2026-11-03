"""Drive web/twin/view/ in headless Chromium on the pod: screenshots (default, autoplay, grid, help) and frame stats."""
import json, sys, time, os
from playwright.sync_api import sync_playwright

FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan", "--use-angle=vulkan",
         "--use-vulkan=native", "--ignore-gpu-blocklist", "--disable-vulkan-surface", "--enable-gpu",
         "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows",
         "--no-first-run", "--no-default-browser-check"]
base, out = sys.argv[1], sys.argv[2]
kinds = [k.replace("default", "") for k in sys.argv[3].split("|")]
only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else {"default", "autoplay", "drag", "grid", "help", "model"}
os.makedirs(out, exist_ok=True)
REPORT = "(() => { const r = window.twinView.report(); const t = r.timings; r.timings = t ? JSON.stringify(t).slice(0, 1500) : null; return r; })()"

def rate(page, secs):
    f0 = page.evaluate("window.twinView.disp.frames"); t0 = time.time()
    page.evaluate("window.twinView.resetPerf()")
    time.sleep(secs)
    f1 = page.evaluate("window.twinView.disp.frames"); t1 = time.time()
    rep = page.evaluate(REPORT)
    rep["fps_rendered"] = (f1 - f0) / (t1 - t0)
    return rep

results = {}
with sync_playwright() as pw:
    br = pw.chromium.launch(headless=True, args=FLAGS)
    try:
        for kind in kinds:
            dpr = 1
            if "@" in kind:
                kind, dpr = kind.split("@"); dpr = int(dpr)
            tag = (kind or "default") + (f"_dpr{dpr}" if dpr != 1 else "")
            ctx = br.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=dpr)
            page = ctx.new_page()
            logs = []
            page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"[:400]))
            page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"[:400]))
            url = f"{base}/view/" + ((kind if kind.startswith("?") else f"?renderer={kind}") if kind else "")
            if kind.startswith("?"): tag = "q_" + kind[1:].replace("=", "_").replace(",", "_").replace("&", "_")
            R = {"url": url}
            t0 = time.time()
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            try:
                page.wait_for_function("window.twinView && (window.twinView.disp.frames > 3 || window.twinView.problems.fatal)", timeout=120000, polling=200)
            except Exception as e:
                R["error"] = f"no frames: {e}"[:300]
            R["wall_to_frame3_s"] = time.time() - t0
            try:
                R["adapter"] = page.evaluate("navigator.gpu.requestAdapter().then(a => a ? (a.info ? {vendor: a.info.vendor, arch: a.info.architecture, desc: a.info.description} : 'no info') : null)")
            except Exception as e:
                R["adapter"] = str(e)[:200]
            if "error" not in R and not page.evaluate("!!window.twinView.problems.fatal"):
                time.sleep(2.0)
                R["still"] = rate(page, 3.0)
                page.screenshot(path=f"{out}/{tag}_default.png")
                if "autoplay" in only:
                    page.keyboard.press("a")
                    time.sleep(1.5)
                    R["autoplay"] = rate(page, 4.0)
                    page.screenshot(path=f"{out}/{tag}_autoplay.png")
                    page.keyboard.press("a")
                if "drag" in only:
                    # drag right in the main view with tool 1: measure the state change and the frame rate while moving
                    page.keyboard.press("1")
                    s0 = page.evaluate("JSON.stringify(window.twinView.state.instruments)")
                    box = page.locator("#main-view").bounding_box()
                    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                    page.mouse.move(cx, cy); page.mouse.down()
                    page.evaluate("window.twinView.resetPerf()")
                    f0 = page.evaluate("window.twinView.disp.frames"); t0 = time.time()
                    for i in range(60):
                        page.mouse.move(cx + 4 * i, cy, steps=1); time.sleep(1 / 60)
                    page.mouse.up()
                    f1 = page.evaluate("window.twinView.disp.frames"); t1 = time.time()
                    d = page.evaluate(REPORT)
                    d["fps_rendered"] = (f1 - f0) / (t1 - t0)
                    d["state_changed"] = page.evaluate("JSON.stringify(window.twinView.state.instruments)") != s0
                    R["drag"] = d
                    page.screenshot(path=f"{out}/{tag}_drag.png")
                if "grid" in only:
                    page.keyboard.press("v")
                    time.sleep(1.5)
                    page.keyboard.press("a")
                    time.sleep(1.0)
                    R["grid"] = rate(page, 3.0)
                    page.keyboard.press("a")
                    time.sleep(0.5)
                    page.screenshot(path=f"{out}/{tag}_grid.png")
                    page.keyboard.press("Escape")
                    time.sleep(0.3)
                if "model" in only:
                    import subprocess
                    mp = subprocess.Popen(["veridien", "webtwin", "model", "--null", "--url", base.replace("http", "ws") + "/ws/model"],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    try:
                        page.wait_for_function("window.twinView.feed && window.twinView.feed.attached", timeout=15000)
                        time.sleep(1.0)
                        page.keyboard.press("v")
                        time.sleep(4.0)
                        R["model"] = page.evaluate(REPORT)
                        page.keyboard.press("m")
                        time.sleep(0.6)
                        page.screenshot(path=f"{out}/{tag}_model_grid.png")
                        page.keyboard.press("Escape")
                    except Exception as e:
                        R["model_error"] = str(e)[:300]
                    finally:
                        mp.terminate(); mp.wait(5)
                if "help" in only:
                    page.click("#help-btn")
                    time.sleep(0.5)
                    page.screenshot(path=f"{out}/{tag}_help.png")
                    page.keyboard.press("Escape")
            else:
                page.screenshot(path=f"{out}/{tag}_default.png")
            R["final"] = page.evaluate(REPORT) if page.evaluate("!!window.twinView") else None
            R["logs"] = logs[-40:]
            results[tag] = R
            print(json.dumps({tag: {k: v for k, v in R.items() if k != "logs"}}, default=str)[:3000], flush=True)
            ctx.close()
            time.sleep(0.3)
    finally:
        br.close()
json.dump(results, open(f"{out}/bench.json", "w"), indent=1, default=str)
