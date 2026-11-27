"""AegisOT demo UI (Gradio): try the ML model alone, or the full system, like a real user would.

    pip install gradio
    # terminal 1 - the system (ML + agents, agents read the SAME recording the model scores)
    BACKEND_MODE=csv BACKEND_CSV=data/processed/final.csv uvicorn src.api.main:app --port 8001
    # terminal 2 - this UI
    python scripts/demo_ui.py --key dev-key            # then open http://localhost:7860

The UI is only a CLIENT of the real API (nothing is faked): every button sends HTTP requests to the running server.
  Tab 1  ML model only      recorded window  ->  POST /api/v1/anomaly/window
  Tab 2  Full system        recorded window  ->  ML  ->  POST /api/v1/incident/analyze  (agents + Gemini)
  Tab 3  What-if reading    edit one sensor reading by hand  ->  ML only (POST /api/v1/anomaly/predict)
The recording's label (`condition`) is used to PICK rows, and is never sent to the system.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import time
from pathlib import Path
from turtle import title

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAP_S = 2.0
CFG = {"api": "http://localhost:8001", "key": "dev-key", "csv": str(ROOT / "data" / "processed" / "final.csv")}
_DF: dict[str, pd.DataFrame] = {}
SEV_COLOR = {"low": "#2e9e5b", "medium": "#d9a400", "high": "#e8710a", "critical": "#d93025"}


# ───────────────────────── data ─────────────────────────
def recording() -> pd.DataFrame:
    if "df" not in _DF:
        df = pd.read_csv(CFG["csv"])
        df["_t"] = pd.to_datetime(df["host_received_timestamp_utc"], utc=True, format="ISO8601")
        _DF["df"] = df.sort_values("_t").reset_index(drop=True)
    return _DF["df"]


def conditions() -> list[str]:
    return [str(c) for c in recording()["condition"].unique()]


def pick_window(cond: str, size: int, seed: int) -> pd.DataFrame:
    """`size` consecutive rows (no pause > 2 s) of one recorded condition. A different seed = a different sample."""
    d = recording()
    d = d[d["condition"] == cond]
    seg = (d["_t"].diff().dt.total_seconds().fillna(0) > GAP_S).cumsum()
    parts = [g for _, g in d.groupby(seg.values) if len(g) >= size]
    if not parts:
        parts = [max((g for _, g in d.groupby(seg.values)), key=len)]
    rnd = random.Random(int(seed))
    g = rnd.choice(parts)
    start = rnd.randint(0, max(0, len(g) - size))
    return g.iloc[start:start + size]


def samples_of(rows: pd.DataFrame) -> list[dict]:
    keep = rows.drop(columns=["_t", "condition", "host_received_timestamp_utc"])
    return [{"timestamp": r["host_received_timestamp_utc"],
             "sensor_data": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in k_.items()}}
            for (_, r), (_, k_) in zip(rows.iterrows(), keep.iterrows())]


# ───────────────────────── http ─────────────────────────
def api(method: str, path: str, body: dict | None = None, params: dict | None = None, timeout: float = 180):
    try:
        r = httpx.request(method, CFG["api"] + path, json=body, params=params, timeout=timeout,
                          headers={"X-API-Key": CFG["key"]})
    except httpx.HTTPError as e:
        return 0, {"error": f"cannot reach {CFG['api']}: {e}"}
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"error": r.text[:300]}


def server_status() -> str:
    s, h = api("GET", "/health", timeout=5)
    if s != 200:
        return badge("SERVER", "not reachable", "#d93025") + f" <small>start it, then reload: {html.escape(CFG['api'])}</small>"
    _, m = api("GET", "/api/v1/anomaly/health", timeout=5)
    integ = h.get("integrations", {})
    return " ".join([
        badge("server", "up", "#2e9e5b"),
        badge("ML model", "loaded" if m.get("model_loaded") else "NOT loaded", "#2e9e5b" if m.get("model_loaded") else "#d93025"),
        badge("LLM", "on" if h.get("llm_enabled") else "off", "#2e9e5b" if h.get("llm_enabled") else "#888"),
        badge("agents read", integ.get("backend", "?"), "#2e9e5b" if integ.get("backend") == "csv" else "#d9a400")])


# ───────────────────────── html helpers ─────────────────────────
NAMES = {"temperature_c": "Motor temperature", "rpm": "Motor speed (RPM)", "speed_percent": "Speed setting",
         "gyro_mag": "Vibration level", "bus_voltage_v": "Supply voltage", "current_ma": "Motor current",
         "power_mw": "Power", "gyro_magnitude": "Vibration level", "gyro_x_rad_s": "Vibration (X)",
         "gyro_y_rad_s": "Vibration (Y)", "gyro_z_rad_s": "Vibration (Z)", "accel_z_m_s2": "Acceleration (Z)"}
FAULT_NAMES = {"overheating": "Motor overheating", "abnormal_vibration": "Abnormal vibration",
               "electrical_fault": "Electrical fault", "speed_deviation": "Speed deviation", "machine_fault": "Machine fault",
               "sensor_fault": "Sensor fault", "network_issue": "Network issue", "security_incident": "Security incident",
               "unknown": "Undetermined"}


def nice(name: str) -> str:
    return NAMES.get(name, name.replace("_", " ").capitalize())


def dot(ok: bool | None, text: str) -> str:
    c = {True: "#2e9e5b", False: "#d93025", None: "#9aa0a6"}[ok]
    return (f"<span style='display:inline-flex;align-items:center;margin-right:18px;font-size:13px;opacity:.85'>"
            f"<span style='width:8px;height:8px;border-radius:50%;background:{c};margin-right:6px'></span>{html.escape(text)}</span>")


def server_status() -> str:
    s, h = api("GET", "/health", timeout=5)
    if s != 200:
        return dot(False, f"Server not reachable ({CFG['api']})")
    _, m = api("GET", "/api/v1/anomaly/health", timeout=5)
    integ = h.get("integrations", {})
    return (dot(True, "Server online") + dot(bool(m.get("model_loaded")), "ML model " + ("ready" if m.get("model_loaded") else "not loaded"))
            + dot(True if h.get("llm_enabled") else None, "AI diagnosis " + ("on" if h.get("llm_enabled") else "off"))
            + dot(True if integ.get("backend") == "csv" else None, f"Telemetry: {integ.get('backend', '?')}"))


def card(inner: str, color: str = "#9aa0a6") -> str:
    return (f"<div style='border:1px solid rgba(128,128,128,.25);border-left:5px solid {color};padding:14px 18px;"
            f"border-radius:10px;margin:4px 0;background:rgba(128,128,128,.06)'>{inner}</div>")


def pill(text: str, color: str) -> str:
    return (f"<span style='display:inline-block;padding:2px 10px;border-radius:999px;background:{color};color:#fff;"
            f"font-size:12px;font-weight:600;letter-spacing:.3px;margin-right:6px'>{html.escape(text)}</span>")


def stepper(sensors: str, model: str, agents: str) -> str:
    """Three-stage pipeline strip. states: 'idle' | 'run' | 'ok' | 'bad' | 'skip' """
    col = {"idle": "#9aa0a6", "run": "#4a6fa5", "ok": "#2e9e5b", "bad": "#d93025", "skip": "#9aa0a6"}
    def box(n, title, st, sub=""):
        return (f"<div style='flex:1;padding:10px 14px;border-radius:10px;border:1px solid rgba(128,128,128,.25);"
                f"border-top:4px solid {col[st]};background:rgba(128,128,128,.06)'><div style='font-size:12px;opacity:.7'>STEP {n}</div>"
                f"<div style='font-weight:600'>{title}</div><div style='font-size:12px;opacity:.75'>{html.escape(sub)}</div></div>")
    arrow = "<div style='align-self:center;opacity:.4;font-size:20px'>›</div>"
    return ("<div style='display:flex;gap:8px;margin:6px 0 10px'>" + box(1, "Sensor data", *sensors) + arrow
            + box(2, "ML model", *model) + arrow + box(3, "AI agents", *agents) + "</div>")


def error_card(s: int, body: dict) -> str:
    detail = body.get("detail") or body.get("error") or body
    return card(f"<b>Request failed (HTTP {s})</b><br><span style='opacity:.8'>{html.escape(str(detail)[:400])}</span>", "#d93025")


def ml_card(ml: dict) -> str:
    anomaly = ml["is_anomaly"]
    color = "#d93025" if anomaly else "#2e9e5b"
    score = ml["anomaly_score"]
    bar = (f"<div style='height:8px;border-radius:4px;background:rgba(128,128,128,.25);position:relative;margin:10px 0 2px'>"
           f"<div style='width:{score * 100:.0f}%;height:8px;border-radius:4px;background:{color}'></div>"
           f"<div style='position:absolute;left:50%;top:-4px;height:16px;border-left:2px solid rgba(128,128,128,.7)'></div></div>"
           f"<div style='display:flex;justify-content:space-between;font-size:11px;opacity:.6'><span>normal</span>"
           f"<span>decision line</span><span>anomaly</span></div>")
    return card(f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<span style='font-size:22px;font-weight:700;color:{color}'>{'Anomaly detected' if anomaly else 'Normal operation'}</span>"
                f"<span style='opacity:.8'>score <b>{score:.2f}</b></span></div>{bar}", color)


def features_table(ml: dict) -> pd.DataFrame:
    rows = [{"Measurement": nice(f["name"]), "Influence": round(f.get("contribution", 0), 2),
             "Now": None if f.get("value") is None else round(f["value"], 3),
             "Normal": None if f.get("baseline") is None else round(f["baseline"], 3)}
            for f in ml.get("top_features", [])[:6]]
    return pd.DataFrame(rows, columns=["Measurement", "Influence", "Now", "Normal"])


SEV_TEXT = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "critical": "CRITICAL"}


def agent_card(a: dict) -> str:
    sev = a.get("severity", "low")
    color = SEV_COLOR.get(sev, "#888")
    act = a.get("action") or {}
    steps = "".join(f"<li style='margin:3px 0'>{html.escape(x)}</li>" for x in a.get("recommended_solution", [])[:3])
    title = FAULT_NAMES.get(a.get("fault_type"), "Incident")
    confidence = f"{a.get('confidence', 0):.0%} confidence"

    parts = [
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<span style='font-size:22px;font-weight:700'>{html.escape(title)}</span>"
        f"<span>{pill(SEV_TEXT.get(sev, sev), color)}"
        f"{pill(confidence, '#5f6b7a')}</span></div>",
        f"<p style='font-size:15px;margin:8px 0 10px;opacity:.9'>"
        f"{html.escape(a.get('headline') or a.get('summary') or '')}</p>",
    ]
    if a.get("reason"):
        parts.append(f"<p style='opacity:.85'><b>Not conclusive:</b> {html.escape(a['reason'])}</p>")
    if steps:
        parts.append(f"<div style='font-size:12px;opacity:.65;letter-spacing:.5px'>RECOMMENDED STEPS</div>"
                     f"<ol style='margin:4px 0 10px;padding-left:20px'>{steps}</ol>")
    ok = not act.get("approval_required")
    parts.append(f"<div style='font-size:13px;opacity:.85'>System action: <code>{html.escape(str(act.get('id', '?')))}</code> · "
                 f"{'inspection only, no approval needed' if ok else 'requires operator approval'}</div>")
    return card("".join(parts), color)


def evidence_table(a: dict) -> pd.DataFrame:
    rows = [{"Measurement": nice(e["channel"]), "Now": e["value"], "Normal": e["baseline"], "Change": e["change"]}
            for e in a.get("evidence", [])]
    return pd.DataFrame(rows, columns=["Measurement", "Now", "Normal", "Change"])


def votes_table(a: dict) -> pd.DataFrame:
    rows = []
    for v in a.get("votes", []):
        top = max(v["scores"].items(), key=lambda kv: kv[1]) if v.get("scores") else ("-", 0)
        rows.append({"Agent": v["role"].capitalize(), "Type": "AI (Gemini)" if v["engine"] == "llm" else "Rules",
                     "Assessment": top[0].replace("_", " "), "Score": round(top[1], 2)})
    return pd.DataFrame(rows, columns=["Agent", "Type", "Assessment", "Score"])


def plots(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = np.arange(len(rows))
    temp = pd.DataFrame({"sample": t, "Temperature (°C)": rows["temperature_c"].to_numpy(dtype=float)})
    mag = np.sqrt(sum(rows[c].to_numpy(dtype=float) ** 2 for c in ("gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")))
    vib = pd.DataFrame({"sample": t, "Vibration (rad/s)": mag})
    return temp, vib


def window_sub(rows: pd.DataFrame) -> str:
    return f"{len(rows)} readings"


# ───────────────────────── tab handlers ─────────────────────────
def _body(cond: str, rows: pd.DataFrame) -> dict:
    return {"incident_id": f"UI-{cond}-{int(time.time())}", "asset_id": "esp32-conveyor-01", "samples": samples_of(rows)}


def run_ml(cond: str, size: int, seed: int):
    rows = pick_window(cond, int(size), int(seed))
    s, ml = api("POST", "/api/v1/anomaly/window", _body(cond, rows))
    temp, vib = plots(rows)
    if s != 200:
        return error_card(s, ml), pd.DataFrame(), temp, vib, {}
    return ml_card(ml), features_table(ml), temp, vib, ml


def run_full(cond: str, size: int, seed: int):
    rows = pick_window(cond, int(size), int(seed))
    temp, vib = plots(rows)
    e = pd.DataFrame()
    sensors = ("ok", window_sub(rows))
    yield stepper(sensors, ("run", "analysing..."), ("idle", "")), "", e, e, temp, vib, {}
    s, ml = api("POST", "/api/v1/anomaly/window", _body(cond, rows))
    if s != 200:
        yield stepper(sensors, ("bad", "error"), ("idle", "")), error_card(s, ml), e, e, temp, vib, {}
        return
    mlsub = ("bad" if ml["is_anomaly"] else "ok", f"score {ml['anomaly_score']:.2f}")
    if not ml["is_anomaly"]:
        yield (stepper(sensors, mlsub, ("skip", "not needed")),
               ml_card(ml) + card("Everything is within normal range. No incident is created and the agents are not called.", "#2e9e5b"),
               e, e, temp, vib, {"ml": ml})
        return
    yield stepper(sensors, mlsub, ("run", "diagnosing...")), ml_card(ml), e, e, temp, vib, {"ml": ml}
    s2, a = api("POST", "/api/v1/incident/analyze", ml, params={"view": "compact", "force": "true"})
    if s2 != 200:
        yield stepper(sensors, mlsub, ("bad", "error")), ml_card(ml) + error_card(s2, a), e, e, temp, vib, {"ml": ml}
        return
    a["action"] = {"id": a.get("recommended_action"), "params": a.get("action_params"), "risk": a.get("risk_level"),
                   "approval_required": a.get("approval_required")}
    a["summary"] = a.get("explanation")
    a["evidence"] = [{"channel": x["channel"], "value": x["value"], "baseline": x["baseline"], "change": x["change"]}
                     for x in a.get("signals", [])]
    secs = (a.get("latency_ms") or 0) / 1000
    yield (stepper(sensors, mlsub, ("ok", f"{secs:.1f} s")), agent_card(a), evidence_table(a), votes_table(a), temp, vib,
           {"ml": ml, "agents": a})


# what-if: fields the user can change; everything else comes from a real "normal" row
WHATIF = [("temperature_c", "Motor temperature (°C)"), ("speed_percent", "Motor speed (%)"), ("rpm", "RPM"),
          ("current_ma", "Current (mA)"), ("bus_voltage_v", "Supply voltage (V)"),
          ("gyro_x_rad_s", "Vibration gyro X (rad/s)"), ("gyro_y_rad_s", "Vibration gyro Y (rad/s)"),
          ("gyro_z_rad_s", "Vibration gyro Z (rad/s)")]


def normal_row() -> pd.Series:
    d = recording()
    n = d[d["condition"] == "normal"]
    return n.iloc[len(n) // 2]


def whatif_defaults() -> list[float]:
    r = normal_row()
    return [round(float(r[k]), 3) for k, _ in WHATIF]


def run_whatif(*values):
    r = normal_row()
    sd = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
          for k, v in r.drop(labels=["host_received_timestamp_utc", "condition", "_t"]).items()}
    for (k, _), v in zip(WHATIF, values):
        sd[k] = float(v)
    s, ml = api("POST", "/api/v1/anomaly/predict", {"incident_id": f"UI-whatif-{int(time.time())}",
                                                    "asset_id": "esp32-conveyor-01", "sensor_data": sd})
    if s != 200:
        return error_card(s, ml), pd.DataFrame(), {}
    return ml_card(ml), features_table(ml), ml


LABELS = {"normal": "Normal operation", "overheated": "Overheating", "vibration_issue": "Vibration issue"}


# ───────────────────────── layout ─────────────────────────
def build_ui():
    import gradio as gr

    conds = conditions()
    choices = [(LABELS.get(c, c), c) for c in conds]
    default_cond = "overheated" if "overheated" in conds else conds[0]
    v6 = int(gr.__version__.split(".")[0]) >= 6
    kw = {} if v6 else {"theme": gr.themes.Soft(primary_hue="blue", neutral_hue="slate")}
    with gr.Blocks(title="AegisOT", **kw) as demo:
        gr.HTML("<div style='padding:6px 0 2px'><span style='font-size:26px;font-weight:700'>AegisOT</span>"
                "<span style='opacity:.6;margin-left:12px'>Predictive monitoring for industrial conveyors</span></div>")
        status = gr.HTML(server_status())

        with gr.Tabs():
            with gr.Tab("Full system"):
                with gr.Row(equal_height=True):
                    c2 = gr.Dropdown(choices, value=default_cond, label="Scenario (recorded data)", scale=3)
                    b2 = gr.Button("Run analysis", variant="primary", scale=1)
                with gr.Accordion("Options", open=False):
                    with gr.Row():
                        n2 = gr.Slider(3, 60, value=20, step=1, label="Readings in the window")
                        s2 = gr.Number(value=0, precision=0, label="Sample number (another window)")
                pipe2 = gr.HTML(stepper(("idle", ""), ("idle", ""), ("idle", "")))
                out2 = gr.HTML()
                with gr.Accordion("Evidence and agent breakdown", open=False):
                    with gr.Row():
                        ev2 = gr.Dataframe(label="Measured evidence", interactive=False)
                        vt2 = gr.Dataframe(label="Specialist agents", interactive=False)
                with gr.Accordion("Sensor charts", open=False):
                    with gr.Row():
                        p2a = gr.LinePlot(x="sample", y="Temperature (°C)", height=220)
                        p2b = gr.LinePlot(x="sample", y="Vibration (rad/s)", height=220)
                with gr.Accordion("API response (JSON)", open=False):
                    j2 = gr.JSON()
                b2.click(run_full, [c2, n2, s2], [pipe2, out2, ev2, vt2, p2a, p2b, j2])

            with gr.Tab("ML model"):
                with gr.Row(equal_height=True):
                    c1 = gr.Dropdown(choices, value=default_cond, label="Scenario (recorded data)", scale=3)
                    b1 = gr.Button("Run model", variant="primary", scale=1)
                with gr.Accordion("Options", open=False):
                    with gr.Row():
                        n1 = gr.Slider(3, 60, value=20, step=1, label="Readings in the window")
                        s1 = gr.Number(value=0, precision=0, label="Sample number (another window)")
                out1 = gr.HTML()
                ft1 = gr.Dataframe(label="What drove the score", interactive=False)
                with gr.Accordion("Sensor charts", open=False):
                    with gr.Row():
                        p1a = gr.LinePlot(x="sample", y="Temperature (°C)", height=220)
                        p1b = gr.LinePlot(x="sample", y="Vibration (rad/s)", height=220)
                with gr.Accordion("API response (JSON)", open=False):
                    j1 = gr.JSON()
                b1.click(run_ml, [c1, n1, s1], [out1, ft1, p1a, p1b, j1])

            with gr.Tab("What-if"):
                gr.Markdown("Start from a real normal reading, change a value and ask the model.")
                with gr.Row():
                    boxes = [gr.Number(value=v, label=lbl) for (k, lbl), v in zip(WHATIF, whatif_defaults())]
                b3 = gr.Button("Ask the model", variant="primary")
                out3 = gr.HTML()
                ft3 = gr.Dataframe(label="What drove the score", interactive=False)
                b3.click(run_whatif, boxes, [out3, ft3, gr.JSON(visible=False)])
        gr.Button("Refresh status", size="sm", variant="secondary").click(server_status, None, status)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="AegisOT Gradio demo (client of the running API)")
    ap.add_argument("--api", default=CFG["api"], help="the running server (ML + agents)")
    ap.add_argument("--key", default=CFG["key"], help="AGENT_API_KEY")
    ap.add_argument("--csv", default=CFG["csv"], help="the recording used to build input windows")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio link (for showing to someone remotely)")
    a = ap.parse_args()
    CFG.update(api=a.api.rstrip("/"), key=a.key, csv=a.csv)
    import gradio as gr
    extra = {"theme": gr.themes.Soft(primary_hue="blue", neutral_hue="slate")} if int(gr.__version__.split(".")[0]) >= 6 else {}
    build_ui().launch(server_name="0.0.0.0", server_port=a.port, share=a.share, **extra)


if __name__ == "__main__":
    main()
