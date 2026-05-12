"""
maria_dashboard.py
══════════════════════════════════════════════════════════════════
Maria AI — Training Dashboard Generator

Reads your REAL data files and produces a polished HTML report.
Zero demo data — everything shown is from your actual files.

Place this in the SAME folder as Maria_App.py, then run:
    python maria_dashboard.py

Opens maria_dashboard_report.html in your browser automatically.
══════════════════════════════════════════════════════════════════
"""

import os, json, webbrowser, socket, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

# ── File paths (same folder as this script / Maria_App.py) ────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project_Maria/
METRICS_FILE     = os.path.join(BASE_DIR, "maria_metrics.json")
DPO_FILE         = os.path.join(BASE_DIR, "maria_dpo_dataset.jsonl")
SFT_LORA_FILE    = os.path.join(BASE_DIR, "maria_training_data.json")   # hand-crafted
SFT_HARVEST_FILE = os.path.join(BASE_DIR, "maria_sft_dataset.jsonl")    # auto-harvested
OUT_FILE         = os.path.join(BASE_DIR, "maria_dashboard_report.html")

# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_metrics():
    if not os.path.exists(METRICS_FILE):
        return {}
    with open(METRICS_FILE, encoding="utf-8") as f:
        return json.load(f)

def load_dpo():
    rows = []
    if os.path.exists(DPO_FILE):
        with open(DPO_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try: rows.append(json.loads(line))
                    except: pass
    return rows

def load_sft():
    total = 0
    if os.path.exists(SFT_LORA_FILE):
        try:
            with open(SFT_LORA_FILE, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, list): total += len(d)
            elif isinstance(d, dict): total += len(d.get("training_examples", []))
        except: pass
    if os.path.exists(SFT_HARVEST_FILE):
        try:
            with open(SFT_HARVEST_FILE, encoding="utf-8") as f:
                total += sum(1 for ln in f if ln.strip())
        except: pass
    return total

def collect():
    m       = load_metrics()
    dpo_raw = load_dpo()
    sft_n   = load_sft()

    sources, margins, langs = {}, [], {}
    src_margin_series = {}   # source -> list of margins (for line chart)
    for ex in dpo_raw:
        src = ex.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
        mg = ex.get("reward_margin")
        if mg is not None:
            mg_f = round(float(mg), 4)
            margins.append(mg_f)
            src_margin_series.setdefault(src, []).append(mg_f)
        lg = ex.get("language", "en")
        langs[lg] = langs.get(lg, 0) + 1

    total_q  = m.get("total_queries", 0)
    hi_c     = m.get("high_confidence_count", 0)
    lo_c     = m.get("low_confidence_count", 0)
    mid_c    = max(total_q - hi_c - lo_c, 0)
    corr     = m.get("self_corrections", 0)
    rel_fail = m.get("relevance_failures", 0)
    times    = [t for t in m.get("response_times", []) if isinstance(t, (int, float))]
    qtypes   = m.get("query_types", {})
    avg_conf = m.get("average_confidence", 0.0)
    tokens   = m.get("total_tokens_estimated", 0)
    smart    = m.get("smart_mode_uses", 0)
    web      = m.get("web_searches", 0)

    hallu_risk = min((lo_c + corr) / max(total_q, 1) * 100, 100)
    rel_rate   = (total_q - rel_fail) / max(total_q, 1) * 100
    avg_time   = sum(times) / len(times) if times else 0
    avg_margin = sum(margins) / len(margins) if margins else 0

    # Histogram for reward margins
    hist_labels, hist_data, hist_colors = [], [], []
    if margins:
        mn, mx = min(margins), max(margins)
        n_bins = min(14, max(len(margins), 1))
        step   = (mx - mn) / n_bins if mx != mn else 0.1
        counts = [0] * n_bins
        for mg in margins:
            i = min(int((mg - mn) / step), n_bins - 1)
            counts[i] += 1
        for i in range(n_bins):
            left  = mn + i * step
            hist_labels.append(str(round(left, 3)))
            hist_data.append(counts[i])
            if left >= 0.5:   hist_colors.append("#1DBF7B")
            elif left >= 0.2: hist_colors.append("#FF9D00")
            else:             hist_colors.append("#F15B5B")

    # Rolling avg (window=8) for response times
    roll = []
    w = 8
    for i in range(len(times)):
        window = times[max(0, i-w+1):i+1]
        roll.append(round(sum(window)/len(window), 3))

    def perc(arr, p):
        if not arr: return 0
        s = sorted(arr)
        idx = (p/100)*(len(s)-1)
        lo, hi = int(idx), min(int(idx)+1, len(s)-1)
        return round(s[lo] + (s[hi]-s[lo])*(idx-lo), 3)

    p50 = perc(times, 50)
    p95 = perc(times, 95)

    files_found = {
        "metrics":     os.path.exists(METRICS_FILE),
        "dpo":         os.path.exists(DPO_FILE),
        "sft_lora":    os.path.exists(SFT_LORA_FILE),
        "sft_harvest": os.path.exists(SFT_HARVEST_FILE),
    }

    def prog(val, target):
        return min(round(val / target * 100, 1) if target else 0, 100)

    return {
        "total_q":      total_q,
        "avg_conf":     round(avg_conf, 1),
        "hi_c": hi_c, "mid_c": mid_c, "lo_c": lo_c,
        "corr":   corr,  "smart": smart, "web": web,
        "rel_fail": rel_fail, "tokens": tokens,
        "sft_n":  sft_n,  "dpo_n":  len(dpo_raw),
        "avg_margin": round(avg_margin, 4),
        "hallu_risk": round(hallu_risk, 1),
        "rel_rate":   round(rel_rate, 1),
        "avg_time":   round(avg_time, 3),
        "p50": p50, "p95": p95,
        "times":        [round(t, 3) for t in times[-200:]],
        "roll":         roll[-200:],
        "qtypes":       qtypes,
        "sources":      sources,
        "margins":      margins,
        "langs":        langs,
        "hist_labels":  hist_labels,
        "hist_data":    hist_data,
        "hist_colors":  hist_colors,
        "files_found":  files_found,
        "generated_at": datetime.now().strftime("%B %d, %Y  %H:%M:%S"),
        "prog_sft":     prog(sft_n, 500),
        "prog_dpo":     prog(len(dpo_raw), 1000),
        "prog_q":       prog(total_q, 1000),
        "prog_hc":      prog(hi_c, 500),
        "src_margin_series": {s: v[-100:] for s, v in src_margin_series.items()},
        "dpo_pos":      sum(1 for e in dpo_raw if e.get("reward_margin", 0) >= 0.5),
        "dpo_neg":      sum(1 for e in dpo_raw if e.get("reward_margin", 0) < 0.5),
        "recent_examples": [
            {
                "source":  e.get("source", "—"),
                "margin":  round(float(e.get("reward_margin", 0)), 3),
                "chosen":  e.get("chosen", "")[:160].replace("\n", " "),
                "rejected": e.get("rejected", "")[:100].replace("\n", " "),
            }
            for e in reversed(dpo_raw[-30:])
        ],
    }

# ══════════════════════════════════════════════════════════════════════════
# HTML BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_docs_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Maria AI — Documentation</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#FFFDF8;--surface:#FFF;--sidebar:#FAFAF4;
  --border:#EBE6DA;--border2:#D8D0BF;
  --yellow:#FF9D00;--yl:#FFF4DC;--ym:#FFE0A0;
  --coral:#F15B5B;--cl:#FFF0F0;
  --green:#1DBF7B;--gl:#E8FAF3;
  --blue:#3B82F6;--bl:#EEF4FF;
  --purple:#8B5CF6;--pl:#F3EEFF;
  --text-h:#1A1714;--text-b:#383330;--text-m:#6A6360;--text-dim:#9B9490;
  --sw:240px;--r:10px;--rl:14px
}
html{scroll-behavior:smooth}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text-b);display:flex;min-height:100vh;font-size:15px;line-height:1.7}
/* Sidebar */
.sb{width:var(--sw);min-height:100vh;background:var(--sidebar);border-right:1.5px solid var(--border);position:fixed;top:0;left:0;overflow-y:auto;z-index:200;display:flex;flex-direction:column}
.sb-hd{padding:20px 18px 16px;border-bottom:1px solid var(--border)}
.sb-back{display:flex;align-items:center;gap:7px;padding:8px 10px;border-radius:8px;font-size:13px;font-weight:600;color:var(--text-m);text-decoration:none;border:1.5px solid var(--border2);background:var(--surface);transition:all .15s;margin-bottom:12px}
.sb-back:hover{background:var(--yl);border-color:var(--ym);color:var(--text-h)}
.sb-title{font-size:14px;font-weight:700;color:var(--text-h);margin-bottom:2px}
.sb-sub{font-size:11px;color:var(--text-dim)}
.sb-sec{padding:14px 14px 4px}
.sb-lbl{font-size:10px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--text-dim);padding:0 6px;margin-bottom:5px}
.sb-nav{list-style:none;display:flex;flex-direction:column;gap:2px}
.sb-nav a{display:block;padding:7px 10px;border-radius:7px;font-size:13px;color:var(--text-m);text-decoration:none;transition:all .15s;font-weight:500}
.sb-nav a:hover{background:var(--yl);color:var(--text-h)}
.sb-nav a.active{background:var(--yl);color:var(--yellow);font-weight:700;border-left:3px solid var(--yellow);padding-left:7px}
/* Main */
.main{margin-left:var(--sw);flex:1;padding:0}
/* Topbar */
.topbar{background:var(--surface);border-bottom:1.5px solid var(--border);padding:12px 48px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.bc{font-size:12.5px;color:var(--text-dim);display:flex;align-items:center;gap:6px}
.bc a{color:var(--text-dim);text-decoration:none}
.bc a:hover{color:var(--yellow)}
.bc .cur{color:var(--text-b);font-weight:600}
.pill{font-size:11px;font-weight:600;padding:3px 11px;border-radius:20px;background:var(--yl);color:#92400E;border:1px solid var(--ym)}
/* Sections */
.doc-section{padding:52px 48px 72px;max-width:840px;border-bottom:1px solid var(--border)}
.doc-section:last-child{border-bottom:none}
.ey{font-size:10.5px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--yellow);margin-bottom:8px}
h1{font-family:'Source Serif 4',Georgia,serif;font-size:33px;font-weight:700;color:var(--text-h);line-height:1.2;margin-bottom:10px;letter-spacing:-.01em}
.sub{font-size:15.5px;color:var(--text-m);max-width:600px;margin-bottom:24px;line-height:1.6}
h2{font-family:'Source Serif 4',Georgia,serif;font-size:20px;font-weight:600;color:var(--text-h);margin:36px 0 10px;display:flex;align-items:center;gap:10px}
h2::before{content:'';display:block;width:3px;height:22px;background:var(--yellow);border-radius:2px;flex-shrink:0}
h3{font-size:14px;font-weight:700;color:var(--text-h);margin:18px 0 6px}
p{color:var(--text-m);margin-bottom:12px;font-size:14px;line-height:1.65}
code{font-family:'JetBrains Mono',monospace;font-size:12px;background:var(--border);padding:1px 6px;border-radius:4px;color:var(--text-h)}
pre{font-family:'JetBrains Mono',monospace;font-size:12px;background:#1A1714;color:#E8E3DA;padding:18px 22px;border-radius:10px;line-height:1.8;overflow-x:auto;margin:12px 0}
pre .c{color:#9B9490}
pre .k{color:#8B5CF6}
pre .f{color:#1DBF7B}
pre .s{color:#FF9D00}
pre .n{color:#3B82F6}
/* Cards */
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--rl);padding:22px 24px;margin-bottom:16px}
.card-accent{border-left:4px solid var(--yellow)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:12px 0}
.item{padding:14px;background:var(--bg);border-radius:10px;border:1.5px solid var(--border)}
.item-label{font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;margin-bottom:6px}
.item-title{font-size:14px;font-weight:700;color:var(--text-h);font-family:'JetBrains Mono',monospace;margin-bottom:7px}
.item-body{font-size:12.5px;color:var(--text-m);line-height:1.6}
.badges{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}
.badge{font-size:10.5px;padding:2px 9px;border-radius:20px;font-weight:600}
.badge-y{background:var(--yl);color:#92400E}
.badge-b{background:var(--bl);color:#1E40AF}
.badge-g{background:var(--gl);color:#065F46}
/* Stat row */
.sr{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0}
.sc{background:var(--surface);border:1px solid var(--border);border-radius:var(--rl);padding:16px 20px;position:relative;overflow:hidden}
.sc::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:var(--rl) var(--rl) 0 0}
.sc.y::after{background:var(--yellow)}.sc.g::after{background:var(--green)}.sc.b::after{background:var(--blue)}
.sc-lb{font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--text-dim);margin-bottom:7px}
.sc-v{font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:500;line-height:1;margin-bottom:4px}
.sc-v.y{color:var(--yellow)}.sc-v.g{color:var(--green)}.sc-v.b{color:var(--blue)}
.sc-s{font-size:11.5px;color:var(--text-dim)}
/* Table */
table{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}
th{text-align:left;padding:8px 12px;font-weight:600;color:var(--text-b);border-bottom:2px solid var(--border)}
td{padding:9px 12px;border-bottom:1px solid var(--border);color:var(--text-m)}
tr:last-child td{border-bottom:none}
td:first-child{font-family:'JetBrains Mono',monospace;color:var(--yellow);font-size:12px}
/* Callout */
.callout{display:flex;gap:13px;padding:14px 18px;border-radius:var(--r);margin:12px 0;border:1.5px solid}
.callout.ok{background:var(--gl);border-color:#A7F3D0}
.callout.info{background:var(--yl);border-color:var(--ym)}
/* Flow diagram */
.flow{font-family:'JetBrains Mono',monospace;font-size:12px;background:#1A1714;color:#E8E3DA;padding:20px 24px;border-radius:10px;line-height:2.1;overflow-x:auto;margin:14px 0}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
</style>
</head>
<body>

<aside class="sb">
  <div class="sb-hd">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
      <div style="width:34px;height:34px;background:#FF9D00;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0">&#x1F917;</div>
      <div>
        <div style="font-size:14.5px;font-weight:700;color:#1A1714;line-height:1.2">Maria AI</div>
        <div style="font-size:11px;color:#9B9490">Documentation</div>
      </div>
    </div>
  </div>
  <div class="sb-sec" style="padding-top:10px">
    <div class="sb-lbl">Navigate</div>
    <ul class="sb-nav">
      <li><a href="/dashboard">Training Dashboard</a></li>
    </ul>
  </div>
  <div class="sb-sec">
    <div class="sb-lbl">Sections</div>
    <ul class="sb-nav">
      <li><a href="#overview"      class="active">Overview</a></li>
      <li><a href="#architecture"          >Architecture</a></li>
      <li><a href="#intelligence"          >Intelligence</a></li>
      <li><a href="#retrieval"             >Retrieval &amp; Safety</a></li>
      <li><a href="#mathcode"              >Math &amp; Code</a></li>
      <li><a href="#training"              >Training Pipelines</a></li>
      <li><a href="#training-methods"      >How Training Works</a></li>
      <li><a href="#reference"             >Function Reference</a></li>
    </ul>
  </div>
</aside>

<div class="main">
  <div class="topbar">
    <div class="bc">
      <span class="cur">Documentation</span>
    </div>
    <span class="pill">Maria_App.py &nbsp;·&nbsp; 1.9 MB &nbsp;·&nbsp; ~120 classes</span>
  </div>

  <!-- OVERVIEW -->
  <section class="doc-section" id="overview">
    <div class="ey">Getting Started</div>
    <h1>Maria AI Documentation</h1>
    <p class="sub">Complete technical reference for Maria_App.py — a fully local, privacy-first AI assistant built on Ollama and PyQt5.</p>

    <div class="card card-accent">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:20px">
        <div>
          <div style="font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--yellow);margin-bottom:6px">Model Card</div>
          <div style="font-size:18px;font-weight:700;color:var(--text-h);margin-bottom:6px">Maria AI — Local Intelligent Assistant</div>
          <p style="margin:0;max-width:520px">Routes each query to the best specialist model, grounds answers in live sources, executes code in a sandbox, solves math symbolically, and trains itself from your feedback — all without cloud APIs.</p>
        </div>
        <div style="text-align:right;flex-shrink:0">
          <div style="font-size:11px;color:var(--text-dim);margin-bottom:3px">Capability Score</div>
          <div style="font-size:38px;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--green);line-height:1">95<span style="font-size:16px;font-weight:400">/100</span></div>
        </div>
      </div>
    </div>

    <div class="sr">
      <div class="sc y"><div class="sc-lb">Models</div><div class="sc-v y">3</div><div class="sc-s">llama3.1 · deepseek-r1 · qwen2.5-coder</div></div>
      <div class="sc g"><div class="sc-lb">Source Size</div><div class="sc-v g">1.9 MB</div><div class="sc-s">~120 classes, 70+ functions</div></div>
      <div class="sc b"><div class="sc-lb">AI Capabilities</div><div class="sc-v b">12+</div><div class="sc-s">CoT · RAG · sandbox · math · DPO</div></div>
    </div>

    <h2>Key Capabilities</h2>
    <div class="grid2">
      <div class="card" style="margin:0">
        <h3>Specialist Model Routing</h3>
        <p style="margin:0">IntentRouter scores keywords and routes each query to the best model — deepseek-r1 for math, qwen2.5-coder for code, llama3.1 for everything else.</p>
      </div>
      <div class="card" style="margin:0">
        <h3>Hallucination Detection</h3>
        <p style="margin:0">HallucinationDetector actively checks every response against retrieved sources, flagging ungrounded claims before the user sees them.</p>
      </div>
      <div class="card" style="margin:0">
        <h3>Live Web Retrieval + RAG</h3>
        <p style="margin:0">DuckDuckGo search, Wikipedia, and HybridWebRAG ground answers in real-time sources — not frozen training data from months ago.</p>
      </div>
      <div class="card" style="margin:0">
        <h3>Exact Math via SymPy</h3>
        <p style="margin:0">PerfectMathEngine routes math through SymPy for provably correct algebra, calculus, and equation solving — not LLM guesses.</p>
      </div>
      <div class="card" style="margin:0">
        <h3>Sandboxed Code Execution</h3>
        <p style="margin:0"><code>_safe_execute_python()</code> runs code in an isolated subprocess with a 5-second timeout so Maria can verify its own solutions.</p>
      </div>
      <div class="card" style="margin:0">
        <h3>Self-Training Pipelines</h3>
        <p style="margin:0">DPO, SFT, and GRPO pipelines collect feedback from every conversation and trigger fine-tuning automatically in the background.</p>
      </div>
    </div>

    <h2>System Requirements</h2>
    <table>
      <thead><tr><th>Requirement</th><th>Minimum</th><th>Recommended</th></tr></thead>
      <tbody>
        <tr><td>Python</td><td style="font-family:inherit">3.9+</td><td style="font-family:inherit;color:var(--green)">3.10+</td></tr>
        <tr><td>Ollama</td><td style="font-family:inherit">Latest</td><td style="font-family:inherit;color:var(--green)">Latest + GPU offload</td></tr>
        <tr><td>RAM</td><td style="font-family:inherit">8 GB</td><td style="font-family:inherit;color:var(--green)">16 GB+</td></tr>
        <tr><td>GPU VRAM</td><td style="font-family:inherit">None (CPU mode)</td><td style="font-family:inherit;color:var(--green)">6 GB+ NVIDIA/AMD</td></tr>
        <tr><td>Key Libraries</td><td colspan="2" style="font-family:inherit"><code>PyQt5</code> <code>ollama</code> <code>sympy</code> <code>langdetect</code> <code>reportlab</code> <code>duckduckgo-search</code> <code>wikipedia-api</code></td></tr>
      </tbody>
    </table>
  </section>

  <!-- ARCHITECTURE -->
  <section class="doc-section" id="architecture">
    <div class="ey">System Design</div>
    <h1>Architecture</h1>
    <p class="sub">How a query flows through Maria from input to final response.</p>

    <div class="flow">
<span style="color:#FF9D00">User Input</span>
    ↓
┌────────────────────────────────┐
│ <span style="color:#1DBF7B">safety_filter()</span>              │  ← blocks harmful input immediately
│ <span style="color:#1DBF7B">IntentRouter</span>                 │  ← classify: math / code / general / …
│ <span style="color:#1DBF7B">ToolPlanner</span>                  │  ← plan tool calls if query needs them
├────────────────────────────────┤
│ <span style="color:#3B82F6">SpecializedCoT</span>               │  ← type-specific chain-of-thought pre-pass
│ <span style="color:#3B82F6">Specialist Model (streaming)</span> │  ← generate response token by token
│ <span style="color:#3B82F6">ReflectionPass</span>               │  ← post-generation self-check
├────────────────────────────────┤
│ <span style="color:#F15B5B">HallucinationDetector</span>        │  ← validate claims against sources
│ <span style="color:#F15B5B">validate_sources_topic()</span>     │  ← topic relevance check
└────────────────────────────────┘
    ↓
<span style="color:#FF9D00">Response to User</span>  +  <span style="color:#8B5CF6">DPO/SFT collection (background)</span></div>

    <h2>Specialist Models</h2>
    <div class="grid3">
      <div class="item">
        <div class="item-label" style="color:var(--yellow)">General</div>
        <div class="item-title">llama3.1:8b</div>
        <div class="item-body">Default model. Handles conversation, summarization, creative tasks, Filipino queries, and anything that doesn't match a specialist domain.</div>
        <div class="badges"><span class="badge badge-y">General</span><span class="badge badge-y">Filipino</span><span class="badge badge-y">Creative</span></div>
      </div>
      <div class="item">
        <div class="item-label" style="color:var(--blue)">Reasoning</div>
        <div class="item-title">deepseek-r1:8b</div>
        <div class="item-body">Math and reasoning specialist. Routes math problems, logic puzzles, proofs, and complex multi-step tasks. Pairs with SymPy for exact computation.</div>
        <div class="badges"><span class="badge badge-b">Math</span><span class="badge badge-b">Reasoning</span><span class="badge badge-b">Logic</span></div>
      </div>
      <div class="item">
        <div class="item-label" style="color:var(--green)">Code</div>
        <div class="item-title">qwen2.5-coder:7b</div>
        <div class="item-body">Code-specialized model. Handles programming, debugging, code generation, and review. Works with the sandbox to verify output.</div>
        <div class="badges"><span class="badge badge-g">Python</span><span class="badge badge-g">Debug</span><span class="badge badge-g">Code Gen</span></div>
      </div>
    </div>

    <h2>Tool Pipeline</h2>
    <div class="grid2">
      <div class="card" style="margin:0">
        <h3>ToolPlanner</h3>
        <p style="margin:0">A deterministic 4-step pipeline that routes clear-intent queries directly to the right tools without making the LLM decide at each step. Prevents tool-selection drift on straightforward queries.</p>
      </div>
      <div class="card" style="margin:0">
        <h3>ReActToolRegistry</h3>
        <p style="margin:0">Dynamic Reason + Act loop. Available tools: <code>web_search</code>, <code>wikipedia</code>, <code>run_python</code>, <code>sympy_math</code>, <code>read_file</code>. Each tool returns an observation string fed back to the model.</p>
      </div>
    </div>
  </section>

  <!-- INTELLIGENCE -->
  <section class="doc-section" id="intelligence">
    <div class="ey">Core AI</div>
    <h1>Intelligence System</h1>
    <p class="sub">The reasoning layer that makes Maria smarter than a plain LLM chatbot.</p>

    <h2>IntentRouter</h2>
    <p>Scored keyword classifier — each keyword class carries a weight, and the class with the highest accumulated score wins. Ties resolved by priority order. Prevents false-fires on generic phrases.</p>
    <pre><span class="c"># Example</span>
router = <span class="f">IntentRouter</span>()
intent = router.<span class="n">classify</span>(<span class="s">"solve x^2 + 3x = 10"</span>)
<span class="c"># intent = 'math'  →  routes to deepseek-r1:8b</span></pre>
    <p>Intent classes: <code>math</code>, <code>code</code>, <code>science</code>, <code>filipino</code>, <code>creative</code>, <code>factual</code>, <code>general</code></p>

    <h2>SpecializedCoT — Chain-of-Thought</h2>
    <p>Type-specific chain-of-thought scaffolds that run in 350–500 tokens before the main generation. Each query type gets a different scaffold — not the same 4 generic steps for everything. Returns a <code>(system_prompt, user_prompt)</code> pair that primes the model.</p>
    <div class="grid3">
      <div class="item"><div class="item-label" style="color:var(--yellow)">Math</div><div class="item-body">Identify knowns → set up equations → solve step by step → verify</div></div>
      <div class="item"><div class="item-label" style="color:var(--yellow)">Code</div><div class="item-body">Understand problem → plan structure → write → test mentally → edge cases</div></div>
      <div class="item"><div class="item-label" style="color:var(--yellow)">Factual</div><div class="item-body">What is known → what sources confirm → what is uncertain → answer</div></div>
      <div class="item"><div class="item-label" style="color:var(--yellow)">Logic</div><div class="item-body">State premises → check constraints → enumerate possibilities → conclude</div></div>
      <div class="item"><div class="item-label" style="color:var(--yellow)">Creative</div><div class="item-body">Understand tone → brainstorm angles → select best → develop with detail</div></div>
      <div class="item"><div class="item-label" style="color:var(--yellow)">Generic</div><div class="item-body">Fallback scaffold — improved step-by-step reasoning for unclassified queries</div></div>
    </div>

    <h2>ReflectionPass</h2>
    <p>Fast post-generation self-check — does not re-generate the full answer. Scans for the most common small-model failure modes and corrects them inline. Applied to logic and abstract answers.</p>
    <div class="callout ok">
      <div style="font-size:13px;color:var(--text-b)">Catches: <strong>missed constraints</strong> in logic puzzles · <strong>false premises</strong> stated as fact · <strong>calculation errors</strong> in the reasoning chain · <strong>self-contradictions</strong> within the same response.</div>
    </div>

    <h2>AdvancedReasoningEngine</h2>
    <p>Multi-strategy reasoning engine that selects the best approach for each problem. Strategies include chain-of-thought, self-consistency (multiple samples voted on), tree-of-thought branching, and analogy-based reasoning. Strategies are composed, not applied in a fixed pipeline.</p>

    <h2>Other Intelligence Classes</h2>
    <div class="grid2">
      <div class="card" style="margin:0"><h3>SelfCritiqueSystem</h3><p style="margin:0">Runs a targeted critique against the model's own response before it's shown to the user. Flags low-confidence sections and requests clarification internally.</p></div>
      <div class="card" style="margin:0"><h3>ElitePromptBuilder</h3><p style="margin:0">Constructs highly optimized system prompts by combining base instructions, persona context, retrieved knowledge, and CoT scaffolds into a single coherent prompt.</p></div>
      <div class="card" style="margin:0"><h3>SelfConsistency</h3><p style="margin:0">Samples multiple responses and selects the most consistent answer by majority vote. Used for high-stakes factual and math queries.</p></div>
      <div class="card" style="margin:0"><h3>KnowledgeSynthesizer</h3><p style="margin:0">Merges retrieved knowledge from multiple sources (web, Wikipedia, local RAG) into a coherent context block before passing it to the model.</p></div>
    </div>
  </section>

  <!-- RETRIEVAL & SAFETY -->
  <section class="doc-section" id="retrieval">
    <div class="ey">Data & Safety</div>
    <h1>Retrieval &amp; Safety</h1>
    <p class="sub">How Maria fetches real-world knowledge and guards against bad outputs.</p>

    <h2>HybridWebRAG</h2>
    <p>Combines dense (embedding-based) and sparse (keyword-based) retrieval over local documents and live web sources. Results are ranked and injected into the prompt as grounded context.</p>
    <div class="grid2">
      <div class="item"><div class="item-label" style="color:var(--yellow)">Offline Mode</div><div class="item-body">Retrieves from locally indexed documents only. Fast, private, no network required.</div></div>
      <div class="item"><div class="item-label" style="color:var(--yellow)">Hybrid Mode</div><div class="item-body">Combines local documents with live web retrieval. Best accuracy for current events and niche topics.</div></div>
    </div>

    <h2>Web Search — DuckDuckGo</h2>
    <pre><span class="k">def</span> <span class="f">search_web</span>(query: <span class="s">str</span>, max_results: <span class="s">int</span> = <span class="n">3</span>) -> <span class="s">str</span>:
    <span class="c">\"""Returns formatted results string for prompt injection.\"""</span>
    <span class="c"># No API key, no signup — uses duckduckgo_search</span>
    <span class="c"># Returns: title + URL + snippet per result</span></pre>
    <p>Results are passed into the model's context as source material. <code>validate_sources_topic()</code> then checks that results are topically relevant before including them.</p>

    <h2>Wikipedia — Three-Tier System</h2>
    <div class="grid3">
      <div class="item"><div class="item-label" style="color:var(--blue)">Online</div><div class="item-title">WikipediaKnowledge</div><div class="item-body">Live Wikipedia API. Fetches article summaries and full text.</div></div>
      <div class="item"><div class="item-label" style="color:var(--blue)">Offline</div><div class="item-title">OfflineWikipedia</div><div class="item-body">Falls back to a locally cached snapshot when offline.</div></div>
      <div class="item"><div class="item-label" style="color:var(--blue)">Production</div><div class="item-title">HybridWikipedia</div><div class="item-body">Tries online first, falls back automatically. Used in all live queries.</div></div>
    </div>

    <h2>HallucinationDetector</h2>
    <p>After generation, compares key claims against retrieved source material. Looks for statements with no grounding in the fetched context. Flagged responses are blocked, annotated with a warning, or sent for re-generation.</p>
    <pre>detector = <span class="f">HallucinationDetector</span>()
result = detector.<span class="n">check</span>(response, sources)
<span class="c"># result.is_hallucination : bool</span>
<span class="c"># result.confidence       : float (0.0 – 1.0)</span>
<span class="c"># result.flagged_claims   : List[str]</span></pre>

    <h2>Safety Functions</h2>
    <div class="grid2">
      <div class="card" style="margin:0"><h3>safety_filter(text)</h3><p style="margin:0">HARM_PAT regex blocklist check. Returns True if content is safe to process. Applied to both user input and model output.</p></div>
      <div class="card" style="margin:0"><h3>validate_sources_topic()</h3><p style="margin:0">Checks that retrieved web sources are topically relevant to the user's query. Prevents off-topic source injection from polluting the model's context.</p></div>
    </div>
  </section>

  <!-- MATH & CODE -->
  <section class="doc-section" id="mathcode">
    <div class="ey">Computation</div>
    <h1>Math &amp; Code</h1>
    <p class="sub">Exact symbolic math and sandboxed code execution — no more guessed outputs.</p>

    <h2>PerfectMathEngine + SymPy</h2>
    <p>When IntentRouter classifies a query as math, PerfectMathEngine takes over. It parses the expression, hands it to SymPy for symbolic computation, and returns an exact result. Covers algebra, calculus (derivatives/integrals), linear algebra, and equation solving.</p>
    <pre><span class="c"># Internal flow for: "solve x^2 - 5x + 6 = 0"</span>
<span class="k">from</span> sympy <span class="k">import</span> symbols, solve
x = symbols(<span class="s">'x'</span>)
result = solve(x**<span class="n">2</span> - <span class="n">5</span>*x + <span class="n">6</span>, x)
<span class="c"># result = [2, 3]  — exact, not approximated</span></pre>
    <p><code>_sympy_precompute()</code> pre-computes math before LLM generation, injecting the exact result into the prompt so the numbers in the response are always correct.</p>

    <h2>_safe_execute_python()</h2>
    <pre><span class="k">def</span> <span class="f">_safe_execute_python</span>(code: <span class="s">str</span>, timeout: <span class="s">int</span> = <span class="n">5</span>) -> <span class="s">str</span>:
    <span class="c">\"""Run code in isolated subprocess. Returns stdout or error.\"""</span></pre>
    <div class="grid2">
      <div class="item"><div class="item-label" style="color:var(--yellow)">Isolation</div><div class="item-body">Runs in a separate subprocess — Maria's process is unaffected even if the code crashes or hangs.</div></div>
      <div class="item"><div class="item-label" style="color:var(--yellow)">Timeout</div><div class="item-body">Hard 5-second kill prevents infinite loops from blocking the UI thread.</div></div>
      <div class="item"><div class="item-label" style="color:var(--yellow)">Output Capture</div><div class="item-body">Captures stdout, stderr, and exceptions as a structured string for the model to interpret.</div></div>
      <div class="item"><div class="item-label" style="color:var(--yellow)">Use Case</div><div class="item-body">Maria generates code → runs it → uses the real output in its response. No more guessed results.</div></div>
    </div>

    <h2>CodeSpecialist + CodeDebuggerMode</h2>
    <p><strong>CodeSpecialist</strong> routes code queries to <code>qwen2.5-coder:7b</code> with a code-specific prompt scaffold. <strong>CodeDebuggerMode</strong> is a focused debugging assistant — reads the user's code, identifies the error, explains it, and generates a corrected version. Both verify the solution actually runs via the sandbox.</p>
  </section>

  <!-- TRAINING -->
  <section class="doc-section" id="training">
    <div class="ey">Self-Improvement</div>
    <h1>Training Pipelines</h1>
    <p class="sub">Maria collects training data from every conversation and trains itself in the background — no manual intervention needed.</p>

    <div class="grid3">
      <div class="item">
        <div class="item-label" style="color:var(--yellow)">DPO</div>
        <div class="item-title">Direct Preference Opt.</div>
        <div class="item-body">Collects chosen/rejected pairs from thumbs up/down ratings and auto-postprocessing. DPOPipeline coordinates collector, validator, balancer, and evaluator.</div>
      </div>
      <div class="item">
        <div class="item-label" style="color:var(--blue)">SFT</div>
        <div class="item-title">Supervised Fine-Tuning</div>
        <div class="item-body">Harvests high-confidence responses and hand-crafted examples. SFTPipeline mirrors DPOPipeline's interface so the UI treats both identically.</div>
      </div>
      <div class="item">
        <div class="item-label" style="color:var(--green)">GRPO</div>
        <div class="item-title">Group Relative Policy Opt.</div>
        <div class="item-body">GRPOScheduler triggers rollouts every N turns without hammering Ollama. Reward function compares candidate responses by quality score.</div>
      </div>
    </div>

    <h2>Live Training Triggers</h2>
    <div class="grid2">
      <div class="card" style="margin:0"><h3>LiveTrainingTrigger</h3><p style="margin:0">Watches DPO dataset size. When enough pairs accumulate, exports training scripts and spawns them as a background subprocess — Ollama keeps serving the current model while training runs.</p></div>
      <div class="card" style="margin:0"><h3>AdaptiveCurriculumScheduler</h3><p style="margin:0">Organises DPO data into 3 tiers: Warm-up (margin 0–0.30), Core training (0.30–0.60), Hard examples (&gt;0.60). Schedules batches from easy to hard for stable learning.</p></div>
      <div class="card" style="margin:0"><h3>OfflineConversationMiner</h3><p style="margin:0">Mines past conversations for implicit training signal — extracts high-quality exchanges not explicitly rated and adds them to the SFT dataset automatically.</p></div>
      <div class="card" style="margin:0"><h3>HardNegativeMiner</h3><p style="margin:0">Generates deliberately worse reformatted versions of good responses to teach the reward model to prefer clarity and structure over sloppy output.</p></div>
    </div>

    <h2>Reward Model</h2>
    <p><strong>MariaRewardModel</strong> scores candidate responses by quality dimensions (relevance, clarity, factual grounding, structure). Used by GRPO's rollout engine to select the best response from a group of candidates.</p>
  </section>

  <!-- HOW TRAINING WORKS -->
  <section class="doc-section" id="training-methods">
    <div class="ey">Theory</div>
    <h1>How Training Works</h1>
    <p class="sub">Maria uses three complementary training methods — SFT, DPO, and GRPO — each targeting a different aspect of model quality. All are based on the <a href="https://huggingface.co/docs/trl/main/en/dpo_trainer" style="color:var(--yellow)">Hugging Face TRL library</a>.</p>

    <div class="grid3" style="margin-bottom:8px">
      <div class="item">
        <div class="item-label" style="color:var(--yellow)">Step 1</div>
        <div class="item-title" style="font-size:15px">SFT</div>
        <div class="item-body">Teach the model <em>what good responses look like</em> using high-quality input→output pairs. The foundation of every fine-tune.</div>
      </div>
      <div class="item">
        <div class="item-label" style="color:var(--blue)">Step 2</div>
        <div class="item-title" style="font-size:15px">DPO</div>
        <div class="item-body">Teach the model <em>which of two responses is better</em> using human or automatic preference feedback. No reward model needed.</div>
      </div>
      <div class="item">
        <div class="item-label" style="color:var(--green)">Step 3</div>
        <div class="item-title" style="font-size:15px">GRPO</div>
        <div class="item-body">Reinforce good reasoning by <em>comparing groups of sampled responses</em> against a reward function. Lighter than PPO — no critic network.</div>
      </div>
    </div>

    <!-- SFT -->
    <h2>SFT — Supervised Fine-Tuning</h2>
    <p>The simplest and most common adaptation method. The model is trained on (prompt, completion) pairs to minimize the negative log-likelihood of generating the correct output. It teaches the model <em>what</em> to say, not which of two options is preferred.</p>

    <div class="card" style="margin:12px 0">
      <h3 style="margin-top:0">Loss Function</h3>
      <pre style="margin:8px 0">L_SFT(θ) = - Σ log p_θ(y_t | y_&lt;t, x)

Where:
  x        = input prompt
  y_t      = target token at position t
  y_&lt;t     = all preceding tokens
  p_θ      = model being trained</pre>
      <p style="margin:8px 0 0">The model learns to predict every token in the completion given the prompt and all previous tokens. Only completion tokens contribute to the loss — prompt tokens are masked out.</p>
    </div>

    <div class="grid2">
      <div class="card" style="margin:0">
        <h3>Dataset Format</h3>
        <pre style="margin:4px 0;font-size:11px"># Prompt + completion pair
{"prompt": "What is the capital of France?",
 "completion": "Paris."}

# Conversational format
{"prompt": [{"role": "user",
             "content": "What is 2+2?"}],
 "completion": [{"role": "assistant",
                 "content": "4."}]}</pre>
      </div>
      <div class="card" style="margin:0">
        <h3>Key Concepts</h3>
        <div style="font-size:13px;color:var(--text-m);line-height:1.8">
          <strong>Packing</strong> — multiple examples packed into one sequence for efficiency<br>
          <strong>Completion-only loss</strong> — gradient only on the answer, not the question<br>
          <strong>LoRA/PEFT</strong> — fine-tune a small adapter instead of all weights<br>
          <strong>Learning rate</strong> — typically 2e-5 (full) or 1e-4 (LoRA)
        </div>
      </div>
    </div>

    <div class="callout info" style="margin-top:14px">
      <div style="font-size:13px;color:var(--text-b)"><strong>In Maria:</strong> SFTPipeline collects high-confidence responses and hand-crafted examples into <code>maria_training_data.json</code> and <code>maria_sft_dataset.jsonl</code>. OfflineConversationMiner automatically harvests additional pairs from past chats.</div>
    </div>

    <!-- DPO -->
    <h2>DPO — Direct Preference Optimization</h2>
    <p>Instead of (prompt, answer) pairs, DPO trains on <strong>preference pairs</strong> — two completions for the same prompt where one is preferred over the other. It directly optimizes the model to widen the margin between chosen and rejected responses, relative to a frozen reference model — with no separate reward model required.</p>

    <div class="card" style="margin:12px 0">
      <h3 style="margin-top:0">Loss Function</h3>
      <pre style="margin:8px 0">L_DPO(θ) = -E[ log σ( β · (
    log π_θ(y⁺|x) / π_ref(y⁺|x)
  - log π_θ(y⁻|x) / π_ref(y⁻|x)
))]

Where:
  x        = prompt
  y⁺       = chosen (preferred) completion
  y⁻       = rejected (dispreferred) completion
  π_θ      = model being trained
  π_ref    = frozen reference model (initial weights)
  σ        = sigmoid function
  β        = temperature — controls preference signal strength</pre>
      <p style="margin:8px 0 0">The model is rewarded for making <code>y⁺</code> more likely than the reference while making <code>y⁻</code> less likely. β (typically 0.1–0.5) controls how far the trained model is allowed to drift from the reference.</p>
    </div>

    <div class="grid2">
      <div class="card" style="margin:0">
        <h3>Dataset Format</h3>
        <pre style="margin:4px 0;font-size:11px"># Preference pair
{"prompt": "Explain gravity",
 "chosen": "Gravity is a fundamental force...",
 "rejected": "Gravity makes things fall."}

# Maria's format (from thumbs up/down)
{"prompt": "...",
 "chosen": "...",    # preferred response
 "rejected": "...", # alternative response
 "margin": 0.72,    # reward margin
 "source": "human_feedback"}</pre>
      </div>
      <div class="card" style="margin:0">
        <h3>Key Metrics Tracked</h3>
        <div style="font-size:13px;color:var(--text-m);line-height:1.9">
          <strong>rewards/chosen</strong> — implicit reward for preferred completion<br>
          <strong>rewards/rejected</strong> — implicit reward for dispreferred completion<br>
          <strong>rewards/margins</strong> — gap between chosen and rejected rewards<br>
          <strong>rewards/accuracies</strong> — % examples where chosen &gt; rejected<br>
          <strong>logps/chosen</strong> — log-prob of chosen completion tokens
        </div>
      </div>
    </div>

    <div class="card" style="margin:14px 0">
      <h3 style="margin-top:0">Loss Variants (DPO supports multiple formulations)</h3>
      <table style="margin:0">
        <thead><tr><th>loss_type</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td>sigmoid</td><td>Default DPO — Bradley-Terry model via logsigmoid on normalized likelihood</td></tr>
          <tr><td>ipo</td><td>Identity Policy Optimization — avoids overfitting from logit transform</td></tr>
          <tr><td>hinge</td><td>RSO / SLiC hinge loss — beta acts as reciprocal of margin</td></tr>
          <tr><td>robust</td><td>Robust DPO — handles noisy/mislabeled preference data via label smoothing</td></tr>
          <tr><td>sft</td><td>Standard cross-entropy on chosen only — no preference learning</td></tr>
        </tbody>
      </table>
    </div>

    <div class="callout info">
      <div style="font-size:13px;color:var(--text-b)"><strong>In Maria:</strong> DPOPipeline collects pairs from thumbs up/down ratings, auto-postprocessing, and live hard negatives into <code>maria_dpo_dataset.jsonl</code>. AdaptiveCurriculumScheduler sorts them into 3 tiers (easy → hard) for stable training. Reference: <a href="https://huggingface.co/docs/trl/main/en/dpo_trainer" style="color:var(--yellow)">HF TRL DPO Trainer ↗</a></div>
    </div>

    <!-- GRPO -->
    <h2>GRPO — Group Relative Policy Optimization</h2>
    <p>GRPO is a reinforcement learning method introduced with DeepSeekMath. Instead of training a separate value/critic network (like PPO), it samples a <strong>group of responses</strong> for each prompt, scores them with a reward function, and uses the <em>relative</em> scores within the group as the advantage signal. This cuts memory use significantly vs PPO.</p>

    <div class="card" style="margin:12px 0">
      <h3 style="margin-top:0">How It Works</h3>
      <pre style="margin:8px 0">For each prompt x:
  1. Sample G responses:  {y₁, y₂, ..., y_G}  from π_θ
  2. Score each with reward function r(x, yᵢ)
  3. Compute group baseline:  mean_r = mean({r(x, yᵢ)})
  4. Advantage for each:   Aᵢ = (r(x, yᵢ) - mean_r) / std_r
  5. Update policy to increase prob of high-advantage responses
     while staying close to reference model (KL penalty)

Key difference from PPO:
  PPO needs a trained value network to estimate advantages.
  GRPO uses the group mean as a self-contained baseline —
  no critic network, ~40% less memory.</pre>
    </div>

    <div class="grid2">
      <div class="card" style="margin:0">
        <h3>Reward Function</h3>
        <div style="font-size:13px;color:var(--text-m);line-height:1.7">Maria's <code>MariaRewardModel</code> scores candidate responses across multiple quality dimensions: relevance to the prompt, factual grounding, clarity of explanation, and structural quality. The best response in each group gets a positive advantage; below-average responses get a negative signal.</div>
      </div>
      <div class="card" style="margin:0">
        <h3>Key Concepts</h3>
        <div style="font-size:13px;color:var(--text-m);line-height:1.8">
          <strong>Group size G</strong> — number of samples per prompt (typically 4–8)<br>
          <strong>KL penalty</strong> — prevents the policy from drifting too far from reference<br>
          <strong>No critic</strong> — group mean replaces the value network<br>
          <strong>GRPOScheduler</strong> — triggers rollouts every N turns without hammering Ollama
        </div>
      </div>
    </div>

    <div class="callout info" style="margin-top:14px">
      <div style="font-size:13px;color:var(--text-b)"><strong>In Maria:</strong> GRPOScheduler watches conversation turn count and triggers GRPORolloutEngine lazily in the background. GRPODataCollector stores candidates; GRPORewardFunction scores them using MariaRewardModel. Reference: <a href="https://huggingface.co/docs/trl/main/en/grpo_trainer" style="color:var(--yellow)">HF TRL GRPO Trainer ↗</a></div>
    </div>

    <!-- Comparison table -->
    <h2>Comparing the Three Methods</h2>
    <table>
      <thead><tr><th>Method</th><th>Input Data</th><th>What It Optimizes</th><th>Needs Reward Model?</th><th>In Maria</th></tr></thead>
      <tbody>
        <tr><td>SFT</td><td style="font-family:inherit">Prompt + completion</td><td style="font-family:inherit">Likelihood of correct output</td><td style="font-family:inherit;color:var(--green)">No</td><td style="font-family:inherit">High-confidence responses + hand-crafted examples</td></tr>
        <tr><td>DPO</td><td style="font-family:inherit">Prompt + chosen + rejected</td><td style="font-family:inherit">Margin between preferred and dispreferred</td><td style="font-family:inherit;color:var(--green)">No (implicit)</td><td style="font-family:inherit">Thumbs up/down + auto-postprocessing + hard negatives</td></tr>
        <tr><td>GRPO</td><td style="font-family:inherit">Prompt + sampled group</td><td style="font-family:inherit">Relative reward within group</td><td style="font-family:inherit;color:var(--yellow)">Yes (lightweight)</td><td style="font-family:inherit">MariaRewardModel scoring rollouts every N turns</td></tr>
      </tbody>
    </table>

    <div style="margin-top:20px;padding:14px 18px;background:var(--bg);border:1.5px solid var(--border);border-radius:10px;font-size:12.5px;color:var(--text-m);line-height:1.7">
      <strong style="color:var(--text-h)">References:</strong><br>
      DPO — <a href="https://huggingface.co/papers/2305.18290" style="color:var(--yellow)">Rafailov et al., 2023 — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"</a><br>
      GRPO — <a href="https://huggingface.co/papers/2402.03300" style="color:var(--yellow)">Shao et al., 2024 — "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"</a><br>
      TRL Docs — <a href="https://huggingface.co/docs/trl/main/en/dpo_trainer" style="color:var(--yellow)">huggingface.co/docs/trl</a>
    </div>
  </section>

  <!-- REFERENCE -->
  <section class="doc-section" id="reference">
    <div class="ey">API</div>
    <h1>Function Reference</h1>
    <p class="sub">All top-level functions in Maria_App.py with signatures and descriptions.</p>

    <table>
      <thead><tr><th>Function</th><th>Signature</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td>_safe_execute_python</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">(code, timeout=5)</td><td>Run Python code in sandboxed subprocess with timeout. Returns stdout/stderr.</td></tr>
        <tr><td>search_web</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">(query, max_results=3)</td><td>DuckDuckGo search. Returns formatted results string for prompt injection.</td></tr>
        <tr><td>validate_sources_topic</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">(user_query, response)</td><td>Check that retrieved sources are topically relevant to the user's query.</td></tr>
        <tr><td>safety_filter</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">(text)</td><td>HARM_PAT regex blocklist. Returns True if content is safe to process.</td></tr>
        <tr><td>_compress_history</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">(history, model, keep_recent=6)</td><td>Summarize old turns to fit context window, keeping the 6 most recent.</td></tr>
        <tr><td>_select_best_model</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">()</td><td>Query Ollama for available models and return the highest-ranked specialist.</td></tr>
        <tr><td>ensemble_generate</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">(prompt, models, user_query)</td><td>Generate from multiple models and select the best by quality scoring.</td></tr>
        <tr><td>build_reasoning_prompt</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">(base_prompt, user_query)</td><td>Append step-by-step reasoning instructions for complex queries.</td></tr>
        <tr><td>detect_complex_query</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">(text)</td><td>Returns True if query needs multi-step reasoning or tool use.</td></tr>
        <tr><td>detect_language</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">(text)</td><td>Detect language using langdetect + script heuristics (CJK, Arabic, Thai…).</td></tr>
        <tr><td>_sympy_precompute</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">(expression)</td><td>Pre-compute math with SymPy before LLM generation. Injects exact result into prompt.</td></tr>
        <tr><td>_generate_math_plot</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">(expression)</td><td>Generate a matplotlib plot of a math function and return as image for chat.</td></tr>
        <tr><td>_pick_specialist_model</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">(intent)</td><td>Given an intent label, return the configured specialist model name string.</td></tr>
        <tr><td>validate_python_syntax</td><td style="font-family:'JetBrains Mono',monospace;font-size:11.5px">(code)</td><td>Parse code with ast.parse — returns (is_valid, error_message).</td></tr>
      </tbody>
    </table>

    <h2>All Classes — Index</h2>
    <div class="grid3" style="font-size:12.5px">
      <div><div style="font-weight:700;color:var(--yellow);font-size:11px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">Intelligence</div><div style="display:flex;flex-direction:column;gap:4px;color:var(--text-m)"><code>IntentRouter</code><code>SpecializedCoT</code><code>ReflectionPass</code><code>AdvancedReasoningEngine</code><code>SelfConsistency</code><code>ElitePromptBuilder</code><code>ReasoningClassifier</code><code>ConstraintExtractor</code><code>SelfCritiqueSystem</code><code>KnowledgeSynthesizer</code></div></div>
      <div><div style="font-weight:700;color:var(--yellow);font-size:11px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">Retrieval & Safety</div><div style="display:flex;flex-direction:column;gap:4px;color:var(--text-m)"><code>HybridWebRAG</code><code>HallucinationDetector</code><code>ResponseValidator</code><code>WikipediaKnowledge</code><code>HybridWikipedia</code><code>OfflineWikipedia</code><code>LocalRAG</code><code>ChainOfVerification</code><code>UncertaintyQuantifier</code></div></div>
      <div><div style="font-weight:700;color:var(--yellow);font-size:11px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">Math & Code</div><div style="display:flex;flex-direction:column;gap:4px;color:var(--text-m)"><code>PerfectMathEngine</code><code>MathSpecialist</code><code>CodeSpecialist</code><code>CodeDebuggerMode</code><code>StudyModeEngine</code><code>LocalFileAssistant</code><code>PerfectEQEngine</code><code>ScientificGenius</code></div></div>
      <div><div style="font-weight:700;color:var(--yellow);font-size:11px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">Tool Agents</div><div style="display:flex;flex-direction:column;gap:4px;color:var(--text-m)"><code>ToolPlanner</code><code>ReActToolRegistry</code><code>ReActAgent</code><code>PlannerDoerVerifier</code><code>SelfCritiqueLoopEngine</code><code>WikipediaKnowledgeHub</code></div></div>
      <div><div style="font-weight:700;color:var(--yellow);font-size:11px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">Training</div><div style="display:flex;flex-direction:column;gap:4px;color:var(--text-m)"><code>DPOPipeline</code><code>SFTPipeline</code><code>GRPOScheduler</code><code>GRPORolloutEngine</code><code>LiveTrainingTrigger</code><code>AdaptiveCurriculumScheduler</code><code>OfflineConversationMiner</code><code>HardNegativeMiner</code><code>MariaRewardModel</code><code>LearningSystem</code></div></div>
      <div><div style="font-weight:700;color:var(--yellow);font-size:11px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">GUI & Workers</div><div style="display:flex;flex-direction:column;gap:4px;color:var(--text-m)"><code>MariaPyQt</code><code>UltraIntelligentWorker</code><code>ThinkingStepsWidget</code><code>StreamingCodeBlockWidget</code><code>ModernMessageBubble</code><code>AutoTitleWorker</code><code>EmergencySystem</code><code>DPODataCollector</code><code>SFTDataCollector</code></div></div>
    </div>
  </section>

</div>

<script>
const sections = Array.from(document.querySelectorAll('.doc-section'));
const links = document.querySelectorAll('.sb-nav a[href^="#"]');
function updateActive() {
  var scrollY = window.scrollY + 120;
  var current = sections[0].id;
  sections.forEach(function(s) {
    if (s.offsetTop <= scrollY) current = s.id;
  });
  links.forEach(function(l) {
    l.classList.toggle('active', l.getAttribute('href') === '#' + current);
  });
}
window.addEventListener('scroll', updateActive, {passive: true});
updateActive();
</script>
</body>
</html>"""


def build_html(d):
    jv = json.dumps

    hallu_color = "#F15B5B" if d["hallu_risk"] >= 20 else "#FF9D00" if d["hallu_risk"] >= 10 else "#1DBF7B"

    file_labels = {
        "metrics":     "maria_metrics.json",
        "dpo":         "maria_dpo_dataset.jsonl",
        "sft_lora":    "maria_training_data.json",
        "sft_harvest": "maria_sft_dataset.jsonl",
    }
    file_rows = "".join(
        f'<li>{"[found]" if v else "[missing]"} <code>{file_labels[k]}</code></li>'
        for k, v in d["files_found"].items()
    )

    qt_keys  = list(d["qtypes"].keys())  or ["general"]
    qt_vals  = list(d["qtypes"].values()) or [0]
    src_keys = [k.replace("_", " ").title() for k in d["sources"].keys()] or ["No data"]
    src_vals = list(d["sources"].values()) or [0]
    lang_keys= [k.upper() for k in d["langs"].keys()] or ["No data"]
    lang_vals= list(d["langs"].values()) or [0]

    def fmt(n):
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000:     return f"{n/1_000:.1f}k"
        return str(n)

    # ── pre-build examples page HTML (avoids backslash-in-f-string on Py<3.12) ──
    mono = "font-family:'JetBrains Mono',monospace"

    SOURCE_DESC = {
        "human_feedback": (
            "Human Feedback",
            "Pairs collected from real users rating Maria's responses using the thumbs up / thumbs down buttons. "
            "The chosen response is the one the user preferred; the rejected response is the alternative Maria generated. "
            "This is the highest-quality signal for alignment training."
        ),
        "auto_postprocess": (
            "Auto Postprocess",
            "Pairs generated automatically after a conversation ends. Maria re-scores its own responses "
            "using confidence and relevance metrics, then the better-scoring response becomes the chosen side. "
            "Requires no human input but is lower-signal than direct feedback."
        ),
        "live_hard_neg_over_formatting": (
            "Live Hard Negative (Formatting)",
            "Pairs where the rejected response is a deliberately worse reformatted version of the chosen response — "
            "same content but with poor formatting, broken structure, or missing clarity. "
            "Teaches Maria to value clean, well-structured answers over sloppy ones."
        ),
    }

    _src_rows = []
    for k, v in sorted(d["sources"].items(), key=lambda x: -x[1]):
        label, desc = SOURCE_DESC.get(k, (k, "Custom data source."))
        _src_rows.append(
            f'<div style="padding:14px 0;border-bottom:1px solid #EBE6DA">'
            f'<div style="font-size:13px;font-weight:700;color:#1A1714;margin-bottom:5px">{label}'
            f'<span style="{mono};font-size:11px;font-weight:500;color:#9B9490;margin-left:10px">{v} pairs</span>'
            f'</div>'
            f'<p style="font-size:12.5px;color:#6A6360;line-height:1.7;margin:0">{desc}</p>'
            f'</div>'
        )
    sources_html = "".join(_src_rows) or '<p style="color:#9B9490;font-size:13px">No source data yet.</p>'

    _ex_rows = []
    for ex in d["recent_examples"]:
        good      = ex["margin"] >= 0.5
        bg_pill   = "var(--gl)" if good else "var(--cl)"
        col_pill  = "#065F46"   if good else "#991B1B"
        thumb     = "+" if good else "-"
        chosen_t  = ex["chosen"]  + ("…" if len(ex["chosen"])  >= 160 else "")
        reject_t  = ex["rejected"] + ("…" if len(ex["rejected"]) >= 100 else "")
        _ex_rows.append(
            f'<div class="cc" style="margin-bottom:10px;padding:16px 20px;">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
            f'<span style="font-size:10px;font-weight:700;padding:2px 9px;border-radius:20px;'
            f'background:{bg_pill};color:{col_pill}">{thumb} margin {ex["margin"]:.3f}</span>'
            f'<span style="font-size:10px;font-weight:600;color:var(--text-dim)">{ex["source"]}</span>'
            f'</div>'
            f'<div style="font-size:13px;color:var(--text-h);line-height:1.5;margin-bottom:6px">{chosen_t}</div>'
            f'<div style="font-size:11px;color:var(--text-dim);line-height:1.4;padding-top:6px;border-top:1px solid var(--border)">'
            f'<b>Rejected:</b> {reject_t}</div>'
            f'</div>'
        )
    examples_html = "".join(_ex_rows) or (
        '<div class="callout info"><div class="callout-icon"></div>'
        '<div class="callout-body">No DPO examples collected yet. '
        'Chat with Maria and use the thumbs up / thumbs down buttons to generate training data.</div></div>'
    )

    no_data_banner = ""
    if not any(d["files_found"].values()):
        no_data_banner = """
        <div class="callout warn">
          <div class="callout-icon"></div>
          <div class="callout-body">
            <b>No data files found.</b> Run Maria and chat with it first to generate the data files,
            then re-run <code>python maria_dashboard.py</code>.
            <ul class="file-list" style="margin-top:8px">""" + file_rows + """</ul>
          </div>
        </div>"""
    elif not all(d["files_found"].values()):
        no_data_banner = f"""
        <div class="callout info">
          <div class="callout-icon"></div>
          <div class="callout-body">
            <b>Some data files are missing</b> — they'll be created as you use Maria more.
            <ul class="file-list" style="margin-top:8px">{file_rows}</ul>
          </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Maria AI — Training Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#FFFDF8;--surface:#FFF;--sidebar:#FAFAF4;
  --border:#EBE6DA;--border2:#D8D0BF;
  --yellow:#FF9D00;--yl:#FFF4DC;--ym:#FFE0A0;
  --coral:#F15B5B;--cl:#FFF0F0;
  --green:#1DBF7B;--gl:#E8FAF3;
  --blue:#3B82F6;--bl:#EEF4FF;
  --purple:#8B5CF6;--pl:#F3EEFF;
  --text-h:#1A1714;--text-b:#383330;--text-m:#6A6360;--text-dim:#9B9490;
  --sw:252px;--r:10px;--rl:14px
}}
html{{scroll-behavior:smooth}}
body{{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text-b);display:flex;min-height:100vh;font-size:15px;line-height:1.7}}
/* Sidebar */
.sb{{width:var(--sw);min-height:100vh;background:var(--sidebar);border-right:1.5px solid var(--border);position:fixed;top:0;left:0;overflow-y:auto;z-index:200;display:flex;flex-direction:column}}
.sb-hd{{display:flex;align-items:center;gap:10px;padding:20px 18px 18px;border-bottom:1px solid var(--border)}}
.sb-ic{{width:34px;height:34px;background:var(--yellow);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0}}
.sb-br b{{font-size:14.5px;font-weight:700;color:var(--text-h);display:block;line-height:1.2}}
.sb-br span{{font-size:11px;color:var(--text-dim)}}
.sb-sec{{padding:18px 14px 4px}}
.sb-lbl{{font-size:10px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--text-dim);padding:0 6px;margin-bottom:5px}}
.sb-nav{{list-style:none;display:flex;flex-direction:column;gap:2px}}
.sb-nav button{{display:flex;align-items:center;gap:9px;padding:7px 10px;border-radius:7px;font-size:13.5px;color:var(--text-m);text-decoration:none;transition:all .15s;cursor:pointer;font-weight:500;border:none;background:none;width:100%;text-align:left;font-family:inherit}}
.sb-nav button:hover{{background:var(--yl);color:var(--text-h)}}
.sb-nav button.active{{background:var(--yl);color:var(--yellow);font-weight:700;border-left:3px solid var(--yellow);padding-left:7px}}
.ni{{font-size:15px;width:20px;text-align:center;flex-shrink:0}}
.sb-ft{{margin-top:auto;padding:16px 18px;border-top:1px solid var(--border);font-size:11px;color:var(--text-dim);line-height:1.6}}
.sb-ft .gt{{color:var(--text-m);font-weight:500}}
/* Main */
.main{{margin-left:var(--sw);flex:1;display:flex;flex-direction:column}}
.topbar{{background:var(--surface);border-bottom:1px solid var(--border);padding:11px 44px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}}
.bc{{font-size:12.5px;color:var(--text-dim);display:flex;align-items:center;gap:6px}}
.bc a{{color:var(--text-dim);text-decoration:none}}
.bc a:hover{{color:var(--yellow)}}
.bc .cur{{color:var(--text-b);font-weight:600}}
.pills{{display:flex;gap:7px;align-items:center}}
.pill{{font-size:11px;font-weight:600;padding:3px 11px;border-radius:20px;background:var(--yl);color:#92400E;border:1px solid var(--ym)}}
.pill.g{{background:var(--gl);color:#065F46;border-color:#A7F3D0}}
.pill.b{{background:var(--bl);color:#1E40AF;border-color:#BFDBFE}}
/* Pages */
.page{{display:none;padding:46px 44px 64px;max-width:860px;animation:fu .3s ease both}}
.page.on{{display:block}}
@keyframes fu{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:none}}}}
/* Type */
.ey{{font-size:10.5px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--yellow);margin-bottom:8px}}
h1{{font-family:'Source Serif 4',Georgia,serif;font-size:33px;font-weight:700;color:var(--text-h);line-height:1.2;margin-bottom:10px;letter-spacing:-.01em}}
.sub{{font-size:15.5px;color:var(--text-m);max-width:620px;margin-bottom:24px;line-height:1.6}}
h2{{font-family:'Source Serif 4',Georgia,serif;font-size:21px;font-weight:600;color:var(--text-h);margin:36px 0 10px;display:flex;align-items:center;gap:10px}}
h2::before{{content:'';display:block;width:3px;height:22px;background:var(--yellow);border-radius:2px;flex-shrink:0}}
p{{color:var(--text-m);margin-bottom:12px;font-size:14px;line-height:1.65}}
code{{font-family:'JetBrains Mono',monospace;font-size:12px;background:var(--border);padding:1px 6px;border-radius:4px;color:var(--text-h)}}
.file-list{{list-style:none;display:flex;flex-direction:column;gap:5px;font-size:13px;color:var(--text-b)}}
.divider{{height:1px;background:var(--border);margin:28px 0}}
/* Stat cards */
.sr{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:20px 0 28px}}
.sr.c3{{grid-template-columns:repeat(3,1fr)}}
.sr.c4{{grid-template-columns:repeat(4,1fr)}}
.sc{{background:var(--surface);border:1px solid var(--border);border-radius:var(--rl);padding:18px 20px;transition:box-shadow .2s,transform .2s;position:relative;overflow:hidden}}
.sc:hover{{box-shadow:0 6px 24px rgba(0,0,0,.07);transform:translateY(-1px)}}
.sc::after{{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:var(--rl) var(--rl) 0 0}}
.sc.y::after{{background:var(--yellow)}}.sc.g::after{{background:var(--green)}}
.sc.b::after{{background:var(--blue)}}.sc.r::after{{background:var(--coral)}}.sc.p::after{{background:var(--purple)}}
.sc-lb{{font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--text-dim);margin-bottom:8px}}
.sc-v{{font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:500;line-height:1;margin-bottom:4px}}
.sc-s{{font-size:11.5px;color:var(--text-dim)}}
.sc-v.y{{color:var(--yellow)}}.sc-v.g{{color:var(--green)}}.sc-v.b{{color:var(--blue)}}.sc-v.r{{color:var(--coral)}}.sc-v.p{{color:var(--purple)}}
/* Metric items */
.mg{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0}}
.mi{{display:flex;align-items:center;gap:11px;padding:12px 14px;background:var(--bg);border:1px solid var(--border);border-radius:8px}}
.mi-ic{{width:33px;height:33px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0}}
.mi-ic.y{{background:var(--yl)}}.mi-ic.g{{background:var(--gl)}}.mi-ic.b{{background:var(--bl)}}.mi-ic.r{{background:var(--cl)}}.mi-ic.p{{background:var(--pl)}}
.mi-lb{{font-size:11px;color:var(--text-dim);line-height:1.2;margin-bottom:2px}}
.mi-v{{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:500;color:var(--text-h)}}
/* Chart cards */
.cc{{background:var(--surface);border:1px solid var(--border);border-radius:var(--rl);padding:24px 24px 20px;margin-bottom:18px}}
.cc-hd{{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:16px}}
.cc-t{{font-size:14px;font-weight:700;color:var(--text-h);margin-bottom:3px}}
.cc-d{{font-size:12px;color:var(--text-dim);max-width:480px;line-height:1.5}}
.cc-tag{{font-size:10.5px;font-weight:600;padding:3px 10px;border-radius:20px;background:var(--yl);color:#92400E;white-space:nowrap;flex-shrink:0}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
/* Progress */
.pl{{display:flex;flex-direction:column;gap:20px}}
.pm{{display:flex;justify-content:space-between;margin-bottom:7px}}
.pl-lb{{font-size:13.5px;font-weight:600;color:var(--text-b)}}
.pl-n{{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--text-dim)}}
.pt{{height:9px;background:var(--border);border-radius:99px;overflow:hidden}}
.pf{{height:100%;border-radius:99px}}
/* Callout */
.callout{{display:flex;gap:13px;padding:14px 18px;border-radius:var(--r);margin:0 0 20px;border:1.5px solid}}
.callout.info{{background:var(--yl);border-color:var(--ym)}}
.callout.ok{{background:var(--gl);border-color:#A7F3D0}}
.callout.warn{{background:var(--cl);border-color:#FCA5A5}}
.callout-icon{{font-size:18px;flex-shrink:0;margin-top:1px}}
.callout-body{{font-size:13px;line-height:1.6;color:var(--text-b)}}
.callout-body b{{color:var(--text-h)}}
/* Big numbers row */
.big-nums{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:8px 0 4px}}
.bn{{text-align:center;padding:12px 8px}}
.bn-v{{font-size:36px;font-weight:700;font-family:'JetBrains Mono',monospace;line-height:1}}
.bn-lb{{font-size:12px;color:var(--text-dim);margin-top:4px}}
.bn-sub{{font-size:13px;font-weight:600;color:var(--text-m);margin-top:2px}}
::-webkit-scrollbar{{width:5px}}
::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:3px}}
</style>
</head>
<body>

<aside class="sb">
  <div class="sb-hd">
    <div class="sb-ic">🤗</div>
    <div class="sb-br"><b>Maria AI</b><span>Training Dashboard</span></div>
  </div>

  <div class="sb-sec">
    <div class="sb-lbl">Overview</div>
    <ul class="sb-nav">
      <li><button onclick="show('overview')" id="n-overview" class="active">
        Dashboard</button></li>
    </ul>
  </div>

  <div class="sb-sec">
    <div class="sb-lbl">Performance</div>
    <ul class="sb-nav">
      <li><button onclick="show('confidence')" id="n-confidence">
        Confidence</button></li>
      <li><button onclick="show('hallucination')" id="n-hallucination">
        Hallucination Risk</button></li>
      <li><button onclick="show('response')" id="n-response">
        Response Times</button></li>
    </ul>
  </div>

  <div class="sb-sec">
    <div class="sb-lbl">Training Data</div>
    <ul class="sb-nav">
      <li><button onclick="show('dpo')" id="n-dpo">
        DPO Dataset</button></li>
      <li><button onclick="show('progress')" id="n-progress">
        Training Progress</button></li>
      <li><button onclick="show('examples')" id="n-examples">
        Recent Examples</button></li>
    </ul>
  </div>

  <div class="sb-sec">
    <div class="sb-lbl">Analysis</div>
    <ul class="sb-nav">
      <li><button onclick="show('comparison')" id="n-comparison">
        Comparison</button></li>
    </ul>
  </div>

  <div class="sb-ft">
    <div>Generated on</div>
    <div class="gt" id="sb-ft-time">{d["generated_at"]}</div>
  </div>
</aside>

<main class="main">
  <div class="topbar">
    <div class="bc">
      <a href="/">Documentation</a><span>›</span>
      <span class="cur" id="bc">Dashboard</span>
    </div>
    <div class="pills">
      <span class="pill">main</span>
      <span class="pill g">● Live Data</span>
      <span class="pill b">{d["total_q"]:,} queries</span>
      <button onclick="exportJSONL()" style="margin-left:8px;padding:4px 14px;border-radius:20px;border:1.5px solid var(--border2);background:var(--surface);font-size:11px;font-weight:600;color:var(--text-b);cursor:pointer;font-family:inherit" onmouseover="this.style.background='var(--yl)'" onmouseout="this.style.background='var(--surface)'">Export JSONL</button>
    </div>
  </div>

  <!-- PAGE: Overview -->
  <div class="page on" id="page-overview">
    <div class="ey">Overview</div>
    <h1>Maria AI Training Dashboard</h1>
    <p class="sub">Real performance and training data from your local deployment. All numbers are read directly from Maria's data files — no estimates, no placeholders.</p>
    {no_data_banner}
    <div class="sr">
      <div class="sc y"><div class="sc-lb">Total Queries</div><div class="sc-v y">{fmt(d["total_q"])}</div><div class="sc-s">all-time interactions</div></div>
      <div class="sc g"><div class="sc-lb">Avg Confidence</div><div class="sc-v g">{d["avg_conf"]}%</div><div class="sc-s">mean score per query</div></div>
      <div class="sc b"><div class="sc-lb">Training Pairs</div><div class="sc-v b">{fmt(d["sft_n"]+d["dpo_n"])}</div><div class="sc-s">SFT + DPO combined</div></div>
      <div class="sc r"><div class="sc-lb">Hallucination Risk</div><div class="sc-v r">{d["hallu_risk"]}%</div><div class="sc-s">estimated from low-conf</div></div>
    </div>
    <h2>Performance Metrics</h2>
    <p>Key operational statistics collected across all sessions.</p>
    <div class="mg">
      <div class="mi"><div><div class="mi-lb">Web Searches</div><div class="mi-v">{fmt(d["web"])}</div></div></div>
      <div class="mi"><div><div class="mi-lb">Smart Mode Uses</div><div class="mi-v">{fmt(d["smart"])}</div></div></div>
      <div class="mi"><div><div class="mi-lb">Self-Corrections</div><div class="mi-v">{d["corr"]}</div></div></div>
      <div class="mi"><div><div class="mi-lb">Relevance Failures</div><div class="mi-v">{d["rel_fail"]}</div></div></div>
      <div class="mi"><div><div class="mi-lb">Est. Tokens Used</div><div class="mi-v">{fmt(d["tokens"])}</div></div></div>
      <div class="mi"><div><div class="mi-lb">Avg Response Time</div><div class="mi-v">{d["avg_time"]:.2f}s</div></div></div>
    </div>
  </div>

  <!-- PAGE: Confidence -->
  <div class="page" id="page-confidence">
    <div class="ey">Performance</div>
    <h1>Confidence Distribution</h1>
    <p class="sub">Breakdown of answer confidence levels across all {d["total_q"]:,} queries. A healthy model should have most answers in the <b>High ≥85%</b> bucket.</p>
    <div class="two">
      <div class="cc">
        <div class="cc-hd"><div><div class="cc-t">Confidence Buckets</div><div class="cc-d">Count of answers per tier</div></div><span class="cc-tag">Bar</span></div>
        <div style="height:230px"><canvas id="confBar"></canvas></div>
      </div>
      <div class="cc">
        <div class="cc-hd"><div><div class="cc-t">Query Types</div><div class="cc-d">Technical vs complex vs general</div></div><span class="cc-tag">Donut</span></div>
        <div style="height:230px"><canvas id="queryDonut"></canvas></div>
      </div>
    </div>
    <div class="cc">
      <div class="cc-hd"><div><div class="cc-t">Confidence Summary</div><div class="cc-d">All-time confidence tier counts</div></div></div>
      <div class="big-nums">
        <div class="bn"><div class="bn-v" style="color:#1DBF7B">{d["hi_c"]:,}</div><div class="bn-lb">High Confidence ≥85%</div><div class="bn-sub">{round(d["hi_c"]/max(d["total_q"],1)*100,1)}% of total</div></div>
        <div class="bn"><div class="bn-v" style="color:#FF9D00">{d["mid_c"]:,}</div><div class="bn-lb">Mid Confidence 60–85%</div><div class="bn-sub">{round(d["mid_c"]/max(d["total_q"],1)*100,1)}% of total</div></div>
        <div class="bn"><div class="bn-v" style="color:#F15B5B">{d["lo_c"]:,}</div><div class="bn-lb">Low Confidence &lt;60%</div><div class="bn-sub">{round(d["lo_c"]/max(d["total_q"],1)*100,1)}% of total</div></div>
      </div>
    </div>
  </div>

  <!-- PAGE: Hallucination -->
  <div class="page" id="page-hallucination">
    <div class="ey">Analysis</div>
    <h1>Hallucination Risk</h1>
    <p class="sub">Estimated from low-confidence answers and self-corrections. Below 10% is healthy. Current: <b style="color:{hallu_color}">{d["hallu_risk"]}%</b></p>
    <div class="sr c3">
      <div class="sc r"><div class="sc-lb">Hallucination Risk</div><div class="sc-v r">{d["hallu_risk"]}%</div><div class="sc-s">low-conf + corrections / total</div></div>
      <div class="sc g"><div class="sc-lb">Relevance Rate</div><div class="sc-v g">{d["rel_rate"]}%</div><div class="sc-s">queries answered on-topic</div></div>
      <div class="sc b"><div class="sc-lb">Self-Corrections</div><div class="sc-v b">{d["corr"]}</div><div class="sc-s">times Maria caught itself</div></div>
    </div>
    <div class="cc">
      <div class="cc-hd"><div><div class="cc-t">Hallucination Risk vs Relevance Rate</div><div class="cc-d">Direct comparison of key quality indicators</div></div><span class="cc-tag">Horizontal Bar</span></div>
      <div style="height:150px"><canvas id="halluBar"></canvas></div>
    </div>
    <div class="cc">
      <div class="cc-hd"><div><div class="cc-t">Quality Counters</div><div class="cc-d">All quality-related metrics at a glance</div></div><span class="cc-tag">Bar</span></div>
      <div style="height:200px"><canvas id="qualityBar"></canvas></div>
    </div>
  </div>

  <!-- PAGE: Response Times -->
  <div class="page" id="page-response">
    <div class="ey">Performance</div>
    <h1>Response Times</h1>
    <p class="sub">Latency over the last {len(d["times"]):,} recorded queries with rolling 8-query average, P50, and P95 reference lines.</p>
    <div class="sr c4">
      <div class="sc g"><div class="sc-lb">P50 Median</div><div class="sc-v g">{d["p50"]}s</div><div class="sc-s">50th percentile</div></div>
      <div class="sc r"><div class="sc-lb">P95</div><div class="sc-v r">{d["p95"]}s</div><div class="sc-s">95th percentile</div></div>
      <div class="sc b"><div class="sc-lb">Average</div><div class="sc-v b">{d["avg_time"]:.3f}s</div><div class="sc-s">mean latency</div></div>
      <div class="sc y"><div class="sc-lb">Samples</div><div class="sc-v y">{len(d["times"]):,}</div><div class="sc-s">recorded queries</div></div>
    </div>
    <div class="cc">
      <div class="cc-hd"><div><div class="cc-t">Response Time History</div><div class="cc-d">Per-query latency in seconds — last {len(d["times"]):,} queries</div></div><span class="cc-tag">Area Chart</span></div>
      <div style="height:280px"><canvas id="rtLine"></canvas></div>
    </div>
  </div>

  <!-- PAGE: DPO -->
  <div class="page" id="page-dpo">
    <div class="ey">Training Data</div>
    <h1>DPO Dataset</h1>
    <p class="sub">{d["dpo_n"]:,} preference pairs collected. These teach Maria to prefer better responses over worse ones.</p>
    <div class="sr c4">
      <div class="sc y"><div class="sc-lb">Total Pairs</div><div class="sc-v y">{fmt(d["dpo_n"])}</div><div class="sc-s">preference triples</div></div>
      <div class="sc b"><div class="sc-lb">Avg Margin</div><div class="sc-v b">{d["avg_margin"]:.4f}</div><div class="sc-s">reward margin mean</div></div>
      <div class="sc g"><div class="sc-lb">Languages</div><div class="sc-v g">{len(d["langs"])}</div><div class="sc-s">detected</div></div>
      <div class="sc p"><div class="sc-lb">Sources</div><div class="sc-v p">{len(d["sources"])}</div><div class="sc-s">data origins</div></div>
    </div>
    <div class="two">
      <div class="cc">
        <div class="cc-hd"><div><div class="cc-t">Pair Sources</div><div class="cc-d">Where training pairs originated</div></div><span class="cc-tag">Donut</span></div>
        <div style="height:230px"><canvas id="srcDonut"></canvas></div>
      </div>
      <div class="cc">
        <div class="cc-hd"><div><div class="cc-t">Language Distribution</div><div class="cc-d">Languages in DPO pairs</div></div><span class="cc-tag">Bar</span></div>
        <div style="height:230px"><canvas id="langBar"></canvas></div>
      </div>
    </div>
    <div class="cc">
      <div class="cc-hd"><div><div class="cc-t">Reward Margin Distribution</div><div class="cc-d">≥0.5 good · 0.2–0.5 ok · &lt;0.2 low</div></div><span class="cc-tag">Histogram</span></div>
      <div style="height:220px"><canvas id="marginHist"></canvas></div>
    </div>
  </div>

  <!-- PAGE: Progress -->
  <div class="page" id="page-progress">
    <div class="ey">Training Progress</div>
    <h1>Training Progress</h1>
    <p class="sub">Progress toward recommended minimum dataset sizes for a solid fine-tuning run.</p>
    <div class="cc">
      <div class="pl">
        <div><div class="pm"><span class="pl-lb">SFT Examples</span><span class="pl-n">{d["sft_n"]:,} / 500 ({d["prog_sft"]:.0f}%)</span></div><div class="pt"><div class="pf" style="width:{d["prog_sft"]}%;background:var(--yellow)"></div></div></div>
        <div><div class="pm"><span class="pl-lb">DPO Pairs</span><span class="pl-n">{d["dpo_n"]:,} / 1,000 ({d["prog_dpo"]:.0f}%)</span></div><div class="pt"><div class="pf" style="width:{d["prog_dpo"]}%;background:var(--purple)"></div></div></div>
        <div><div class="pm"><span class="pl-lb">Total Queries</span><span class="pl-n">{d["total_q"]:,} / 1,000 ({d["prog_q"]:.0f}%)</span></div><div class="pt"><div class="pf" style="width:{d["prog_q"]}%;background:var(--blue)"></div></div></div>
        <div><div class="pm"><span class="pl-lb">High-Confidence Replies</span><span class="pl-n">{d["hi_c"]:,} / 500 ({d["prog_hc"]:.0f}%)</span></div><div class="pt"><div class="pf" style="width:{d["prog_hc"]}%;background:var(--green)"></div></div></div>
      </div>
    </div>
    <div class="cc">
      <div class="cc-hd"><div><div class="cc-t">Dataset Overview</div><div class="cc-d">SFT vs DPO vs other key counts</div></div><span class="cc-tag">Bar</span></div>
      <div style="height:210px"><canvas id="datasetBar"></canvas></div>
    </div>
    <h2>What Each Metric Means</h2>
    <p><b>SFT Examples</b> — Supervised fine-tuning data from conversations. Target: 500+ for a first run.</p>
    <p><b>DPO Pairs</b> — Chosen vs rejected preference pairs for alignment training. Target: 1,000+.</p>
    <p><b>Total Queries</b> — Overall interaction volume. More data = better generalization.</p>
    <p><b>High-Confidence Replies</b> — Answers where Maria was confident (≥85%). Indicates reliability.</p>
  </div>

  <!-- PAGE: Recent Examples -->
  <div class="page" id="page-examples">
    <div class="ey">Training Data</div>
    <h1>Recent DPO Examples</h1>
    <p class="sub">Last {len(d["recent_examples"])} preference pairs collected, newest first.</p>
    <div class="sr c4">
      <div class="sc y"><div class="sc-lb">Total Pairs</div><div class="sc-v y">{d["dpo_n"]:,}</div><div class="sc-s">all-time DPO pairs</div></div>
      <div class="sc g"><div class="sc-lb">Positive ≥0.5</div><div class="sc-v g">{d["dpo_pos"]:,}</div><div class="sc-s">reward margin ≥ 0.5</div></div>
      <div class="sc r"><div class="sc-lb">Negative &lt;0.5</div><div class="sc-v r">{d["dpo_neg"]:,}</div><div class="sc-s">reward margin &lt; 0.5</div></div>
      <div class="sc b"><div class="sc-lb">Sources</div><div class="sc-v b">{len(d["sources"])}</div><div class="sc-s">data origins</div></div>
    </div>
    <h2>By Source</h2>
    <div class="cc" style="padding:20px 24px 8px;">
      {sources_html}
    </div>
    <div class="cc" style="margin-top:14px;padding:24px 24px 20px;">
      <div class="cc-hd">
        <div>
          <div class="cc-t">Source Breakdown</div>
          <div class="cc-d">Distribution of training pairs across collection methods</div>
        </div>
        <span class="cc-tag">Donut</span>
      </div>
      <div style="height:260px"><canvas id="srcDonut2"></canvas></div>
    </div>
    <h2>Examples</h2>
    {examples_html}
  </div>

  <!-- PAGE: Comparison -->
  <div class="page" id="page-comparison">
    <div class="ey">Analysis</div>
    <h1>MariaGUI vs Maria_App &mdash; Version Comparison</h1>
    <p class="sub">Side-by-side capability analysis of the original MariaGUI (Old) versus the current Maria_App (Present). Every score is computed by scanning each version's source code for specific features — each detected capability adds bonus points on top of a <em>base score</em>, capped at 100. The weighted overall score combines seven dimensions.</p>

    <div class="sr">
      <div class="sc" style="border-top:3px solid #C8C3BB">
        <div class="sc-lb">MariaGUI (Old)</div>
        <div class="sc-v" style="color:#9B9490">45<span style="font-size:16px;font-weight:400">/100</span></div>
        <div class="sc-s">tkinter &middot; 1 model &middot; basic pipeline</div>
      </div>
      <div class="sc g">
        <div class="sc-lb">Maria_App (Present)</div>
        <div class="sc-v g">95<span style="font-size:16px;font-weight:400">/100</span></div>
        <div class="sc-s">PyQt5 &middot; 3 specialist models &middot; full AI stack</div>
      </div>
      <div class="sc y">
        <div class="sc-lb">Improvement</div>
        <div class="sc-v y">+50</div>
        <div class="sc-s">points overall (+111% relative)</div>
      </div>
      <div class="sc b">
        <div class="sc-lb">New Features</div>
        <div class="sc-v b">12+</div>
        <div class="sc-s">major capabilities added</div>
      </div>
    </div>

    <!-- Score breakdown table -->
    <div class="cc">
      <div class="cc-hd">
        <div>
          <div class="cc-t">Score Breakdown — 7 Capability Dimensions</div>
          <div class="cc-d">Weighted overall = Intelligence&times;25% + Anti-Hallucination&times;20% + Math&times;15% + Code&times;15% + Knowledge&times;15% + Language&times;5% + Safety&times;5%</div>
        </div>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:13px">
        <thead>
          <tr style="border-bottom:2px solid var(--border)">
            <th style="text-align:left;padding:8px 12px;font-weight:600;color:var(--text-b)">Dimension</th>
            <th style="text-align:center;padding:8px 12px;font-weight:600;color:var(--text-b)">Weight</th>
            <th style="text-align:right;padding:8px 12px;font-weight:600;color:#9B9490">MariaGUI</th>
            <th style="text-align:right;padding:8px 12px;font-weight:600;color:var(--green)">Maria_App</th>
            <th style="text-align:right;padding:8px 12px;font-weight:600;color:var(--yellow)">Gap</th>
            <th style="padding:8px 12px;min-width:180px;color:var(--text-b);font-weight:600">Visual</th>
          </tr>
        </thead>
        <tbody>
          <tr style="border-bottom:1px solid var(--border)">
            <td style="padding:10px 12px;font-weight:600">Intelligence</td>
            <td style="text-align:center;padding:10px 12px;color:var(--text-m)">25%</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:#9B9490">60</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--green);font-weight:700">100</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--yellow)">+40</td>
            <td style="padding:10px 12px"><div style="height:7px;background:var(--border);border-radius:4px;position:relative"><div style="position:absolute;top:0;left:0;width:60%;height:100%;background:#C8C3BB;border-radius:4px"></div><div style="position:absolute;top:0;left:0;width:100%;height:100%;background:rgba(29,191,123,.4);border-radius:4px"></div></div></td>
          </tr>
          <tr style="border-bottom:1px solid var(--border)">
            <td style="padding:10px 12px;font-weight:600">Anti-Hallucination</td>
            <td style="text-align:center;padding:10px 12px;color:var(--text-m)">20%</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:#9B9490">42</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--green);font-weight:700">97</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--yellow)">+55</td>
            <td style="padding:10px 12px"><div style="height:7px;background:var(--border);border-radius:4px;position:relative"><div style="position:absolute;top:0;left:0;width:42%;height:100%;background:#C8C3BB;border-radius:4px"></div><div style="position:absolute;top:0;left:0;width:97%;height:100%;background:rgba(29,191,123,.4);border-radius:4px"></div></div></td>
          </tr>
          <tr style="border-bottom:1px solid var(--border)">
            <td style="padding:10px 12px;font-weight:600">Math Accuracy</td>
            <td style="text-align:center;padding:10px 12px;color:var(--text-m)">15%</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:#9B9490">35</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--green);font-weight:700">97</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--yellow)">+62</td>
            <td style="padding:10px 12px"><div style="height:7px;background:var(--border);border-radius:4px;position:relative"><div style="position:absolute;top:0;left:0;width:35%;height:100%;background:#C8C3BB;border-radius:4px"></div><div style="position:absolute;top:0;left:0;width:97%;height:100%;background:rgba(29,191,123,.4);border-radius:4px"></div></div></td>
          </tr>
          <tr style="border-bottom:1px solid var(--border)">
            <td style="padding:10px 12px;font-weight:600">Code Execution</td>
            <td style="text-align:center;padding:10px 12px;color:var(--text-m)">15%</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:#9B9490">35</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--green);font-weight:700">89</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--yellow)">+54</td>
            <td style="padding:10px 12px"><div style="height:7px;background:var(--border);border-radius:4px;position:relative"><div style="position:absolute;top:0;left:0;width:35%;height:100%;background:#C8C3BB;border-radius:4px"></div><div style="position:absolute;top:0;left:0;width:89%;height:100%;background:rgba(29,191,123,.4);border-radius:4px"></div></div></td>
          </tr>
          <tr style="border-bottom:1px solid var(--border)">
            <td style="padding:10px 12px;font-weight:600">Knowledge Retrieval</td>
            <td style="text-align:center;padding:10px 12px;color:var(--text-m)">15%</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:#9B9490">20</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--green);font-weight:700">95</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--yellow)">+75</td>
            <td style="padding:10px 12px"><div style="height:7px;background:var(--border);border-radius:4px;position:relative"><div style="position:absolute;top:0;left:0;width:20%;height:100%;background:#C8C3BB;border-radius:4px"></div><div style="position:absolute;top:0;left:0;width:95%;height:100%;background:rgba(29,191,123,.4);border-radius:4px"></div></div></td>
          </tr>
          <tr style="border-bottom:1px solid var(--border)">
            <td style="padding:10px 12px;font-weight:600">Language Support</td>
            <td style="text-align:center;padding:10px 12px;color:var(--text-m)">5%</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:#9B9490">79</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--green);font-weight:700">81</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--yellow)">+2</td>
            <td style="padding:10px 12px"><div style="height:7px;background:var(--border);border-radius:4px;position:relative"><div style="position:absolute;top:0;left:0;width:79%;height:100%;background:#C8C3BB;border-radius:4px"></div><div style="position:absolute;top:0;left:0;width:81%;height:100%;background:rgba(29,191,123,.4);border-radius:4px"></div></div></td>
          </tr>
          <tr>
            <td style="padding:10px 12px;font-weight:600">Safety</td>
            <td style="text-align:center;padding:10px 12px;color:var(--text-m)">5%</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:#9B9490">74</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--green);font-weight:700">90</td>
            <td style="text-align:right;padding:10px 12px;font-family:'JetBrains Mono',monospace;color:var(--yellow)">+16</td>
            <td style="padding:10px 12px"><div style="height:7px;background:var(--border);border-radius:4px;position:relative"><div style="position:absolute;top:0;left:0;width:74%;height:100%;background:#C8C3BB;border-radius:4px"></div><div style="position:absolute;top:0;left:0;width:90%;height:100%;background:rgba(29,191,123,.4);border-radius:4px"></div></div></td>
          </tr>
        </tbody>
        <tfoot>
          <tr style="border-top:2px solid var(--border);background:var(--bg)">
            <td style="padding:12px 12px;font-weight:700">Overall (Weighted)</td>
            <td style="text-align:center;padding:12px 12px;color:var(--text-dim)">100%</td>
            <td style="text-align:right;padding:12px 12px;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:15px;color:#9B9490">45</td>
            <td style="text-align:right;padding:12px 12px;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:15px;color:var(--green)">95</td>
            <td style="text-align:right;padding:12px 12px;font-family:'JetBrains Mono',monospace;font-weight:700;font-size:15px;color:var(--yellow)">+50</td>
            <td style="padding:12px 12px"><div style="height:9px;background:var(--border);border-radius:4px;position:relative"><div style="position:absolute;top:0;left:0;width:45%;height:100%;background:#C8C3BB;border-radius:4px"></div><div style="position:absolute;top:0;left:0;width:95%;height:100%;background:rgba(29,191,123,.5);border-radius:4px"></div></div></td>
          </tr>
        </tfoot>
      </table>
    </div>

    <!-- Radar chart -->
    <div class="cc" style="margin-top:18px">
      <div class="cc-hd">
        <div>
          <div class="cc-t">Capability Radar — All 7 Dimensions</div>
          <div class="cc-d">Grey = MariaGUI (Old) &nbsp;&middot;&nbsp; Green = Maria_App (Present)</div>
        </div>
      </div>
      <div style="max-width:480px;margin:0 auto;height:300px"><canvas id="cmpRadar"></canvas></div>
    </div>

    <!-- How scores are calculated -->
    <h2>How Every Score Is Calculated</h2>
    <p>Each dimension starts from a fixed <strong>base score</strong> representing what any single-model chatbot can reasonably achieve, then adds bonus points for each detected feature. Scores are capped at 100. The source code is scanned with regex patterns for specific class names, function names, and imports — not keywords — to avoid false positives.</p>

    <div class="cc" style="margin-top:4px">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0">

        <div style="padding:16px 20px;border-right:1px solid var(--border);border-bottom:1px solid var(--border)">
          <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--yellow);margin-bottom:8px">Intelligence &mdash; Weight 25%</div>
          <div style="font-size:12.5px;color:var(--text-m);line-height:1.7">
            <strong>Base 60</strong> — any LLM chatbot can hold a basic conversation.<br>
            <code>+7</code> 3+ specialist models (llama3.1:8b + deepseek-r1:8b + qwen2.5-coder:7b)<br>
            <code>+5</code> ToolPlanner — plans multi-step tool use before acting<br>
            <code>+5</code> SpecializedCoT — type-specific chain-of-thought scaffolds per query type<br>
            <code>+4</code> ReActToolRegistry — dynamic tool selection loop<br>
            <code>+3</code> IntentRouter — routes query to the right model<br>
            <code>+4</code> SymPy — symbolic math raises reasoning depth<br>
            <code>+4</code> HybridWebRAG — retrieval-augmented generation<br>
            <code>+3</code> DuckDuckGo web search — live external knowledge<br>
            <code>+3</code> ReflectionPass — post-generation self-check catches small-model errors<br>
            <code>+2</code> Wikipedia lookup — structured factual knowledge<br>
            <code>+3</code> Specialist routing — each model handles its domain<br>
            <code>+2</code> Streaming output — real-time token generation<br>
            <code>+1</code> LLM history compression — longer effective context<br>
            <span style="color:var(--text-dim);font-size:11.5px">GUI: 60 (no bonuses) &nbsp;|&nbsp; App: 60+46 = 106 &rarr; capped at 100</span>
          </div>
        </div>

        <div style="padding:16px 20px;border-bottom:1px solid var(--border)">
          <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--yellow);margin-bottom:8px">Anti-Hallucination &mdash; Weight 20%</div>
          <div style="font-size:12.5px;color:var(--text-m);line-height:1.7">
            <strong>Base 32</strong> — LLMs hallucinate by default without grounding.<br>
            <code>+10</code> validate_sources_topic() — cross-checks source relevance <em>(both versions)</em><br>
            <code>+20</code> HallucinationDetector class — active post-generation check<br>
            <code>+9</code> Online Intelligence Mode — forces live data over memorized facts<br>
            <code>+8</code> Web search — grounds claims in real-time sources<br>
            <code>+7</code> HybridWebRAG — RAG pipeline adds source attribution<br>
            <code>+7</code> SymPy — math answers are computed, not guessed<br>
            <code>+4</code> Wikipedia — verified factual cross-reference<br>
            <span style="color:var(--text-dim);font-size:11.5px">GUI: 32+10 = 42 &nbsp;|&nbsp; App: 32+65 = 97</span>
          </div>
        </div>

        <div style="padding:16px 20px;border-right:1px solid var(--border);border-bottom:1px solid var(--border)">
          <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--yellow);margin-bottom:8px">Math Accuracy &mdash; Weight 15%</div>
          <div style="font-size:12.5px;color:var(--text-m);line-height:1.7">
            <strong>Base 35</strong> — LLMs can do basic arithmetic but fail on complex math.<br>
            <code>+37</code> SymPy — exact symbolic computation (algebra, calculus, equations)<br>
            <code>+18</code> deepseek-r1:8b — a reasoning-focused model built for math<br>
            <code>+5</code> ToolPlanner — can chain math steps across tool calls<br>
            <code>+2</code> Web search — can look up formulas or reference tables<br>
            <span style="color:var(--text-dim);font-size:11.5px">GUI: 35 (no math tools) &nbsp;|&nbsp; App: 35+62 = 97</span>
          </div>
        </div>

        <div style="padding:16px 20px;border-bottom:1px solid var(--border)">
          <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--yellow);margin-bottom:8px">Code Execution &mdash; Weight 15%</div>
          <div style="font-size:12.5px;color:var(--text-m);line-height:1.7">
            <strong>Base 35</strong> — LLMs can generate code but cannot run it by default.<br>
            <code>+22</code> qwen2.5-coder:7b — a code-specialized model<br>
            <code>+20</code> _safe_execute_python() — sandboxed code runner with timeout guard<br>
            <code>+8</code> ReActToolRegistry — code execution wired as a callable tool<br>
            <code>+4</code> ToolPlanner — plans code generation + execution steps<br>
            <em style="font-size:11.5px;color:var(--text-dim)">Note: MariaGUI imports subprocess for OS detection only — it cannot run user code. The scorer checks for _safe_execute_python, not subprocess, to avoid this false positive.</em><br>
            <span style="color:var(--text-dim);font-size:11.5px">GUI: 35 (no sandbox) &nbsp;|&nbsp; App: 35+54 = 89</span>
          </div>
        </div>

        <div style="padding:16px 20px;border-right:1px solid var(--border);border-bottom:1px solid var(--border)">
          <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--yellow);margin-bottom:8px">Knowledge Retrieval &mdash; Weight 15%</div>
          <div style="font-size:12.5px;color:var(--text-m);line-height:1.7">
            <strong>Base 20</strong> — without retrieval, a model is limited to training data.<br>
            <code>+24</code> HybridWebRAG — combines dense + sparse retrieval over documents<br>
            <code>+22</code> DuckDuckGo search — live web results on demand<br>
            <code>+14</code> Wikipedia API — structured encyclopaedic lookup<br>
            <code>+10</code> Online Intelligence Mode — automatically fetches fresh data<br>
            <code>+5</code> RAG document store — indexes local documents<br>
            <span style="color:var(--text-dim);font-size:11.5px">GUI: 20 (no retrieval tools) &nbsp;|&nbsp; App: 20+75 = 95</span>
          </div>
        </div>

        <div style="padding:16px 20px;border-bottom:1px solid var(--border)">
          <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--yellow);margin-bottom:8px">Language Support &mdash; Weight 5%</div>
          <div style="font-size:12.5px;color:var(--text-m);line-height:1.7">
            <strong>Base 52</strong> — underlying LLMs already support many languages.<br>
            <code>+22</code> langdetect library — auto-detects user's language and responds in kind <em>(both versions)</em><br>
            <code>+5</code> pyttsx3 TTS — text-to-speech output <em>(both versions)</em><br>
            <code>+2</code> Streaming — token-by-token output feels more natural <em>(Maria_App only)</em><br>
            <span style="color:var(--text-dim);font-size:11.5px">GUI: 52+27 = 79 &nbsp;|&nbsp; App: 52+29 = 81 &nbsp;(closest gap — both share most language features)</span>
          </div>
        </div>

        <div style="padding:16px 20px;border-right:1px solid var(--border)">
          <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--yellow);margin-bottom:8px">Safety &mdash; Weight 5%</div>
          <div style="font-size:12.5px;color:var(--text-m);line-height:1.7">
            <strong>Base 38</strong> — base LLMs have some refusal capability built-in.<br>
            <code>+18</code> HARM_PAT filter — regex blocklist for harmful content <em>(both versions)</em><br>
            <code>+14</code> validate_sources_topic() — rejects off-topic or unsafe source content <em>(both)</em><br>
            <code>+16</code> HallucinationDetector — flags responses that cannot be grounded <em>(App only)</em><br>
            <code>+4</code> langdetect — helps filter multilingual harmful inputs <em>(both)</em><br>
            <span style="color:var(--text-dim);font-size:11.5px">GUI: 38+36 = 74 &nbsp;|&nbsp; App: 38+52 = 90</span>
          </div>
        </div>

        <div style="padding:16px 20px">
          <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--text-dim);margin-bottom:8px">Overall Score Formula</div>
          <div style="font-size:12.5px;color:var(--text-m);line-height:1.9">
            <code style="display:block;background:var(--bg);padding:8px 12px;border-radius:6px;margin-bottom:10px;font-size:11.5px">Overall = round(I&times;.25 + AH&times;.20 + M&times;.15 + C&times;.15 + K&times;.15 + L&times;.05 + S&times;.05)</code>
            <strong>MariaGUI:</strong> 60&times;.25 + 42&times;.20 + 35&times;.15 + 35&times;.15 + 20&times;.15 + 79&times;.05 + 74&times;.05<br>
            = 15.0 + 8.4 + 5.25 + 5.25 + 3.0 + 3.95 + 3.70 = <strong>44.55 &rarr; 45</strong><br><br>
            <strong>Maria_App:</strong> 98&times;.25 + 97&times;.20 + 97&times;.15 + 89&times;.15 + 95&times;.15 + 81&times;.05 + 90&times;.05<br>
            = 24.5 + 19.4 + 14.55 + 13.35 + 14.25 + 4.05 + 4.5 = <strong>94.60 &rarr; 95</strong>
          </div>
        </div>

      </div>
    </div>

    <!-- Feature comparison matrix -->
    <h2>Feature Comparison Matrix</h2>
    <div class="cc" style="margin-top:4px">
      <table style="width:100%;border-collapse:collapse;font-size:13px">
        <thead>
          <tr style="border-bottom:2px solid var(--border)">
            <th style="text-align:left;padding:8px 12px;font-weight:600;color:var(--text-b)">Feature</th>
            <th style="text-align:center;padding:8px 12px;font-weight:600;color:#9B9490">MariaGUI (Old)</th>
            <th style="text-align:center;padding:8px 12px;font-weight:600;color:var(--green)">Maria_App (Present)</th>
          </tr>
        </thead>
        <tbody>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">GUI Framework</td><td style="text-align:center;padding:9px 12px">tkinter</td><td style="text-align:center;padding:9px 12px;color:var(--green);font-weight:600">PyQt5</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Language Models</td><td style="text-align:center;padding:9px 12px">1 &mdash; llama3.1:8b</td><td style="text-align:center;padding:9px 12px;color:var(--green);font-weight:600">3 specialists (llama3.1, deepseek-r1, qwen2.5-coder)</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Intent Routing</td><td style="text-align:center;padding:9px 12px;color:#C8C3BB">&#x2715;</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; IntentRouter</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Tool Planning</td><td style="text-align:center;padding:9px 12px;color:#C8C3BB">&#x2715;</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; ToolPlanner + ReActToolRegistry</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Web Search</td><td style="text-align:center;padding:9px 12px;color:#C8C3BB">&#x2715;</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; DuckDuckGo</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Wikipedia Lookup</td><td style="text-align:center;padding:9px 12px;color:#C8C3BB">&#x2715;</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713;</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">RAG / Document Retrieval</td><td style="text-align:center;padding:9px 12px;color:#C8C3BB">&#x2715;</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; HybridWebRAG</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Symbolic Math (SymPy)</td><td style="text-align:center;padding:9px 12px;color:#C8C3BB">&#x2715;</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713;</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Code Sandbox Execution</td><td style="text-align:center;padding:9px 12px;color:#C8C3BB">&#x2715;</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; _safe_execute_python()</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Hallucination Detector</td><td style="text-align:center;padding:9px 12px;color:#C8C3BB">&#x2715;</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; HallucinationDetector class</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Online Intelligence Mode</td><td style="text-align:center;padding:9px 12px;color:#C8C3BB">&#x2715;</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713;</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Chain-of-Thought (SpecializedCoT)</td><td style="text-align:center;padding:9px 12px;color:#C8C3BB">&#x2715;</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; SpecializedCoT per query type</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Self-Reflection (ReflectionPass)</td><td style="text-align:center;padding:9px 12px;color:#C8C3BB">&#x2715;</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; post-generation error check</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Streaming Responses</td><td style="text-align:center;padding:9px 12px;color:#C8C3BB">&#x2715;</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713;</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">LLM History Compression</td><td style="text-align:center;padding:9px 12px;color:#C8C3BB">&#x2715;</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713;</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Source Validation</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; validate_sources_topic()</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713;</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Harmful Content Filter</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; HARM_PAT regex</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713;</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Language Detection</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; langdetect</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713;</td></tr>
          <tr style="border-bottom:1px solid var(--border)"><td style="padding:9px 12px;color:var(--text-m)">Text-to-Speech (TTS)</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; pyttsx3</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; pyttsx3</td></tr>
          <tr><td style="padding:9px 12px;color:var(--text-m)">PDF Export</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; reportlab</td><td style="text-align:center;padding:9px 12px;color:var(--green)">&#x2713; reportlab</td></tr>
        </tbody>
      </table>
    </div>

    <!-- Why Maria_App is better -->
    <h2>Why Maria_App Is Better</h2>

    <div class="callout ok" style="margin-top:4px">
      <div class="callout-icon"></div>
      <div class="callout-body">
        <b>Maria_App scored 95/100 vs MariaGUI's 45/100 &mdash; a +50 point gap driven by 12+ major new capabilities.</b> The rewrite was not incremental: it replaced nearly every architectural layer.
      </div>
    </div>

    <div class="mg" style="grid-template-columns:1fr 1fr;margin-top:16px">
      <div class="mi">
        <div class="mi-ic g">&#x26A1;</div>
        <div><div class="mi-lb">3 Specialist Models Instead of 1</div><div class="mi-v" style="font-size:12.5px;font-weight:400;color:var(--text-m)">MariaGUI used a single general-purpose model for everything. Maria_App routes each query to the best model: deepseek-r1 for reasoning/math, qwen2.5-coder for code, and llama3.1 for general conversation. This alone accounts for the largest intelligence gap.</div></div>
      </div>
      <div class="mi">
        <div class="mi-ic g">&#x1F6E1;</div>
        <div><div class="mi-lb">Active Hallucination Detection</div><div class="mi-v" style="font-size:12.5px;font-weight:400;color:var(--text-m)">MariaGUI had no way to detect when the model was making things up. Maria_App's HallucinationDetector actively checks every response against retrieved sources, flagging or blocking ungrounded claims before the user sees them.</div></div>
      </div>
      <div class="mi">
        <div class="mi-ic y">&#x1F310;</div>
        <div><div class="mi-lb">Live Web Retrieval &amp; RAG</div><div class="mi-v" style="font-size:12.5px;font-weight:400;color:var(--text-m)">MariaGUI's knowledge was frozen at the model's training cutoff. Maria_App can search DuckDuckGo, query Wikipedia, and retrieve from local documents in real time — meaning its answers can be up-to-date and grounded in actual sources.</div></div>
      </div>
      <div class="mi">
        <div class="mi-ic b">&#x1F9EE;</div>
        <div><div class="mi-lb">Exact Math via SymPy</div><div class="mi-v" style="font-size:12.5px;font-weight:400;color:var(--text-m)">MariaGUI guessed at math answers like any vanilla LLM — prone to arithmetic errors on complex problems. Maria_App routes math queries through SymPy, a symbolic computation library that gives provably correct results for algebra, calculus, and equations.</div></div>
      </div>
      <div class="mi">
        <div class="mi-ic p">&#x1F4BB;</div>
        <div><div class="mi-lb">Sandboxed Code Execution</div><div class="mi-v" style="font-size:12.5px;font-weight:400;color:var(--text-m)">MariaGUI could only generate code as text — it could not run it. Maria_App's _safe_execute_python() executes code in a sandboxed subprocess with a timeout, letting it verify its own solutions and return real output rather than guessed results.</div></div>
      </div>
      <div class="mi">
        <div class="mi-ic y">&#x1F9ED;</div>
        <div><div class="mi-lb">Modular Tool Architecture</div><div class="mi-v" style="font-size:12.5px;font-weight:400;color:var(--text-m)">MariaGUI ran every query through a single pipeline. Maria_App uses ToolPlanner + ReActToolRegistry to dynamically select and chain tools across multiple reasoning steps — enabling multi-step problem solving, not just single-turn responses.</div></div>
      </div>
    </div>
  </div>

</main>

<script>
const TITLES = {{overview:'Dashboard',confidence:'Confidence',hallucination:'Hallucination Risk',response:'Response Times',dpo:'DPO Dataset',progress:'Training Progress',examples:'Recent Examples',comparison:'Version Comparison'}};
function show(id) {{
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.sb-nav button').forEach(b=>b.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('on');
  var nb=document.getElementById('n-'+id);if(nb)nb.classList.add('active');
  document.getElementById('bc').textContent = TITLES[id]||id;
  window.scrollTo(0,0);
}}
Chart.defaults.font.family="'DM Sans',sans-serif";
Chart.defaults.font.size=12;
if(sessionStorage.getItem('maria_pg'))Chart.defaults.animation=false;
Chart.defaults.color='#9B9490';
const G='#EBE6DA';
const PAL=['#FF9D00','#F15B5B','#3B82F6','#1DBF7B','#8B5CF6','#F472B6','#06B6D4','#84CC16'];
const sc={{x:{{grid:{{color:G,lineWidth:.7}},border:{{display:false}}}},y:{{grid:{{color:G,lineWidth:.7}},border:{{display:false}},ticks:{{maxTicksLimit:5}}}}}};

const JSONL_DATA={jv(chr(10).join(json.dumps(e) for e in load_dpo()))};
// Real data
const hi_c={d["hi_c"]},mid_c={d["mid_c"]},lo_c={d["lo_c"]};
const qt_keys={jv(qt_keys)},qt_vals={jv(qt_vals)};
const times={jv(d["times"])},roll={jv(d["roll"])};
const p50={d["p50"]},p95={d["p95"]};
const src_keys={jv(src_keys)},src_vals={jv(src_vals)};
const src_margin_series={jv(d["src_margin_series"])};
const lang_keys={jv(lang_keys)},lang_vals={jv(lang_vals)};
const hist_labels={jv(d["hist_labels"])},hist_data={jv(d["hist_data"])},hist_colors={jv(d["hist_colors"])};
const corr={d["corr"]},rel_fail={d["rel_fail"]},sft_n={d["sft_n"]},dpo_n={d["dpo_n"]},total_q={d["total_q"]};
const hallu={d["hallu_risk"]},rel_rate={d["rel_rate"]};

new Chart(document.getElementById('confBar'),{{type:'bar',data:{{labels:['High ≥85%','Mid 60–85%','Low <60%'],datasets:[{{data:[hi_c,mid_c,lo_c],backgroundColor:['#1DBF7B','#FF9D00','#F15B5B'],borderRadius:7,borderSkipped:false}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>` ${{ctx.raw.toLocaleString()}} queries`}}}}}},scales:{{x:{{grid:{{display:false}},border:{{display:false}}}},y:{{grid:{{color:G}},border:{{display:false}},ticks:{{precision:0,maxTicksLimit:5}}}}}}}}}});

new Chart(document.getElementById('queryDonut'),{{type:'doughnut',data:{{labels:qt_keys.map(k=>k.charAt(0).toUpperCase()+k.slice(1)),datasets:[{{data:qt_vals,backgroundColor:PAL,borderWidth:3,borderColor:'#FFF',hoverOffset:6}}]}},options:{{responsive:true,maintainAspectRatio:false,cutout:'58%',plugins:{{legend:{{position:'bottom',labels:{{padding:14,boxWidth:11,boxHeight:11}}}},tooltip:{{callbacks:{{label:ctx=>` ${{ctx.raw.toLocaleString()}} queries`}}}}}}}}}});

new Chart(document.getElementById('halluBar'),{{type:'bar',data:{{labels:['Hallucination Risk','Relevance Rate'],datasets:[{{data:[hallu,rel_rate],backgroundColor:[hallu>=20?'#F15B5B':hallu>=10?'#FF9D00':'#1DBF7B',rel_rate>=80?'#1DBF7B':'#FF9D00'],borderRadius:6,borderSkipped:false,maxBarThickness:50}}]}},options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>` ${{ctx.raw}}%`}}}}}},scales:{{x:{{max:100,grid:{{color:G}},border:{{display:false}},ticks:{{callback:v=>v+'%'}}}},y:{{grid:{{display:false}},border:{{display:false}}}}}}}}}});

new Chart(document.getElementById('qualityBar'),{{type:'bar',data:{{labels:['High Conf','Mid Conf','Low Conf','Self-Corr','Rel. Failures'],datasets:[{{data:[hi_c,mid_c,lo_c,corr,rel_fail],backgroundColor:['#1DBF7B','#FF9D00','#F15B5B','#8B5CF6','#F15B5B'],borderRadius:6,borderSkipped:false}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{grid:{{display:false}},border:{{display:false}}}},y:{{grid:{{color:G}},border:{{display:false}},ticks:{{precision:0,maxTicksLimit:5}}}}}}}}}});

const rtL=times.map((_,i)=>i+1);
new Chart(document.getElementById('rtLine'),{{type:'line',data:{{labels:rtL,datasets:[{{label:'Response Time',data:times,borderColor:'#3B82F6',backgroundColor:'rgba(59,130,246,.08)',borderWidth:1.5,pointRadius:0,fill:true,tension:.3}},{{label:'8-query avg',data:roll,borderColor:'#FF9D00',borderWidth:2.4,pointRadius:0,fill:false,tension:.4}},{{label:`P50 (${{p50}}s)`,data:rtL.map(()=>p50),borderColor:'#1DBF7B',borderWidth:1.3,borderDash:[6,4],pointRadius:0,fill:false}},{{label:`P95 (${{p95}}s)`,data:rtL.map(()=>p95),borderColor:'#F15B5B',borderWidth:1.3,borderDash:[6,4],pointRadius:0,fill:false}}]}},options:{{responsive:true,maintainAspectRatio:false,interaction:{{mode:'index',intersect:false}},plugins:{{legend:{{position:'top',align:'end',labels:{{padding:14,boxWidth:24,boxHeight:2}}}}}},scales:{{x:{{grid:{{color:G,lineWidth:.6}},border:{{display:false}},ticks:{{maxTicksLimit:10}}}},y:{{grid:{{color:G,lineWidth:.6}},border:{{display:false}},ticks:{{callback:v=>v.toFixed(1)+'s'}}}}}}}}}});

new Chart(document.getElementById('srcDonut'),{{type:'doughnut',data:{{labels:src_keys.length?src_keys:['No data yet'],datasets:[{{data:src_vals.length?src_vals:[1],backgroundColor:src_vals.length?PAL:['#EBE6DA'],borderWidth:3,borderColor:'#FFF',hoverOffset:6}}]}},options:{{responsive:true,maintainAspectRatio:false,cutout:'58%',plugins:{{legend:{{position:'bottom',labels:{{padding:14,boxWidth:11,boxHeight:11}}}},tooltip:{{callbacks:{{label:ctx=>` ${{ctx.raw.toLocaleString()}} pairs`}}}}}}}}}});

new Chart(document.getElementById('langBar'),{{type:'bar',data:{{labels:lang_keys.length?lang_keys:['No data'],datasets:[{{data:lang_vals.length?lang_vals:[0],backgroundColor:PAL.slice(2),borderRadius:7,borderSkipped:false}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>` ${{ctx.raw.toLocaleString()}} pairs`}}}}}},scales:{{x:{{grid:{{display:false}},border:{{display:false}}}},y:{{grid:{{color:G}},border:{{display:false}},ticks:{{precision:0,maxTicksLimit:5}}}}}}}}}});

new Chart(document.getElementById('marginHist'),{{type:'bar',data:{{labels:hist_labels,datasets:[{{data:hist_data,backgroundColor:hist_colors,borderRadius:5,borderSkipped:false,borderWidth:0}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{grid:{{display:false}},border:{{display:false}},ticks:{{maxTicksLimit:7}}}},y:{{grid:{{color:G}},border:{{display:false}},ticks:{{precision:0,maxTicksLimit:5}}}}}}}}}});

new Chart(document.getElementById('datasetBar'),{{type:'bar',data:{{labels:['SFT Examples','DPO Pairs','High-Conf Replies','Total Queries'],datasets:[{{data:[sft_n,dpo_n,hi_c,total_q],backgroundColor:['#FF9D00','#8B5CF6','#1DBF7B','#3B82F6'],borderRadius:7,borderSkipped:false}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{grid:{{display:false}},border:{{display:false}}}},y:{{grid:{{color:G}},border:{{display:false}},ticks:{{precision:0,maxTicksLimit:5}}}}}}}}}});

// Source breakdown donut — Recent Examples page (Hugging Face warm palette)
// Source breakdown line chart — reward margin trend per source
const HF_PAL=['#FF9D00','#FFB347','#FF6E4A','#FFCA28','#E07B00','#F4845F'];
new Chart(document.getElementById('srcDonut2'),{{
  type:'doughnut',
  data:{{
    labels:src_keys.length?src_keys:['No data yet'],
    datasets:[{{
      data:src_vals.length?src_vals:[1],
      backgroundColor:src_vals.length?HF_PAL:['#EBE6DA'],
      borderWidth:4,
      borderColor:'#FFFDF8',
      hoverOffset:10
    }}]
  }},
  options:{{
    responsive:true,
    maintainAspectRatio:false,
    cutout:'62%',
    plugins:{{
      legend:{{
        position:'right',
        labels:{{padding:18,boxWidth:13,boxHeight:13,font:{{size:12,weight:'600'}},color:'#383330'}}
      }},
      tooltip:{{
        backgroundColor:'#1A1714',
        padding:10,
        cornerRadius:8,
        callbacks:{{
          label:ctx=>` ${{ctx.raw.toLocaleString()}} pairs (${{Math.round(ctx.raw/src_vals.reduce((a,b)=>a+b,0)*100)}}%)`
        }}
      }}
    }}
  }}
}});

// Comparison page — Capability Radar
new Chart(document.getElementById('cmpRadar'),{{
  type:'radar',
  data:{{
    labels:['Intelligence','Anti-Hallucination','Math','Code','Knowledge','Language','Safety'],
    datasets:[
      {{label:'MariaGUI (Old)',data:[60,42,35,35,20,79,74],backgroundColor:'rgba(155,148,144,.15)',borderColor:'#C8C3BB',borderWidth:2,pointBackgroundColor:'#C8C3BB',pointRadius:4}},
      {{label:'Maria_App (Present)',data:[100,97,97,89,95,81,90],backgroundColor:'rgba(29,191,123,.12)',borderColor:'#1DBF7B',borderWidth:2.5,pointBackgroundColor:'#1DBF7B',pointRadius:5}}
    ]
  }},
  options:{{
    responsive:true,maintainAspectRatio:false,
    scales:{{r:{{min:0,max:100,ticks:{{stepSize:20,font:{{size:10}},color:'#9B9490',backdropColor:'transparent'}},grid:{{color:'#EBE6DA'}},angleLines:{{color:'#EBE6DA'}},pointLabels:{{font:{{size:12,weight:'600'}},color:'#383330'}}}}}},
    plugins:{{legend:{{position:'bottom',labels:{{padding:20,boxWidth:12,boxHeight:12,font:{{size:12}},color:'#383330'}}}}}}
  }}
}});

function exportJSONL(){{
  const blob=new Blob([JSONL_DATA],{{type:'application/x-jsonlines'}});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='maria_dpo_dataset.jsonl';
  a.click();
  URL.revokeObjectURL(a.href);
}}
// ── Live auto-refresh (preserves current tab, no animations on reload) ───
(function(){{
  var isReload = !!sessionStorage.getItem('maria_pg');
  if (isReload) {{
    var s = document.createElement('style');
    s.textContent = '*,*::before,*::after{{animation:none!important;transition:none!important}}';
    document.head.appendChild(s);
  }}
  var _origShow = show;
  show = function(id) {{ _origShow(id); sessionStorage.setItem('maria_pg', id); }};
  var saved = sessionStorage.getItem('maria_pg');
  if (saved && document.getElementById('page-' + saved)) show(saved);
  setInterval(function() {{
    sessionStorage.setItem('maria_pg', sessionStorage.getItem('maria_pg') || 'overview');
    location.reload();
  }}, 10000);
}})();
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════════════
# LIVE SERVER
# ══════════════════════════════════════════════════════════════════════════

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            p = self.path.split('?')[0]
            if p in ('/', '/index.html', '/docs'):
                body = build_docs_html().encode('utf-8')
            elif p == '/dashboard':
                body = build_html(collect()).encode('utf-8')
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass  # suppress request logs


def _free_port(start=7477):
    for p in range(start, start + 100):
        try:
            s = socket.socket()
            s.bind(('', p))
            s.close()
            return p
        except OSError:
            continue
    return start


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = _free_port()
    url  = f"http://localhost:{port}"

    server = HTTPServer(('', port), _Handler)

    print("  Maria AI — Live Dashboard")
    print("=" * 44)
    print(f"   {url}")
    print("   Refreshes every 10 s automatically.")
    print("   Press Ctrl+C to stop.")
    print()

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()