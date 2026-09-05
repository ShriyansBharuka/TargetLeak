"""HTML report rendering for targetleak.

Kept out of the core so `analyse` has no opinion about presentation.

The report is built as a **pathology assay sheet**, and that is not decoration:
a lab result is read against a reference range, which is exactly the shape of
this tool's output. AUC 0.9998 means nothing on its own; AUC 0.9998 against a
band where a healthy feature sits between 0.50 and 0.90 is a diagnosis you can
read at a glance. The reference bar is therefore the primary device and every
other choice stays quiet around it.

No JavaScript and no CDN scripts. Typefaces load from Google Fonts behind a
full fallback stack, so a report opened offline degrades to a local mono/serif
rather than breaking. Charts are hand-built SVG - a plotting dependency would
outweigh the entire package.
"""
import datetime
import html
import json

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# Printed-ink palette, not screen primaries: oxidised iron, brass, and a deep
# verdigris. Reads as a laboratory record and stays legible in grayscale print.
INK = {
    "critical": "#8f2c1e",
    "warning": "#7d5f10",
    "info": "#2b5750",
    "rule": "#b9bdb2",
    "muted": "#5f6459",
}

FONTS = ("https://fonts.googleapis.com/css2"
         "?family=IBM+Plex+Mono:wght@400;500;600"
         "&family=IBM+Plex+Sans+Condensed:wght@500;600;700"
         "&family=IBM+Plex+Serif:ital,wght@0,400;0,500;1,400"
         "&display=swap")

CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --paper:#eaece5; --ink:#1a1f19; --dim:#5f6459; --rule:#b9bdb2;
  --tint:#e0e3d9; --crit:#8f2c1e; --warn:#7d5f10; --ok:#2b5750;
  --band:#cfd3c6; --band-hot:#d9b7ae;
  --mono:"IBM Plex Mono","SFMono-Regular","Cascadia Mono",Consolas,monospace;
  --cond:"IBM Plex Sans Condensed","Roboto Condensed","Arial Narrow",sans-serif;
  --serif:"IBM Plex Serif","Iowan Old Style",Georgia,serif;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#151814; --ink:#dfe2d8; --dim:#8d9384; --rule:#3a3f36;
    --tint:#1d211c; --crit:#e08573; --warn:#d8b257; --ok:#7fbdb0;
    --band:#2b302a; --band-hot:#4a2a24;
  }
}
:root[data-theme="dark"]{
  --paper:#151814; --ink:#dfe2d8; --dim:#8d9384; --rule:#3a3f36;
  --tint:#1d211c; --crit:#e08573; --warn:#d8b257; --ok:#7fbdb0;
  --band:#2b302a; --band-hot:#4a2a24;
}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);
  font-size:15px;line-height:1.6;
  /* Faint ruled-paper texture. Carries the ledger without an image. */
  background-image:repeating-linear-gradient(
    to bottom,transparent 0 27px,rgba(120,128,112,.055) 27px 28px);}
.sheet{max-width:960px;margin:0 auto;padding:0 28px 88px}

/* ---- masthead: a specimen header block, not a hero ---- */
.mast{padding:36px 0 0}
.mast-top{display:flex;justify-content:space-between;align-items:flex-end;
  gap:24px;border-bottom:2.5px solid var(--ink);padding-bottom:9px}
.wordmark{font-family:var(--cond);font-weight:700;font-size:27px;
  letter-spacing:.055em;text-transform:uppercase;line-height:1}
.wordmark i{font-style:normal;color:var(--crit)}
.assay{font-family:var(--cond);font-size:11.5px;letter-spacing:.2em;
  text-transform:uppercase;color:var(--dim);text-align:right;line-height:1.5}
.spec{display:grid;grid-template-columns:repeat(auto-fit,minmax(184px,1fr));
  gap:0;border-bottom:1px solid var(--rule)}
.spec div{padding:10px 16px 11px 0;border-right:1px solid var(--rule)}
.spec div:last-child{border-right:0}
.spec dt{font-family:var(--cond);font-size:10px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--dim);margin:0 0 2px}
.spec dd{margin:0;font-family:var(--mono);font-size:13px;word-break:break-all}

/* ---- verdict: a stamped band ---- */
.verdict{margin:26px 0 34px;padding:15px 20px;border-left:5px solid var(--ok);
  background:var(--tint)}
.verdict.critical{border-left-color:var(--crit)}
.verdict.warning{border-left-color:var(--warn)}
.verdict h2{margin:0 0 3px;font-family:var(--cond);font-size:12px;
  letter-spacing:.2em;text-transform:uppercase;color:var(--dim);font-weight:600}
.verdict p{margin:0;font-size:16.5px;line-height:1.45}
.verdict .n{font-family:var(--mono);font-weight:600}
.verdict.critical .n{color:var(--crit)}

/* ---- tally strip ---- */
.tally{display:flex;gap:0;border-top:1px solid var(--rule);
  border-bottom:1px solid var(--rule);margin:0 0 34px}
.tally div{padding:9px 22px 9px 0;margin-right:22px;
  border-right:1px solid var(--rule);font-family:var(--mono);font-size:13px}
.tally div:last-child{border-right:0;margin-right:0}
.tally b{font-weight:600;font-size:17px}
.tally .c b{color:var(--crit)}
.tally .w b{color:var(--warn)}
.tally span{font-family:var(--cond);letter-spacing:.13em;text-transform:uppercase;
  font-size:10px;color:var(--dim);margin-left:7px}

/* ---- the ledger of findings ---- */
.sec{font-family:var(--cond);font-size:11px;letter-spacing:.2em;
  text-transform:uppercase;color:var(--dim);border-bottom:1px solid var(--ink);
  padding-bottom:5px;margin:0 0 0}
.row{display:grid;grid-template-columns:88px 1fr;gap:0 22px;
  border-bottom:1px solid var(--rule);padding:20px 0}
.gut{font-family:var(--mono);font-size:11px;line-height:1.5;color:var(--dim);
  border-left:3px solid var(--rule);padding-left:11px}
.row[data-sev=critical] .gut{border-left-color:var(--crit)}
.row[data-sev=warning] .gut{border-left-color:var(--warn)}
.row[data-sev=info] .gut{border-left-color:var(--rule)}
.gut .sev{display:block;font-family:var(--cond);font-weight:700;font-size:10px;
  letter-spacing:.13em;text-transform:uppercase;margin-top:4px}
.row[data-sev=critical] .sev{color:var(--crit)}
.row[data-sev=warning] .sev{color:var(--warn)}
.hd{display:flex;flex-wrap:wrap;align-items:baseline;gap:0 12px;margin:-2px 0 0}
.col{font-family:var(--mono);font-weight:600;font-size:16px;word-break:break-all}
.col.none{color:var(--dim);font-style:italic;font-family:var(--serif);
  font-weight:400}
.kind{font-family:var(--cond);font-size:11px;letter-spacing:.15em;
  text-transform:uppercase;color:var(--dim)}
.detail{margin:7px 0 0;max-width:64ch}
.lab{font-family:var(--cond);font-size:10px;letter-spacing:.17em;
  text-transform:uppercase;color:var(--dim);margin:19px 0 7px}
.remedy{margin:6px 0 0;padding-left:15px;border-left:1px solid var(--rule);
  max-width:64ch;font-style:italic;color:var(--ink)}
figure{margin:0}
table{border-collapse:collapse;font-family:var(--mono);font-size:12.5px;
  margin:2px 0 0}
th,td{padding:4px 20px 4px 0;text-align:right;font-variant-numeric:tabular-nums;
  border-bottom:1px solid var(--rule);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font-family:var(--cond);font-size:9.5px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--dim);font-weight:600}
.scroll{overflow-x:auto}
pre{font-family:var(--mono);font-size:12.5px;line-height:1.65;margin:2px 0 0;
  padding:17px 19px;background:var(--tint);border-left:3px solid var(--ok);
  overflow-x:auto}
footer{margin-top:40px;font-size:12.5px;color:var(--dim);max-width:70ch}
footer a{color:var(--ok)}
@media (max-width:620px){
  .row{grid-template-columns:1fr;gap:12px}
  .gut{border-left:0;border-top:3px solid var(--rule);padding:8px 0 0}
  .row[data-sev=critical] .gut{border-top-color:var(--crit)}
  .row[data-sev=warning] .gut{border-top-color:var(--warn)}
}
@media print{
  body{background:#fff;background-image:none}
  .row{break-inside:avoid}
}
"""


def _e(x):
    return html.escape(str(x), quote=True)


def _range_bar(score, band, metric="AUC"):
    """The signature device: a measured value against its reference range.

    A lab sheet never prints a number alone - it prints the number beside the
    interval a healthy sample falls in. `band` is [floor, warn, critical, top],
    so the track shows a normal zone, a watch zone and a pathological zone,
    with the measurement dropped on it as a tick.
    """
    try:
        lo, warn, crit, hi = [float(b) for b in band]
        v = float(score)
    except Exception:
        return ""
    # Four separate horizontal bands, because everything here wants its own
    # baseline: zone captions, the track, the axis ticks, and the measured
    # value. Sharing a baseline between the ticks and the value label made
    # "AUC 1.0000" overprint the "1" tick whenever a score pinned the top.
    W, pad = 620, 10
    cap_y, track_y, track_h, tick_y, val_y = 11, 22, 16, 54, 73
    H = 82
    plot = W - pad * 2

    def x(val):
        return pad + (max(lo, min(hi, val)) - lo) / (hi - lo or 1) * plot

    vx = x(v)
    colour = "var(--crit)" if v >= crit else (
        "var(--warn)" if v >= warn else "var(--ok)")
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
           f'style="max-width:{W}px;display:block" '
           f'aria-label="{_e(metric)} {v:.4f} against a reference range of '
           f'{lo:g} to {hi:g}; a healthy feature sits below {warn:g}">']
    out.append(f'<text x="{pad}" y="{cap_y}" font-size="9.5" '
               f'fill="{INK["muted"]}" font-family="var(--cond)" '
               f'letter-spacing="1.6">EXPECTED</text>')
    out.append(f'<text x="{W - pad}" y="{cap_y}" font-size="9.5" '
               f'fill="{INK["muted"]}" font-family="var(--cond)" '
               f'letter-spacing="1.6" text-anchor="end">PATHOLOGICAL</text>')
    for a, b, fill, opacity in ((lo, warn, "var(--band)", "1"),
                                (warn, crit, "var(--band-hot)", "1"),
                                (crit, hi, "var(--crit)", ".26")):
        out.append(f'<rect x="{x(a):.1f}" y="{track_y}" '
                   f'width="{max(x(b) - x(a), 0.5):.1f}" height="{track_h}" '
                   f'fill="{fill}" opacity="{opacity}"/>')
    # Ticks name every zone boundary, so no label sits on a value the scale
    # does not actually reach.
    for val in (lo, warn, crit, hi):
        out.append(f'<line x1="{x(val):.1f}" y1="{track_y}" x2="{x(val):.1f}" '
                   f'y2="{track_y + track_h + 4}" stroke="var(--rule)" '
                   f'stroke-width="1"/>')
        anchor = ("start" if val == lo else
                  "end" if val == hi else "middle")
        out.append(f'<text x="{x(val):.1f}" y="{tick_y}" font-size="10.5" '
                   f'fill="{INK["muted"]}" font-family="var(--mono)" '
                   f'text-anchor="{anchor}">{val:g}</text>')
    # The measurement, on its own baseline and clamped inside the viewBox.
    out.append(f'<line x1="{vx:.1f}" y1="{track_y - 5}" x2="{vx:.1f}" '
               f'y2="{track_y + track_h + 5}" stroke="{colour}" '
               f'stroke-width="2.5"/>')
    out.append(f'<circle cx="{vx:.1f}" cy="{track_y - 7}" r="3.2" '
               f'fill="{colour}"/>')
    label = f"{metric} {v:.4f}"
    half = len(label) * 3.6
    tx = min(max(vx, pad + half), W - pad - half)
    out.append(f'<text x="{tx:.1f}" y="{val_y}" font-size="12.5" '
               f'font-weight="600" fill="{colour}" font-family="var(--mono)" '
               f'text-anchor="middle">{_e(label)}</text>')
    out.append("</svg>")
    return "".join(out)


def _table(data):
    """Evidence as a values table. Numbers in a lab report are read, not eyeballed."""
    g = (data or {}).get("groups") or []
    keys = [k for k in ("mean", "median", "rate", "value", "n")
            if any(k in x for x in g)]
    if not g or not keys:
        return ""
    heads = {"n": "rows", "rate": "positive", "value": "target"}
    head = "".join(f"<th>{_e(heads.get(k, k))}</th>" for k in keys)
    rows = []
    for x in g:
        cells = []
        for k in keys:
            v = x.get(k)
            if v is None:
                cells.append("<td>&mdash;</td>")
            elif k == "n":
                cells.append(f"<td>{int(v):,}</td>")
            elif k == "rate":
                cells.append(f"<td>{v:.1%}</td>")
            else:
                cells.append(f"<td>{v:,.6g}</td>")
        rows.append(f"<tr><td>{_e(str(x.get('label', ''))[:44])}</td>"
                    + "".join(cells) + "</tr>")
    label = {"by_class": "target class", "cat_rates": "category",
             "missing": "state", "pure_cats": "category",
             "deciles": "target decile", "scores": "split"}
    first = label.get((data or {}).get("kind"), "group")
    return (f'<div class=scroll><table><thead><tr><th>{_e(first)}</th>{head}'
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def _score_strip(data):
    """Train / validation / chance as one labelled strip for `diagnose`."""
    keys = [("train", "train"), ("val", "validation"), ("baseline", "chance")]
    have = [(lab, float(data[k])) for k, lab in keys if data.get(k) is not None]
    if not have:
        return ""
    W, row = 620, 26
    H = row * len(have) + 6
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
           f'style="max-width:{W}px;display:block" aria-label="scores">']
    for i, (lab, v) in enumerate(have):
        y = i * row + 4
        w = max(v * (W - 190), 1.5)
        colour = ("var(--ok)" if lab == "chance" else
                  "var(--crit)" if lab == "train" else "var(--warn)")
        out.append(
            f'<text x="104" y="{y + 13}" font-size="11" text-anchor="end" '
            f'font-family="var(--cond)" letter-spacing="1.4" '
            f'fill="{INK["muted"]}">{lab.upper()}</text>'
            f'<rect x="114" y="{y + 2}" width="{w:.1f}" height="13" '
            f'fill="{colour}" opacity=".8"/>'
            f'<text x="{114 + w + 8:.1f}" y="{y + 13}" font-size="12" '
            f'font-family="var(--mono)" fill="currentColor">{v:.4f}</text>')
    out.append("</svg>")
    return "".join(out)


def _figure(data):
    """The right visual for an evidence dict, or '' when none applies."""
    if not data:
        return ""
    parts = []
    if data.get("score") is not None and data.get("band"):
        parts.append(_range_bar(data["score"], data["band"],
                                data.get("metric", "AUC")))
    if data.get("kind") == "scores":
        parts.append(_score_strip(data))
    table = _table(data)
    if table:
        parts.append(table)
    return "".join(parts)


def render(findings, target=None, source=None, fix_code_text=None,
           version=None):
    """Return a complete, self-contained HTML document as a string."""
    ordered = sorted(findings, key=lambda x: SEVERITY_ORDER[x.severity])
    crit = [f for f in findings if f.severity == "critical"]
    warn = [f for f in findings if f.severity == "warning"]
    info = [f for f in findings if f.severity == "info"]

    if crit:
        cls, head = "critical", "specimen rejected"
        body = (f'<span class=n>{len(crit)}</span> critical finding'
                f"{'s' if len(crit) != 1 else ''}. This model's test score is "
                "not evidence of anything until they are resolved.")
    elif warn:
        cls, head = "warning", "inconclusive"
        body = (f'No certain leak. <span class=n>{len(warn)}</span> observation'
                f"{'s' if len(warn) != 1 else ''} to confirm against how the "
                "data is actually produced.")
    else:
        cls, head = "clean", "no findings"
        body = ("Nothing detected by these checks. That is not a clean bill of "
                "health &mdash; it means these particular assays found nothing.")

    p = ["<!doctype html><html lang=en><head><meta charset=utf-8>",
         '<meta name=viewport content="width=device-width,initial-scale=1">',
         f"<title>targetleak &mdash; {_e(target or 'assay')}</title>",
         '<link rel=preconnect href="https://fonts.gstatic.com" crossorigin>',
         f'<link rel=stylesheet href="{FONTS}">',
         f"<style>{CSS}</style></head><body><div class=sheet>",
         "<header class=mast><div class=mast-top>",
         '<div class=wordmark>target<i>leak</i></div>',
         '<div class=assay>leakage assay<br>report</div>',
         "</div><dl class=spec>"]

    fields = [("specimen", source or "in-memory frame"),
              ("target column", target or "&mdash;"),
              ("assayed", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
              ("method", f"targetleak {version}" if version else "targetleak")]
    for k, v in fields:
        safe = v if v == "&mdash;" else _e(v)
        p.append(f"<div><dt>{_e(k)}</dt><dd>{safe}</dd></div>")
    p.append("</dl></header>")

    p += [f'<div class="verdict {cls}"><h2>{_e(head)}</h2><p>{body}</p></div>',
          "<div class=tally>",
          f"<div class=c><b>{len(crit)}</b><span>critical</span></div>",
          f"<div class=w><b>{len(warn)}</b><span>warning</span></div>",
          f"<div><b>{len(info)}</b><span>noted</span></div>",
          "</div>"]

    p.append('<div class=sec>findings</div>')
    for i, f in enumerate(ordered, 1):
        col = (f'<span class=col>{_e(f.column)}</span>' if f.column
               else '<span class="col none">whole dataset</span>')
        p += [f'<article class=row data-sev={f.severity}>',
              f'<div class=gut>TL-{i:02d}<span class=sev>{f.severity}</span></div>',
              "<div>",
              f'<div class=hd>{col}<span class=kind>{_e(f.kind)}</span></div>',
              f'<p class=detail>{_e(f.detail)}</p>']
        fig = _figure(f.data)
        if fig:
            p += ['<div class=lab>measurement</div>', f"<figure>{fig}</figure>"]
        elif f.evidence:
            p += ['<div class=lab>measurement</div>',
                  f'<p class=detail>{_e(f.evidence)}</p>']
        if f.fix:
            p += ['<div class=lab>indicated action</div>',
                  f'<p class=remedy>{_e(f.fix)}</p>']
        p.append("</div></article>")

    if fix_code_text:
        p += ['<div class=sec style="margin-top:38px">remediation</div>',
              f"<pre>{_e(fix_code_text)}</pre>"]

    p += ["<footer>These assays test the <em>data</em>. They cannot tell you "
          "whether a flagged column is legitimately available at prediction "
          "time &mdash; only you know that, which is why every finding above "
          "reports the measurement it was based on rather than a verdict "
          "alone. Generated by ",
          '<a href="https://pypi.org/project/targetleak/">targetleak</a>.',
          "</footer></div></body></html>"]
    return "".join(p)


def render_json(findings, target=None, fix_code_text=None):
    """The machine-readable twin of `render`, for CI."""
    return json.dumps({
        "target": target,
        "critical": sum(f.severity == "critical" for f in findings),
        "warning": sum(f.severity == "warning" for f in findings),
        "findings": [f.as_dict() for f in findings],
        "fix_code": fix_code_text,
    }, indent=2)
