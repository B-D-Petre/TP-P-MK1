"""One-page executive summary (self-contained HTML) generated from the golden layer. No LLM calls."""
import html
from datetime import date

import pandas as pd

from .config import ROOT
from .insights import STAKEHOLDERS, headline, interest_trends, load_gold, stakeholder_view

OUT = ROOT / "reports" / "executive_summary.html"
BADGE = {"rising": "▲ rising", "falling": "▼ falling", "stable": "– stable", "too few reviews": "· too few reviews"}


def badge_class(trend: str, sentiment: str | None) -> str:
    """Rising complaints are bad, rising praise is good; mixed-sentiment views stay neutral."""
    if trend not in ("rising", "falling") or sentiment is None:
        return "flat"
    good = (trend == "rising") == (sentiment == "positive")
    return "good" if good else "bad"


def sparkline(share: pd.Series, w: int = 150, h: int = 38) -> str:
    """Inline SVG sparkline of a monthly share series (y from 0 so noise isn't exaggerated, last point marked)."""
    v = share.fillna(0).tolist()
    if len(v) < 2:
        return ""
    lo, hi = 0.0, max(v)
    span = (hi - lo) or 1
    pts = [(4 + i * (w - 8) / (len(v) - 1), h - 4 - (x - lo) / span * (h - 8)) for i, x in enumerate(v)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    lx, ly = pts[-1]
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="monthly share, {share.index[0]} to {share.index[-1]}">'
            f'<polyline points="{path}" fill="none" stroke="var(--series)" stroke-width="2" stroke-linejoin="round"/>'
            f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3" fill="var(--series)"/></svg>')


def e(s) -> str:
    return html.escape(str(s))


def build_report(gold=None) -> str:
    t = load_gold(gold) if gold else load_gold()
    h = headline(t)
    ch = t["reviews"].groupby("source_group").agg(n=("review_id", "size"), stars=("stars", "mean"),
                                                   one=("stars", lambda x: (x == 1).sum()))
    org, inv = ch.loc["organic"] if "organic" in ch.index else None, ch.loc["invited"] if "invited" in ch.index else None
    _, trends = interest_trends(t)
    trends = trends.assign(neg=(trends["n_reviews"] * trends["pct_negative"]).round())
    driver = trends.sort_values("neg", ascending=False).iloc[0] if not trends.empty else None
    movers = trends[trends["trend"].isin(["rising", "falling"])].sort_values("change_pp", key=abs, ascending=False)

    findings = []
    if org is not None and inv is not None:
        findings.append(
            f"<b>The channel decides the score.</b> Organic reviews average <b>{org.stars:.1f}★</b> against "
            f"<b>{inv.stars:.1f}★</b> for invited ones. Organic is {org.n / h['n_reviews']:.0%} of reviews but "
            f"{org.one / max(ch['one'].sum(), 1):.0%} of all 1★ reviews.")
    if driver is not None:
        findings.append(
            f"<b>Biggest driver of negative reviews: “{e(driver.title)}”</b> — {int(driver.n_reviews):,} reviews, "
            f"{driver.pct_negative:.0%} of them 1–2★, trend: {BADGE[driver.trend]}.")
    if len(movers):
        m = movers.iloc[0]
        findings.append(f"<b>Biggest change: “{e(m.title)}”</b> went from {m.early_share:.1%} to "
                        f"{m.late_share:.1%} of monthly reviews ({m.change_pp:+.1f} pts).")
    else:
        findings.append("<b>No interest changed significantly</b> between the first and last three months: "
                        "the themes are stable over the period.")

    cards, used = [], set()
    for s in STAKEHOLDERS:
        v = stakeholder_view(t, s)
        label, cls = BADGE[v["trend"]], badge_class(v["trend"], s.sentiment)
        v["quotes"] = [q for q in v["quotes"] if q not in used][:1]  # each card gets its own quote
        used.update(v["quotes"])
        ents = ", ".join(e(x) for x in v["top_entities"]["entity"].head(3))
        extra = ""
        if v.get("requests"):
            extra = f'<p class="extra"><b>Top request:</b> {e(v["requests"][0][0])}</p>'
        elif v.get("causes"):
            extra = f'<p class="extra"><b>Stated cause:</b> {e(v["causes"][0])}</p>'
        quote = f'<blockquote>“{e(v["quotes"][0])}”</blockquote>' if v["quotes"] else ""
        cards.append(f"""
<article class="card">
  <h3>{e(s.name)}</h3>
  <p class="q">{e(s.question)}</p>
  <div class="row"><div><div class="big">{v['share_of_reviews']:.0%}</div><div class="lbl">of reviews · {v['n_reviews']:,}</div></div>
    <div class="trend"><span class="badge {cls}">{label}</span>{sparkline(v['monthly_share'])}</div></div>
  <p class="ents"><b>Mentioned most:</b> {ents}</p>{extra}{quote}
</article>""")

    findings_html = "".join(f"<li>{f}</li>" for f in findings)
    org_tile = (f'<div class="kpi"><div class="v">{org.stars:.1f}★ vs {inv.stars:.1f}★</div>'
                f'<div class="l">organic vs invited</div></div>') if org is not None and inv is not None else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trustpilot Review Insights</title>
<style>
:root {{ --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --rule:#e1e0d9;
  --series:#2a78d6; --accent:#2a78d6; --up:#b03030; --down:#006300; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0d0d0d; --surface:#1a1a19;
  --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --rule:#2c2c2a; --series:#3987e5; --accent:#3987e5;
  --up:#e66767; --down:#0ca30c; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:32px 16px 48px; }}
header h1 {{ font-size:28px; margin:0 0 4px; letter-spacing:-.01em; }}
header p {{ color:var(--ink2); margin:0; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:24px 0; }}
.kpi {{ background:var(--surface); border:1px solid var(--rule); border-radius:10px; padding:12px 14px; }}
.kpi .v {{ font-size:24px; font-weight:650; }} .kpi .l {{ color:var(--ink2); font-size:13px; }}
h2 {{ font-size:18px; margin:28px 0 10px; }}
.findings {{ background:var(--surface); border:1px solid var(--rule); border-left:4px solid var(--accent);
  border-radius:10px; padding:14px 18px 14px 34px; margin:0; }}
.findings li {{ margin:6px 0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); gap:14px; }}
.card {{ background:var(--surface); border:1px solid var(--rule); border-radius:10px; padding:14px 16px; }}
.card h3 {{ margin:0; font-size:16px; }} .q {{ color:var(--ink2); margin:2px 0 10px; font-size:13.5px; }}
.row {{ display:flex; justify-content:space-between; align-items:flex-end; gap:8px; }}
.big {{ font-size:28px; font-weight:650; line-height:1; }} .lbl {{ color:var(--ink2); font-size:12.5px; margin-top:4px; }}
.trend {{ text-align:right; }} .spark {{ display:block; margin-top:4px; }}
.badge {{ font-size:12px; font-weight:600; padding:2px 8px; border-radius:999px; border:1px solid var(--rule); }}
.badge.bad {{ color:var(--up); }} .badge.good {{ color:var(--down); }} .badge.flat {{ color:var(--ink2); }}
.ents,.extra {{ font-size:13.5px; margin:10px 0 0; }}
blockquote {{ margin:10px 0 0; padding:6px 10px; border-left:3px solid var(--rule); color:var(--ink2);
  font-style:italic; font-size:13.5px; }}
footer {{ margin-top:28px; color:var(--muted); font-size:12.5px; }}
@media print {{ body {{ background:#fff; }} main {{ padding:0; }} .card,.kpi,.findings {{ break-inside:avoid; }} }}
</style></head>
<body><main>
<header><h1>What {h['n_reviews']:,} reviews say about Trustpilot</h1>
<p>Reviews of Trustpilot on Trustpilot, {e(h['period'])} · topic-and-trend POC · generated {date.today():%d %b %Y}</p></header>
<section class="kpis">
  <div class="kpi"><div class="v">{h['n_reviews']:,}</div><div class="l">reviews analysed</div></div>
  <div class="kpi"><div class="v">{h['avg_stars']:.2f}★</div><div class="l">average rating</div></div>
  <div class="kpi"><div class="v">{h['pct_negative']:.0%}</div><div class="l">rated 1–2★</div></div>
  {org_tile}
  <div class="kpi"><div class="v">{h['n_macro_interests']}</div><div class="l">interests found</div></div>
</section>
<h2>Three things to know</h2>
<ol class="findings">{findings_html}</ol>
<h2>What it means for each team</h2>
<section class="grid">{''.join(cards)}</section>
<footer><p><b>Method.</b> An LLM extracts aspects, opinions and relations from each review (any language → English
concepts); these are merged into canonical entities, connected in a knowledge graph and grouped into interests with
Leiden community detection. Shares are % of each month's reviews; trends compare the first and last three months.
Explore everything in the Streamlit app (<code>uv run streamlit run src/poc_trustpilot/ui/app.py</code>).</p>
<p><b>Caveats.</b> {h['n_reviews']:,} reviews over {e(h['period'])}; LLM extraction can mislabel individual reviews;
trends on small interests are noisy (shown as “too few reviews”).</p></footer>
</main></body></html>"""


def write_report() -> None:
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(build_report(), encoding="utf-8")
    print(f"Executive summary -> {OUT}")
