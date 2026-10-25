"""Generate docs/index.html: the Assignment 5 project page (IFR buckets, discount-factor gradients, PV and risk).

Run from the repository root with the QuantLib interpreter:  python docs/build_page.py
"""
import html, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests")); os.chdir(ROOT)
from test_yield_curve_gradient import (FUNDING_INDEX, FUNDING_NODES, PAYMENT_DATE, SOFR_NODES, VALUE_DATE, build,  # noqa: E402
                                       bucket_lengths, components, curve_time, local_gradient, model_gradient)
from test_valuation_engine import engine_for, make_product  # noqa: E402
from fixedincomelib.date import Date  # noqa: E402

model = build()
sofr, funding = components(model)
T = curve_time(PAYMENT_DATE)
d_sofr, d_spread = sofr.discount_factor(Date(PAYMENT_DATE)), funding.discount_factor(Date(PAYMENT_DATE))
local = [local_gradient(sofr, PAYMENT_DATE), local_gradient(funding, PAYMENT_DATE)]
blocks = model_gradient(model, FUNDING_INDEX, PAYMENT_DATE)
engine, risk = engine_for(model, make_product())
C, PV = engine.cash_amount, engine.value
weights = [bucket_lengths(sofr.state_data[0], T), bucket_lengths(funding.state_data[0], T)]
names = ["SOFR-1B", FUNDING_INDEX]
node_dates = [list(SOFR_NODES), list(FUNDING_NODES)]

# ---- SVG: the two step functions over 0..2.2 years, with the buckets inside [0, T] shaded
W, L, R = 760, 60, 20
x_max = 2.2
def x(t): return L + (W - L - R) * t / x_max
def panel(top, height, times, values, y_lo, y_hi, fmt, colour, shade, title):
    def y(v): return top + height * (1 - (v - y_lo) / (y_hi - y_lo))
    parts = [f'<rect x="{L}" y="{top}" width="{W-L-R}" height="{height}" fill="none" stroke="#d8d8d8"/>']
    lower = 0.0
    for i, (t, v) in enumerate(zip(times, values)):
        upper = x_max if i == len(times) - 1 else t
        inside = min(upper, T)
        if inside > lower:
            parts.append(f'<rect x="{x(lower):.1f}" y="{y(v):.1f}" width="{x(inside)-x(lower):.1f}" height="{top+height-y(v):.1f}" fill="{shade}"/>')
            parts.append(f'<text x="{(x(lower)+x(inside))/2:.1f}" y="{top+height-6}" font-size="11" text-anchor="middle" fill="#333">w = {inside-lower:.6f}</text>')
        parts.append(f'<line x1="{x(lower):.1f}" y1="{y(v):.1f}" x2="{x(upper):.1f}" y2="{y(v):.1f}" stroke="{colour}" stroke-width="2"/>')
        if i < len(times) - 1:
            parts.append(f'<line x1="{x(t):.1f}" y1="{y(v):.1f}" x2="{x(t):.1f}" y2="{y(values[i+1]):.1f}" stroke="{colour}" stroke-width="1" stroke-dasharray="2 2"/>')
        lower = upper
    parts.append(f'<line x1="{x(T):.1f}" y1="{top}" x2="{x(T):.1f}" y2="{top+height}" stroke="#b00" stroke-width="1.5" stroke-dasharray="4 3"/>')
    ticks = [y_lo + k * (y_hi - y_lo) / 4 for k in range(5)]
    parts += [f'<text x="{L-6}" y="{y(v)+4:.1f}" font-size="11" text-anchor="end" fill="#555">{fmt(v)}</text>' for v in ticks]
    parts.append(f'<text x="{L+8}" y="{top+14}" font-size="12" fill="{colour}">{title}</text>')
    return "".join(parts)
svg = f"""<svg viewBox="0 0 {W} 330" width="100%" role="img" aria-label="SOFR forward rate and funding spread steps with the buckets inside [0, T] shaded">
{panel(14, 150, sofr.state_data[0], sofr.state_data[1], 0.02, 0.06, lambda v: f"{v*100:.0f}%", "#1f4e79", "#dbe7f3", "SOFR-1B instantaneous forward rate")}
{panel(180, 110, funding.state_data[0], funding.state_data[1], 0.0, 0.002, lambda v: f"{v*1e4:.0f} bp", "#8a4b00", "#f6e7d2", "SOFR-1B-FLAT spread over SOFR")}
{''.join(f'<line x1="{x(t):.1f}" y1="290" x2="{x(t):.1f}" y2="296" stroke="#888"/><text x="{x(t):.1f}" y="310" font-size="11" text-anchor="middle" fill="#555">{t:g}Y</text>' for t in (0, 0.5, 1, 1.5, 2))}
<text x="{x(T):.1f}" y="325" font-size="11" text-anchor="middle" fill="#b00">payment T = {T:.6f}</text>
</svg>"""

def table(headers, rows):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

gradient_rows, risk_rows = [], []
for position, component in enumerate((sofr, funding)):
    for node in range(component.num_state_data):
        gradient_rows.append([names[position], node_dates[position][node], f"{component.state_data[0][node]:.6f}", f"{component.state_data[1][node]:.3f}",
                              f"{weights[position][node]:.6f}", f"{local[position][node]:.6f}", f"{blocks[position][node]:.6f}"])
        risk_rows.append([names[position], node_dates[position][node], f"{risk[position][node]:,.6f}", f"{risk[position][node] * 1e-4:,.6f}"])

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FRE-GY 9743 Assignment 5 · Raj Pawar</title>
<style>
body{{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:#222;max-width:900px;margin:32px auto;padding:0 16px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:28px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}}
.meta{{color:#555;margin-bottom:16px}} a{{color:#1f4e79}} code{{background:#f4f4f4;padding:1px 4px;border-radius:3px}}
table{{border-collapse:collapse;margin:8px 0 12px;font-size:14px}} th,td{{border:1px solid #ddd;padding:4px 10px;text-align:right}} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
th{{background:#f7f7f7}} .formula{{font-family:"Times New Roman",serif;font-size:16px}}
</style></head><body>
<h1>FRE-GY 9743 · Assignment 5: valuation engine and first-order IFR risk</h1>
<div class="meta">Raj Pawar · branch <a href="https://github.com/pawarraj8888/FRE-GY-9743-Assignments-5/tree/Raj_Pawar_Branch">Raj_Pawar_Branch</a> · value date {VALUE_DATE}, payment date {PAYMENT_DATE}</div>
<p>Each curve component holds a piecewise-constant instantaneous forward rate, so its discount factor is
<span class="formula">D(T) = exp(&minus;&int;<sub>0</sub><sup>T</sup> f)</span> and its derivative with respect to node <i>i</i> is
<span class="formula">&part;D/&part;f<sub>i</sub> = &minus;D &middot; w<sub>i</sub></span>, where <span class="formula">w<sub>i</sub></span> is the length of
node <i>i</i>'s bucket inside [0, T]. The funding curve is <code>SOFR-1B-FLAT</code> = SOFR times its own spread factor, so by the product rule the SOFR block of its
gradient is <span class="formula">D<sub>spread</sub> &middot; &part;D<sub>SOFR</sub>/&part;f</span> and the spread block is <span class="formula">D<sub>SOFR</sub> &middot; &part;D<sub>spread</sub>/&part;s</span>.
The fixed-accrual cashflow pays C = notional &times; accrual on T, so PV = C &middot; D<sub>funding</sub>(T) and its risk is C times that gradient.</p>
<h2>The two step functions and the buckets that matter for the {PAYMENT_DATE} payment</h2>
{svg}
<p>Shaded areas are the integrals the discount factors are built from; the width of each shaded block is the node's bucket length w<sub>i</sub>, which is also its
gradient weight. The third SOFR node starts after the payment date and carries no risk; the first spread node carries risk although its spread is zero.</p>
<h2>Discount-factor gradients</h2>
<p>D<sub>SOFR</sub>(T) = {d_sofr:.6f}, D<sub>spread</sub>(T) = {d_spread:.6f}, D<sub>funding</sub>(T) = {d_sofr * d_spread:.6f}.</p>
{table(["Component", "Node", "Time (years)", "IFR", "w_i", "local dD/df_i", "dD_funding/df_i"], gradient_rows)}
<h2>Price and risk of the cashflow</h2>
<p>Notional 1,000,000, accrual 2026-04-02 to 2026-07-02 ACT/360 (unadjusted): C = {C:,.2f}, PV = C &middot; D<sub>funding</sub> = {PV:,.2f}, cash today 0.
The last column is the risk report's display unit: the signed PV change for a one basis point rise in the node.</p>
{table(["Component", "Node", "dPV/df_i", "per 1 bp"], risk_rows)}
<h2>What is on the branch</h2>
<ul>
<li><code>fixedincomelib/yield_curve/yield_curve_model.py</code>: TODO 1 (local discount-factor gradient) and TODO 2 (model gradient through the reference chain).</li>
<li><code>fixedincomelib/yield_curve/valuation_engine.py</code>: TODO 3 to 5 (funding-curve selection, PV and cash for future, same-day and past payments, risk).</li>
<li><code>hw5_valuation_and_first_order_risk.ipynb</code>: the professor's notebook executed top to bottom.</li>
<li><code>tests/test_yield_curve_gradient.py</code> and <code>tests/test_valuation_engine.py</code>: gradients by hand and by bump-and-reval, product rule, overwrite and accumulate, engine cases, reports.</li>
<li><code>writeup/Assignment5_Writeup_RajPawar.pdf</code>: Part I (implicit function differentiation, the vol-interpolator calibration) and a Part II summary.</li>
</ul>
<p class="meta">Page generated by <code>docs/build_page.py</code> from the library itself; the numbers are not typed in.</p>
</body></html>
"""
(ROOT / "docs" / "index.html").write_text(page)
print("wrote docs/index.html", len(page), "bytes")
