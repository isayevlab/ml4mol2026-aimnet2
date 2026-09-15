"""Shared page shell for the tutorial site. CMU palette, white ground."""

CSS = """
:root{
  --red:#C41230;      /* Carnegie Red */
  --iron:#6D6E71;     /* Iron Gray   */
  --steel:#E0E0E0;    /* Steel Gray  */
  --bg-soft:#F6F6F7;
  --serif:"Source Serif 4","Source Serif Pro",Times,serif;
  --sans:"Open Sans",Helvetica,Arial,sans-serif;
  --mono:"Source Code Pro",ui-monospace,Menlo,Consolas,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth;scroll-padding-top:74px}
body{background:#fff;color:#000;font-family:var(--sans);line-height:1.6;
  -webkit-font-smoothing:antialiased}

/* ---------------------------------------------------------------- nav */
nav{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.96);
  backdrop-filter:blur(6px);border-bottom:1px solid var(--steel)}
nav .in{max-width:900px;margin:0 auto;padding:11px 24px;display:flex;
  align-items:baseline;gap:22px;flex-wrap:wrap}
nav .brand{font-family:var(--serif);font-weight:700;font-size:16px;
  text-decoration:none;color:#000;margin-right:auto;white-space:nowrap}
nav .brand span{color:var(--red)}
nav a{font-size:14px;color:var(--iron);text-decoration:none;white-space:nowrap}
nav a:hover{color:var(--red)}
nav a.on{color:#000;font-weight:600;box-shadow:inset 0 -2px 0 var(--red)}

.wrap{max-width:900px;margin:0 auto;padding:0 24px 96px}

/* ------------------------------------------------------------ headings */
h1{font-family:var(--serif);font-size:clamp(34px,5.5vw,50px);line-height:1.06;
  font-weight:700;letter-spacing:-.018em;margin:48px 0 10px}
.tag{font-size:19px;color:var(--iron);margin-bottom:22px}
.rule{width:96px;height:5px;background:var(--red);margin:0 0 34px}
h2{font-family:var(--serif);font-size:30px;line-height:1.2;font-weight:600;
  letter-spacing:-.01em;margin:52px 0 6px;padding-top:6px}
h2 + .lede{color:var(--iron);font-size:16.5px;margin-bottom:18px}
h3{font-family:var(--sans);font-size:17px;font-weight:700;margin:30px 0 8px;
  letter-spacing:-.005em}
h4{font-size:12.5px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:var(--red);margin:34px 0 12px}
p{font-size:16.5px;margin:0 0 14px}
a{color:#000;text-decoration-color:var(--steel);text-underline-offset:3px}
a:hover{text-decoration-color:var(--red)}
strong{font-weight:700}

ul,ol{margin:0 0 16px;padding-left:0;list-style:none}
ul li{position:relative;padding-left:22px;margin-bottom:7px;font-size:16.5px}
ul li::before{content:"";position:absolute;left:0;top:.66em;width:7px;height:7px;
  background:var(--red)}
ol{counter-reset:n}
ol li{counter-increment:n;position:relative;padding-left:30px;margin-bottom:9px;
  font-size:16.5px}
ol li::before{content:counter(n);position:absolute;left:0;top:0;
  font-family:var(--mono);font-size:13px;color:var(--red);font-weight:600}

/* --------------------------------------------------------------- code */
code{font-family:var(--mono);font-size:.88em;background:var(--bg-soft);
  padding:1px 6px;word-break:break-word}
pre{background:var(--bg-soft);border-left:4px solid var(--red);
  padding:15px 18px;overflow-x:auto;margin:0 0 16px;position:relative}
pre code{background:none;padding:0;font-size:13.5px;line-height:1.62;
  display:block;white-space:pre;word-break:normal}
pre .c{color:var(--iron)}
kbd{font-family:var(--mono);font-size:.82em;border:1px solid var(--steel);
  border-bottom-width:2px;border-radius:3px;padding:1px 5px;background:#fff}

/* -------------------------------------------------------------- tables */
.tw{overflow-x:auto;margin:0 0 18px}
table{border-collapse:collapse;width:100%;font-size:15.5px}
th{text-align:left;font-size:11.5px;font-weight:700;letter-spacing:.11em;
  text-transform:uppercase;color:var(--iron);padding:0 14px 8px 0;
  border-bottom:2px solid #000;white-space:nowrap}
th.asis{text-transform:none;letter-spacing:.02em;font-size:14px}
td{padding:9px 14px 9px 0;border-bottom:1px solid var(--steel);
  vertical-align:top;line-height:1.45}
td:first-child{font-weight:600}
td:first-child code{white-space:nowrap}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.yes{color:#1a7f37;font-weight:700}
.no{color:var(--red);font-weight:700}

/* ------------------------------------------------------------ callouts */
.note{border-left:4px solid var(--red);padding:2px 0 2px 20px;margin:0 0 20px}
.note .k{font-size:12.5px;font-weight:700;letter-spacing:.11em;
  text-transform:uppercase;color:var(--red);margin-bottom:5px}
.note p:last-child{margin-bottom:0}
.note.grey{border-left-color:var(--iron)}
.note.grey .k{color:var(--iron)}

/* ------------------------------------------------------------- big link */
.big{display:block;border:1px solid var(--steel);border-left:5px solid var(--red);
  padding:16px 20px;text-decoration:none;margin-bottom:10px}
.big:hover{border-color:var(--red)}
.big .t{font-family:var(--serif);font-size:21px;font-weight:700;display:block}
.big .d{font-size:15px;color:var(--iron);margin-top:2px;display:block}

/* --------------------------------------------------------------- tabs */
.tabs{display:flex;gap:2px;border-bottom:2px solid var(--steel);margin:0 0 20px;
  flex-wrap:wrap}
.tabs button{font-family:var(--sans);font-size:14.5px;font-weight:600;
  background:none;border:0;border-bottom:3px solid transparent;margin-bottom:-2px;
  padding:9px 15px;cursor:pointer;color:var(--iron)}
.tabs button[aria-selected="true"]{color:#000;border-bottom-color:var(--red)}
.panel[hidden]{display:none}

/* ------------------------------------------------------------ notebooks */
ol.nb{counter-reset:n -1}
ol.nb li{counter-increment:n;display:grid;
  grid-template-columns:28px minmax(0,1fr) auto;gap:4px 14px;align-items:baseline;
  padding:11px 0;border-bottom:1px solid var(--steel);margin:0}
ol.nb li::before{content:counter(n);position:static;font-family:var(--mono);
  font-size:13px;color:var(--red);font-weight:600}
ol.nb .nm{font-size:16.5px}
ol.nb .why{grid-column:2;font-size:14.5px;color:var(--iron);line-height:1.45}
ol.nb a.cl{font-size:12px;font-family:var(--mono);color:var(--iron);
  white-space:nowrap;letter-spacing:.02em}
.ex{font-weight:700}
.ex::after{content:"exercise";font-family:var(--sans);font-size:10px;font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;color:#fff;background:var(--red);
  padding:2px 7px;margin-left:9px;vertical-align:2px}

footer{border-top:1px solid var(--steel);margin-top:60px;padding-top:22px;
  font-size:14px;color:var(--iron)}
footer a{color:var(--iron)}

@media(max-width:600px){
  .wrap{padding:0 16px 72px}
  nav .in{padding:10px 16px;gap:14px}
  nav .brand{width:100%;margin-bottom:2px}
  ol.nb li{grid-template-columns:22px minmax(0,1fr)}
  ol.nb a.cl{grid-column:2;grid-row:3}
  h2{font-size:25px}
}
"""

PAGES = [("index.html", "Overview"), ("install.html", "Install"),
         ("notebooks.html", "Notebooks"), ("troubleshooting.html", "Troubleshooting"),
         ("teaching.html", "For instructors"), ("slides/", "Slides")]


def page(current, title, description, body, script=""):
    def link(href, label):
        cls = ' class="on"' if href == current else ""
        return f'<a href="{href}"{cls}>{label}</a>'

    nav = "".join(link(h, l) for h, l in PAGES)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&family=Source+Code+Pro:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap">
<style>{CSS}</style>
</head>
<body>
<nav><div class="in">
  <a class="brand" href="index.html">AIMNet2<span> · </span>tutorial</a>
  {nav}
</div></nav>
<div class="wrap">
{body}
<footer>
Olexandr Isayev, Carnegie Mellon University. AiMat Summer School 2026,
<em>Machine Learning for Molecules</em>. Materials MIT licensed.<br>
AIMNet2: Anstine, Zubatyuk and Isayev, <em>Chem. Sci.</em> <strong>2025</strong>, <em>16</em>, 10228.
AIMNet2-NSE: Kalita et al., <em>Angew. Chem. Int. Ed.</em> <strong>2026</strong>, e202516763.
</footer>
</div>
{script}
</body>
</html>
"""


TABS_JS = """<script>
document.querySelectorAll('[data-tabs]').forEach(group => {
  const buttons = [...group.querySelectorAll('button')];
  const panels  = [...group.parentElement.querySelectorAll('.panel')];
  const show = i => {
    buttons.forEach((b, k) => b.setAttribute('aria-selected', k === i));
    panels.forEach((p, k) => p.hidden = k !== i);
    try { localStorage.setItem(group.dataset.tabs, String(i)); } catch (e) {}
  };
  buttons.forEach((b, i) => b.addEventListener('click', () => show(i)));
  let start = 0;
  try {
    const saved = localStorage.getItem(group.dataset.tabs);
    if (saved !== null && buttons[+saved]) start = +saved;
  } catch (e) {}
  show(start);
});
</script>"""
