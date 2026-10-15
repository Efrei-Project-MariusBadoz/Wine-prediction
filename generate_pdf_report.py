import re
import os
import subprocess
from markdown_it import MarkdownIt

# 1. Read Markdown
with open("rapport_technique.md", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Clean LaTeX to elegant HTML/Unicode
replacements = [
    (r"\$Q1 - 1\.5 \s*\\times\s* IQR\$", "<em>Q1 − 1.5 × IQR</em>"),
    (r"\$Q3 \+ 1\.5 \s*\\times\s* IQR\$", "<em>Q3 + 1.5 × IQR</em>"),
    (r"\$r = \+0\.39\$", "<em>r</em> = +0.39"),
    (r"\$r = -0\.27\$", "<em>r</em> = −0.27"),
    (r"\$r = -0\.23\$", "<em>r</em> = −0.23"),
    (r"\$r = -0\.20\$", "<em>r</em> = −0.20"),
    (r"\$r = -0\.69\$", "<em>r</em> = −0.69"),
    (r"\$p < 10\^\{-100\}\$", "<em>p</em> &lt; 10<sup>−100</sup>"),
    (r"\$p < 10\^\{-78\}\$", "<em>p</em> &lt; 10<sup>−78</sup>"),
    (r"\$p < 10\^\{-64\}\$", "<em>p</em> &lt; 10<sup>−64</sup>"),
    (r"\$p < 10\^\{-46\}\$", "<em>p</em> &lt; 10<sup>−46</sup>"),
    (r"\$z = w\^T x \+ b\$", "<em>z = w<sup>T</sup>x + b</em>"),
    (r"\$\\sigma\(z\) = 1 / \(1 \+ e\^\{-z\}\)\$", "<em>σ(z) = 1 / (1 + e<sup>−z</sup>)</em>"),
    (r"\$w_i = 1 / d\$", "<em>w<sub>i</sub> = 1 / d</em>"),
    (r"\$C = 10\$", "<em>C</em> = 10"),
    (r"\$\\gamma = \\text\{'auto'\}\$", "<em>γ</em> = 'auto'"),
    (r"\$C=10, \\gamma=\\text\{auto\}\$", "<em>C</em> = 10, <em>γ</em> = 'auto'"),
    (r"\$k=7, \\text\{distance\}\$", "<em>k</em> = 7, distance"),
    (r"\$N = 1\\,300\$", "<em>N</em> = 1 300"),
    (r"\$N = 5\\,197\$", "<em>N</em> = 5 197"),
    (r"\$([A-Za-z_0-9]+)\$", r"<em>\1</em>")
]

for pat, repl in replacements:
    text = re.sub(pat, repl, text)

# 3. Render Markdown to HTML
md = MarkdownIt().enable("table")
body_html = md.render(text)

# 4. Wrap with academic styling
full_html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Rapport Technique : Classification de la Qualité du Vin</title>
    <style>
        @page {{
            size: A4;
            margin: 22mm 20mm 22mm 20mm;
            @bottom-right {{
                content: counter(page);
            }}
        }}
        
        body {{
            font-family: 'Times New Roman', Times, 'DejaVu Serif', serif;
            font-size: 11pt;
            line-height: 1.55;
            color: #1e293b;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
        }}
        
        /* Headers */
        h1 {{
            font-size: 20pt;
            color: #1b365d;
            border-bottom: 2px solid #1b365d;
            padding-bottom: 8px;
            margin-top: 0;
            margin-bottom: 16px;
            font-weight: bold;
            line-height: 1.25;
        }}
        
        h2 {{
            font-size: 14pt;
            color: #1b365d;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 5px;
            margin-top: 24px;
            margin-bottom: 12px;
            page-break-after: avoid;
            font-weight: bold;
        }}
        
        h3 {{
            font-size: 12pt;
            color: #6b2d5c;
            margin-top: 18px;
            margin-bottom: 8px;
            page-break-after: avoid;
            font-weight: bold;
        }}
        
        p {{
            margin-top: 0;
            margin-bottom: 10px;
            text-align: justify;
        }}
        
        ul, ol {{
            margin-top: 0;
            margin-bottom: 12px;
            padding-left: 24px;
        }}
        
        li {{
            margin-bottom: 4px;
        }}
        
        /* Header Meta Block */
        p:has(strong:first-child) {{
            line-height: 1.6;
        }}
        
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 20px 0;
        }}
        
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 18px 0;
            page-break-inside: avoid;
            font-size: 10pt;
        }}
        
        th {{
            background-color: #1b365d;
            color: #ffffff;
            font-weight: bold;
            text-align: center;
            padding: 8px 10px;
            border: 1px solid #1b365d;
        }}
        
        td {{
            padding: 7px 10px;
            border: 1px solid #cbd5e1;
            text-align: center;
        }}
        
        tr:nth-child(even) td {{
            background-color: #f8fafc;
        }}
        
        td:first-child {{
            text-align: left;
            font-weight: 500;
        }}
        
        /* Inline code */
        code {{
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 9.5pt;
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 1px 4px;
            border-radius: 3px;
            border: 1px solid #e2e8f0;
        }}
        
        strong {{
            color: #0f172a;
        }}
        
        /* Print rules */
        @media print {{
            body {{
                font-size: 10.5pt;
            }}
            h2, h3 {{
                page-break-after: avoid;
            }}
            table, tr, td, th {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
{body_html}
</body>
</html>
"""

# Save intermediate HTML
html_path = "rapport_technique.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(full_html)
print(f"Generated HTML report: {html_path}")

# 5. Convert to PDF via headless Google Chrome
chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
pdf_path = "rapport_technique.pdf"
abs_html_uri = f"file://{os.path.abspath(html_path)}"

cmd = [
    chrome_path,
    "--headless",
    "--disable-gpu",
    f"--print-to-pdf={pdf_path}",
    "--no-pdf-header-footer",
    abs_html_uri
]

print("Converting HTML to PDF via headless Chrome...")
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode == 0 and os.path.exists(pdf_path):
    print(f"PDF successfully generated: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
else:
    print(f"Error during PDF generation: {res.stderr}")
