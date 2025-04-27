"""
renderer.py
-----------
Render final HTML reports from Jinja2 templates.
"""
from jinja2 import Template

def render_report(template_file: str, context: dict, output_file: str = 'docs/report.html') -> None:
    """
    Render the provided Jinja2 template with context to output_file.
    """
    with open(template_file, 'r') as f:
        template_str = f.read()
    template = Template(template_str)
    report_html = template.render(context)
    with open(output_file, 'w') as f:
        f.write(report_html)
    print(f"Report written to {output_file}")