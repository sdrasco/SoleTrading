# report/report_rendering.py

import pandas as pd
import numpy as np
from jinja2 import Template

def render_report(template_file, context, output_file="../docs/report.html"):
    with open(template_file, "r") as f:
        template_str = f.read()
    template = Template(template_str)
    report_html = template.render(context)
    with open(output_file, "w") as f:
        f.write(report_html)
    print(f"Report written to {output_file}")


def format_date_cell(d):
    """
    Helper to format a datetime object into an HTML cell with 
    a data-sort attribute for custom sorting in a table.
    """
    if pd.isnull(d):
        return ""
    return (
        f'<span style="white-space: nowrap;" data-sort="{d.strftime("%Y-%m-%d")}">'
        f'{d.day} {d.strftime("%b %Y")}'
        '</span>'
    )