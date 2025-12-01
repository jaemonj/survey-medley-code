#!/usr/bin/env python3
"""
Build the master analysis registry by combining all YAML files in analyses/
and generating a human-readable Markdown summary (block format).
"""

from pathlib import Path

import yaml

ANALYSIS_DIR = Path(__file__).parent / 'analyses'
MASTER_YAML = Path(__file__).parent / 'master_registry.yaml'
MASTER_MD = Path(__file__).parent / 'README.md'

# Preferred stage order for grouping in the markdown output
STAGE_ORDER = [
    'preprocessing',
    'time series',
    'time series and group',
    'group',
    'other',
]


def load_yaml_files(yaml_dir: Path):
    """Load all YAML files in a directory and return a list of dicts."""
    all_entries = []
    for yaml_file in sorted(yaml_dir.glob('*.yaml')):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            all_entries.append(data)
    return all_entries


def write_master_yaml(entries, output_file: Path):
    with open(output_file, 'w') as f:
        yaml.safe_dump(entries, f, sort_keys=False)


def stringify_entry(entry):
    """Convert YAML lists, None, or scalars into strings."""
    if entry is None:
        return 'None'
    if isinstance(entry, list):
        if not entry:
            return 'None'
        return ', '.join(str(x) for x in entry)
    return str(entry)


def clean_text_field(value):
    """
    Normalize text fields for Markdown:
    - Convert None → 'None'
    - Convert lists → single string via stringify_entry
    - Remove extra whitespace
    - Replace newlines with spaces
    """
    s = stringify_entry(value)
    return s.strip().replace('\n', ' ')


def sort_by_stage(entries):
    """Sort entries according to STAGE_ORDER; unknown stages go last."""
    stage_rank = {stage: i for i, stage in enumerate(STAGE_ORDER)}
    return sorted(
        entries, key=lambda e: stage_rank.get(e.get('stage', ''), len(STAGE_ORDER))
    )


def write_markdown_summary(entries, output_file: Path):
    md_lines = ['# Master Analysis Registry\n']

    # --- Summary Table ---
    md_lines.append('## Summary\n')
    md_lines.append('<table>')

    for stage in STAGE_ORDER:
        stage_entries = [e for e in entries if e.get('stage') == stage]
        if not stage_entries:
            continue

        # Stage header row
        md_lines.append(
            f'<tr><td colspan="5"><strong>{stage.title()}</strong></td></tr>'
        )
        md_lines.append(
            '<tr><th>ID</th><th>Description</th><th>Status</th><th>Result Files</th><th>Notes</th></tr>'
        )

        for e in stage_entries:
            desc = clean_text_field(e.get('description'))
            notes = clean_text_field(e.get('notes'))
            status = clean_text_field(e.get('status'))
            pretty_id = e.get('id', 'None').replace('_', ' ')

            # Result files
            result_files = e.get('result_files') or []
            if result_files:
                links = ', '.join(
                    [
                        f'<a href="../analyses/{pretty_id.replace(" ", "_")}/{rf}">{rf}</a>'
                        for rf in result_files
                    ]
                )
            else:
                links = 'None'

            md_lines.append(
                f'<tr><td>{pretty_id}</td><td>{desc}</td><td>{status}</td><td>{links}</td><td>{notes}</td></tr>'
            )

    md_lines.append('</table>\n---\n')
    md_lines.append('## Detailed Reports\n')

    for stage in STAGE_ORDER:
        stage_entries = [e for e in entries if e.get('stage') == stage]
        if not stage_entries:
            continue

        md_lines.append(f'\n## {stage.title()}\n')
        for e in stage_entries:
            pretty_id = e.get('id', 'Unknown ID')
            md_lines.append(f'### {pretty_id}')

            md_lines.append(f'**Name:** {clean_text_field(e.get("name"))}<br>')
            md_lines.append(
                f'**Description:** {clean_text_field(e.get("description"))}<br>'
            )
            md_lines.append(
                f'**Code Directory:** {clean_text_field(e.get("code_dir"))}<br>'
            )
            md_lines.append(
                f'**Dependencies:** {stringify_entry(e.get("dependencies"))}<br>'
            )
            md_lines.append(
                f'**Script Entry:** {stringify_entry(e.get("script_entry"))}<br>'
            )
            md_lines.append(
                f'**Notebook Entry:** {stringify_entry(e.get("notebook_entry"))}<br>'
            )
            md_lines.append(
                f'**Other Files:** {stringify_entry(e.get("other_files"))}<br>'
            )
            md_lines.append(
                f'**Output Directory:** {stringify_entry(e.get("output_dir"))}<br>'
            )

            # Result files in detailed section
            result_files = e.get('result_files') or []
            if result_files:
                rf_links = ', '.join(
                    [
                        f'<a href="../analyses/{pretty_id.replace(" ", "_")}/{rf}">{rf}</a>'
                        for rf in result_files
                    ]
                )
            else:
                rf_links = 'None'
            md_lines.append(f'**Result Files:** {rf_links}<br>')

            md_lines.append(
                f'**Hypothesis:** {clean_text_field(e.get("hypothesis"))}<br>'
            )
            md_lines.append(
                f'**Conclusion:** {clean_text_field(e.get("conclusion"))}<br>'
            )
            md_lines.append(f'**Notes:** {clean_text_field(e.get("notes"))}<br>')
            md_lines.append(f'**Status:** {clean_text_field(e.get("status"))}<br>')
            md_lines.append(
                f'**Last Updated:** {clean_text_field(e.get("last_updated"))}<br>'
            )
            md_lines.append(f'**Authors:** {stringify_entry(e.get("authors"))}<br>')
            md_lines.append('\n---\n')

    with open(output_file, 'w') as f:
        f.write('\n'.join(md_lines))


def write_markdown_summary_OLD(entries, output_file: Path):
    """Write a Markdown summary with sectioned tables and sectioned detailed reports."""
    md_lines = ['# Master Analysis Registry\n']

    # --- Summary Table ---
    md_lines.append('## Summary\n')
    md_lines.append('<table>')

    for stage in STAGE_ORDER:
        stage_entries = [e for e in entries if e.get('stage') == stage]
        if not stage_entries:
            continue

        # Stage header row spanning all columns
        md_lines.append(
            f'<tr><td colspan="4"><strong>{stage.title()}</strong></td></tr>'
        )
        md_lines.append(
            '<tr><th>ID</th><th>Description</th><th>Status</th><th>Notes</th></tr>'
        )

        for e in stage_entries:
            desc = clean_text_field(e.get('description'))
            notes = clean_text_field(e.get('notes'))
            status = clean_text_field(e.get('status'))
            pretty_id = e.get('id', 'None').replace('_', ' ')
            md_lines.append(
                f'<tr><td>{pretty_id}</td><td>{desc}</td><td>{status}</td><td>{notes}</td></tr>'
            )

    md_lines.append('</table>')

    # Divider before detailed section
    md_lines.append('\n---\n')
    md_lines.append('## Detailed Reports\n')

    # ==========================================================
    # Detailed Reports Grouped by Stage
    # ==========================================================
    for stage in STAGE_ORDER:
        stage_entries = [e for e in entries if e.get('stage') == stage]
        if not stage_entries:
            continue

        md_lines.append(f'\n## {stage.title()}\n')
        for e in stage_entries:
            md_lines.append(f'### {e.get("id", "Unknown ID")}')

            md_lines.append(f'**Name:** {clean_text_field(e.get("name"))}<br>')
            md_lines.append(
                f'**Description:** {clean_text_field(e.get("description"))}<br>'
            )
            md_lines.append(
                f'**Code Directory:** {clean_text_field(e.get("code_dir"))}<br>'
            )
            md_lines.append(
                f'**Dependencies:** {stringify_entry(e.get("dependencies"))}<br>'
            )
            md_lines.append(
                f'**Script Entry:** {stringify_entry(e.get("script_entry"))}<br>'
            )
            md_lines.append(
                f'**Notebook Entry:** {stringify_entry(e.get("notebook_entry"))}<br>'
            )
            md_lines.append(
                f'**Other Files:** {stringify_entry(e.get("other_files"))}<br>'
            )
            md_lines.append(
                f'**Output Directory:** {stringify_entry(e.get("output_dir"))}<br>'
            )
            md_lines.append(
                f'**Hypothesis:** {clean_text_field(e.get("hypothesis"))}<br>'
            )
            md_lines.append(
                f'**Conclusion:** {clean_text_field(e.get("conclusion"))}<br>'
            )
            md_lines.append(f'**Notes:** {clean_text_field(e.get("notes"))}<br>')
            md_lines.append(f'**Status:** {clean_text_field(e.get("status"))}<br>')
            md_lines.append(
                f'**Last Updated:** {clean_text_field(e.get("last_updated"))}<br>'
            )
            md_lines.append(f'**Authors:** {stringify_entry(e.get("authors"))}<br>')

            md_lines.append('\n---\n')

    # Write final markdown
    with open(output_file, 'w') as f:
        f.write('\n'.join(md_lines))


def main():
    entries = load_yaml_files(ANALYSIS_DIR)
    # Preserve raw YAML output as-is
    write_master_yaml(entries, MASTER_YAML)
    # Create Markdown with grouping by stage
    write_markdown_summary(entries, MASTER_MD)
    print(f'Master YAML written to {MASTER_YAML}')
    print(f'Markdown summary written to {MASTER_MD}')


if __name__ == '__main__':
    main()
