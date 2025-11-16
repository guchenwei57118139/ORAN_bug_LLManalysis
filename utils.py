# helpers.py
"""
Shared utility functions for the spec gap analysis pipeline.
Consolidates commonly used rendering, formatting, and utility functions.
"""

import re
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from spec_ingestor import SpecSliceIngestor

from models import Section


# ==========================
# String & Formatting Utilities
# ==========================

def slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    text = (text or "scenario").lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = re.sub(r'-{2,}', '-', text)
    return text.strip('-')[:80] or "scenario"


def numeric_sort_key(section_number: str) -> List[str]:
    """Convert section number to sortable key with consistent string types."""
    parts = []
    for part in section_number.split('.'):
        try:
            # Numeric parts: zero-pad for proper string sorting (e.g., "001", "002", "010")
            parts.append(f"{int(part):06d}")
        except ValueError:
            # Non-numeric parts: prefix to ensure they sort after numeric parts
            parts.append(f"zzz_{part}")
    return parts


# ==========================
# Table Rendering
# ==========================

def render_table_markdown(table: Dict[str, Any], *, include_references: bool = True) -> str:
    """
    Render a table dict as Markdown with optional per-table references.
    
    Args:
        table: Dict with 'headers', 'rows', and optionally 'references'
        include_references: Whether to include the references section under the table
    
    Returns:
        Markdown string for the table
    """
    headers = table.get('headers', [])
    rows = table.get('rows', [])
    
    if not headers and not rows:
        return ""
    
    lines = []
    
    # Table headers and separator
    if headers:
        lines.append("| " + " | ".join(h or "" for h in headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    # Table rows
    for row in rows:
        if headers:
            # Pad or truncate row to match header count
            data = row[:len(headers)] + [""] * max(0, len(headers) - len(row))
        else:
            data = row
        lines.append("| " + " | ".join(str(cell or "") for cell in data) + " |")
    
    # Per-table references (if requested and present)
    if include_references:
        refs = table.get("references", [])
        if refs:
            lines.append("")
            lines.append("_References under this table:_")
            
            # Deduplicate references by key
            seen_keys = set()
            unique_refs = []
            for ref in refs:
                key = (
                    ref.get("ref_type"),
                    (ref.get("clause") or "").strip(),
                    (ref.get("document") or "").strip()
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_refs.append(ref)
            
            # Render unique references
            for i, ref in enumerate(unique_refs, 1):
                ref_type = "Clause" if ref.get("ref_type") == "clause" else "Document"
                clause = f"`{ref.get('clause')}`" if ref.get('clause') else "`-`"
                doc = f"`{ref.get('document')}`" if ref.get('document') else "`-`"
                context = (ref.get("context") or "").strip().replace("\n", " ")
                
                lines.append(f"  {i}. **{ref_type}** | clause: {clause} | document: {doc}")
                if context:
                    lines.append(f"     - _\"{context}\"_")

    return "\n".join(lines)


# ==========================
# Section Rendering
# ==========================

def render_section_markdown(
    section: Section, 
    *,
    level: int = 3,
    include_subsections: bool = False,
    include_table_refs: bool = True
) -> str:
    """
    Render a section as Markdown.
    
    Args:
        section: Section object to render
        level: Heading level (1-6)
        include_subsections: Whether to recursively include subsections
        include_table_refs: Whether to include per-table references
    
    Returns:
        Markdown string for the section
    """
    # Clamp heading level
    hlevel = max(1, min(level, 6))
    parts = [f"{'#' * hlevel} §{section.number} {section.title}", ""]
    
    # Section text paragraphs
    for text in section.text:
        parts.append(text)
        parts.append("")
    
    # Tables with numbering
    for idx, table in enumerate(section.tables, 1):
        parts.append(f"**Table {idx}**")
        parts.append("")
        table_md = render_table_markdown(table, include_references=include_table_refs)
        parts.append(table_md or "_<empty table>_")
        parts.append("")
    
    # Recursively include subsections
    if include_subsections and section.subsections:
        for subsection_number in sorted(section.subsections.keys(), key=numeric_sort_key):
            subsection = section.subsections[subsection_number]
            parts.append(render_section_markdown(
                subsection,
                level=hlevel + 1,
                include_subsections=True,
                include_table_refs=include_table_refs
            ))
            parts.append("")
    
    return "\n".join(parts).rstrip()


def render_spec_excerpts(ingestor: "SpecSliceIngestor", section_numbers: List[str]) -> str:
    """
    Render authoritative excerpts for cited sections.
    
    Args:
        ingestor: SpecSliceIngestor instance with parsed document
        section_numbers: List of section numbers to render
    
    Returns:
        Markdown string with all requested sections
    """
    if not section_numbers:
        return "_No sections provided._"
    
    rendered_sections = []
    seen_numbers = set()
    
    # Sort sections numerically and deduplicate
    for section_num in sorted(section_numbers, key=numeric_sort_key):
        if section_num in seen_numbers:
            continue
        seen_numbers.add(section_num)
        
        section = ingestor._find_by_number(section_num)
        if section:
            section_md = render_section_markdown(
                section, 
                level=3, 
                include_subsections=False,
                include_table_refs=True
            )
            rendered_sections.append(section_md)
        else:
            rendered_sections.append(f"### §{section_num} (not found in parsed document)")
    
    return "\n\n".join(rendered_sections).strip() + ("\n" if rendered_sections else "")


# ==========================
# Section Tree Utilities
# ==========================

def find_section_by_number(ingestor: "SpecSliceIngestor", section_number: str) -> Optional[Section]:
    """Find a section by its number in the ingestor's section tree."""
    return ingestor._find_by_number(section_number)


def is_descendant_section(ancestor_number: str, descendant_number: str) -> bool:
    """Check if descendant_number is under ancestor_number (e.g., '9.2.1' under '9.2')."""
    if descendant_number == ancestor_number:
        return True
    return descendant_number.startswith(ancestor_number + ".")


def collect_subsection_numbers(section: Section) -> List[str]:
    """Recursively collect all subsection numbers under a section."""
    numbers = [section.number]
    for subsection in section.subsections.values():
        numbers.extend(collect_subsection_numbers(subsection))
    return numbers


# ==========================
# Reference Utilities
# ==========================

def format_reference_entry(ref: Dict[str, Any]) -> str:
    """Format a single reference entry for display."""
    ref_type = "Clause" if ref.get("ref_type") == "clause" else "Document"
    clause = ref.get("clause") or "-"
    document = ref.get("document") or "-"
    context = (ref.get("context") or "").strip()
    
    entry = f"**{ref_type}** | clause: `{clause}` | document: `{document}`"
    if context:
        entry += f"\n  - _\"{context}\"_"

    return entry


# ==========================
# Analysis Utilities
# ==========================

def count_keyword_occurrences(text: str, keyword: str) -> int:
    """Count case-insensitive occurrences of keyword in text."""
    return text.lower().count(keyword.lower())


def extract_messages_from_involved(involved: Dict[str, Any]) -> List[str]:
    """Extract message names from a scenario's 'involved' dict."""
    if not isinstance(involved, dict):
        return []
    return list(involved.get("messages", []))


def extract_ies_from_involved(involved: Dict[str, Any]) -> List[str]:
    """Extract IE names from a scenario's 'involved' dict."""
    if not isinstance(involved, dict):
        return []
    return list(involved.get("ies", []))


def extract_sections_from_involved(involved: Dict[str, Any]) -> List[str]:
    """Extract section numbers from a scenario's 'involved' dict."""
    if not isinstance(involved, dict):
        return []
    return list(involved.get("sections", []))