# models.py
"""
Data structures for the spec gap analysis pipeline.
Contains the core data models used across multiple modules.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Reference:
    """Represents a reference found in the spec text."""
    ref_type: str                # 'clause' or 'document'
    clause: Optional[str]
    document: Optional[str]
    context: str                 # the full paragraph/row text where it was found


@dataclass
class Section:
    """Represents a section in the spec with its content and structure."""
    number: str
    title: str
    level: int
    full_title: str
    text: List[str] = field(default_factory=list)
    # Each table: {'headers': List[str], 'rows': List[List[str]], 'references': List[Dict[str, Any]]}
    tables: List[Dict[str, Any]] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)
    subsections: Dict[str, "Section"] = field(default_factory=dict)
    parent: Optional["Section"] = None