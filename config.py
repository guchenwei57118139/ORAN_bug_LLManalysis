# config.py
from dataclasses import dataclass, field
from typing import Optional
from types import SimpleNamespace

@dataclass
class IngestConfig:
    # Section filtering
    start_section: Optional[str] = "8"
    end_section: Optional[str] = "9.4"
    exclude_toc: bool = True
    exclude_appendix: bool = True
    # Parsing behavior
    include_tables: bool = True
    # e.g. "9.2.2.1  UE CONTEXT SETUP"
    section_header_regex: str = r"^(\d+(?:\.\d+)*)\s+(.+)$"
@dataclass
class AppConfig:
    ingest: IngestConfig = field(default_factory=IngestConfig)        
    #analysis: AnalysisConfig = field(default_factory=AnalysisConfig)  

# Global defaults used across modules
DEFAULTS = AppConfig()