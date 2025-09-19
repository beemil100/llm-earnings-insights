from typing import List, Optional
from pydantic import BaseModel, Field
from typing import Dict


class SourceRef(BaseModel): # old
    page: int
    snippet: str

class Source(BaseModel):
    page: Optional[int] = None
    snippet: Optional[str] = None

class KPIField(BaseModel):
    value: float
    unit: str
    yoy_pct: Optional[float] = None
    qoq_pct: Optional[float] = None
    source: Optional[SourceRef] = None


class Segment(BaseModel):
    name: str
    revenue: KPIField
    yoy_pct: Optional[float] = None


class Guidance(BaseModel):
    status: str 
    text: Optional[str] = None
    drivers: Optional[List[str]] = None
    source: Optional[Source] = None   # Field(..., description="raise | maintain | lower | none")
    


class ManagementTone(BaseModel):
    score: Optional[float] = None # = Field(..., description="Sentiment score [-1,1]")
    top_phrases: Optional[List[str]] = None


class EarningsExtract(BaseModel):
    ticker: str
    filing_type: str
    period_end: str

    kpis: Dict[str, KPIField]
    guidance: Optional[Guidance] = None
    risks: Optional[List[str]] = []
    management_tone: Optional[ManagementTone] = None
    highlights: Optional[List[str]] = []
    lowlights: Optional[List[str]] = []
