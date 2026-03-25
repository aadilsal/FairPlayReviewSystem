from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class WicketBox(BaseModel):
    # Pixel bbox stored as [x, y, w, h]
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    w: float = Field(..., gt=0)
    h: float = Field(..., gt=0)

    def as_list(self) -> List[float]:
        return [float(self.x), float(self.y), float(self.w), float(self.h)]


class WicketConfigurationOut(BaseModel):
    match_id: int
    user_id: int
    configured: bool
    near_box: Optional[List[float]] = None
    far_box: Optional[List[float]] = None
    updated_at: Optional[str] = None


class WicketConfigurationManualUpdate(BaseModel):
    configured: bool = True
    near_box: WicketBox
    far_box: WicketBox


class WicketConfigurationAutoResult(BaseModel):
    configured: bool
    near_box: Optional[List[float]] = None
    far_box: Optional[List[float]] = None
    raw_detections: List[Dict[str, Any]] = Field(default_factory=list)
