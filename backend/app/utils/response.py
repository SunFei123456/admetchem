from typing import Any, Optional, Dict
from datetime import datetime, timezone
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ResponseModel(BaseModel):
    code: int = 0
    message: str = "ok"
    data: Optional[Any] = None
    timestamp: int = int(datetime.now(tz=timezone.utc).timestamp())

    class Config:
        json_encoders = {
            datetime: lambda v: int(v.replace(tzinfo=timezone.utc).timestamp()),
        }


def _now_ts() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def success(
    data: Any = None,
    message: str = "ok",
    code: int = 0,
    status_code: int = 200,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    body = ResponseModel(
        code=code, message=message, data=data, timestamp=_now_ts()
    ).dict()
    return JSONResponse(status_code=status_code, content=body, headers=headers or {})


def fail(
    message: str = "error",
    code: int = 1,
    status_code: int = 400,
    data: Any = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    body = ResponseModel(
        code=code, message=message, data=data, timestamp=_now_ts()
    ).dict()
    return JSONResponse(status_code=status_code, content=body, headers=headers or {})


def paginate(
    items: Any,
    total: int,
    page: int,
    size: int,
    message: str = "ok",
    code: int = 0,
    status_code: int = 200,
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    payload: Dict[str, Any] = {
        "items": items,
        "total": total,
        "page": page,
        "size": size,
    }
    if extra:
        payload.update(extra)
    body = ResponseModel(
        code=code, message=message, data=payload, timestamp=_now_ts()
    ).dict()
    return JSONResponse(status_code=status_code, content=body)
