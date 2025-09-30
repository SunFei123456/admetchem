from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.models import schemas
from app.services import user_service
from app.utils.response import success, fail

router = APIRouter()


@router.post("/")
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = user_service.get_user_by_email(db, email=user.email)
    if db_user:
        return fail(message="Email already registered", code=10001, status_code=400)
    created = user_service.create_user(db=db, user=user)
    data = schemas.User.model_validate(created, from_attributes=True).model_dump()
    return success(data=data)


@router.get("/")
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = user_service.get_users(db, skip=skip, limit=limit)
    data = [
        schemas.User.model_validate(u, from_attributes=True).model_dump() for u in users
    ]
    return success(data=data)


@router.get("/{user_id}")
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = user_service.get_user(db, user_id=user_id)
    if db_user is None:
        return fail(message="User not found", code=10004, status_code=404)
    data = schemas.User.model_validate(db_user, from_attributes=True).model_dump()
    return success(data=data)
