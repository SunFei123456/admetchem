from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.core.auth import get_current_user, create_access_token
from app.core.config import settings
from app.models import schemas
from app.models.models import User
from app.services import user_service
from app.utils.response import success, fail

router = APIRouter()


@router.post("/register", summary="用户注册")
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    用户注册接口
    - 检查邮箱是否已存在
    - 检查用户名是否已存在
    - 创建新用户
    """
    # 检查邮箱是否已注册
    db_user = user_service.get_user_by_email(db, email=user.email)
    if db_user:
        return fail(message="邮箱已被注册", code=10001, status_code=400)
    
    # 检查用户名是否已存在
    db_user = user_service.get_user_by_username(db, username=user.username)
    if db_user:
        return fail(message="用户名已被使用", code=10002, status_code=400)
    
    # 创建用户
    created = user_service.create_user(db=db, user=user)
    data = schemas.User.model_validate(created, from_attributes=True).model_dump(mode='json')
    return success(data=data, message="注册成功")


@router.post("/login", response_model=None, summary="用户登录")
def login(user_login: schemas.UserLogin, db: Session = Depends(get_db)):
    """
    用户登录接口
    - 验证邮箱和密码
    - 返回JWT访问令牌
    """
    # 验证用户
    user = user_service.authenticate_user(db, user_login.email, user_login.password)
    if not user:
        return fail(message="邮箱或密码错误", code=10003, status_code=401)
    
    # 检查用户是否激活
    if not user.is_active:
        return fail(message="用户账号已被禁用", code=10005, status_code=400)
    
    # 创建访问令牌
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email},  # 将 user.id 转为字符串
        expires_delta=access_token_expires
    )
    
    # 返回用户信息和令牌
    user_data = schemas.User.model_validate(user, from_attributes=True).model_dump(mode='json')
    return success(
        data={
            "user": user_data,
            "token": {
                "access_token": access_token,
                "token_type": "bearer"
            }
        },
        message="登录成功"
    )


@router.get("/me", summary="获取当前用户信息")
def get_me(current_user: User = Depends(get_current_user)):
    """
    获取当前登录用户信息
    需要在请求头中携带有效的JWT令牌
    """
    data = schemas.User.model_validate(current_user, from_attributes=True).model_dump(mode='json')
    return success(data=data)


@router.put("/me", summary="更新当前用户信息")
def update_me(
    user_update: schemas.UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    更新当前用户信息
    - 可以修改用户名
    - 可以修改密码（需要提供旧密码）
    """
    # 如果要修改用户名，检查是否已被使用
    if user_update.username and user_update.username != current_user.username:
        existing_user = user_service.get_user_by_username(db, user_update.username)
        if existing_user:
            return fail(message="用户名已被使用", code=10002, status_code=400)
    
    # 更新用户
    updated_user = user_service.update_user(db, current_user.id, user_update)
    
    if not updated_user:
        return fail(message="更新失败，请检查旧密码是否正确", code=10006, status_code=400)
    
    data = schemas.User.model_validate(updated_user, from_attributes=True).model_dump(mode='json')
    return success(data=data, message="更新成功")


@router.get("/", summary="获取用户列表（管理员）")
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """获取用户列表"""
    users = user_service.get_users(db, skip=skip, limit=limit)
    data = [
        schemas.User.model_validate(u, from_attributes=True).model_dump(mode='json') for u in users
    ]
    return success(data=data)


@router.get("/{user_id}", summary="获取指定用户信息")
def read_user(user_id: int, db: Session = Depends(get_db)):
    """通过ID获取用户信息"""
    db_user = user_service.get_user(db, user_id=user_id)
    if db_user is None:
        return fail(message="用户不存在", code=10004, status_code=404)
    data = schemas.User.model_validate(db_user, from_attributes=True).model_dump(mode='json')
    return success(data=data)
