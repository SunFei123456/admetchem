from sqlalchemy.orm import Session
from typing import Optional
from app.models.models import User
from app.models.schemas import UserCreate, UserUpdate
import bcrypt


def get_password_hash(password: str) -> str:
    """哈希密码"""
    # 限制密码最大长度为 20 字节
    # 为了安全起见，在这里截断
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 20:
        password_bytes = password_bytes[:20]
    
    # 生成盐值并哈希密码
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    # 限制密码最大长度为 20 字节
    # 为了安全起见，在这里截断
    password_bytes = plain_password.encode('utf-8')
    if len(password_bytes) > 20:
        password_bytes = password_bytes[:20]
    
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def get_user(db: Session, user_id: int) -> Optional[User]:
    """通过ID获取用户"""
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """通过邮箱获取用户"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """通过用户名获取用户"""
    return db.query(User).filter(User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    """获取用户列表"""
    return db.query(User).offset(skip).limit(limit).all()


def create_user(db: Session, user: UserCreate) -> User:
    """创建新用户"""
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        is_active=user.is_active,
        avatar=user.avatar
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    验证用户登录
    
    Args:
        db: 数据库会话
        email: 用户邮箱
        password: 密码
    
    Returns:
        验证成功返回用户对象，否则返回None
    """
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def update_user(db: Session, user_id: int, user_update: UserUpdate) -> Optional[User]:
    """
    更新用户信息
    
    Args:
        db: 数据库会话
        user_id: 用户ID
        user_update: 更新数据
    
    Returns:
        更新后的用户对象
    """
    db_user = get_user(db, user_id)
    if not db_user:
        return None
    
    # 更新用户名
    if user_update.username is not None:
        db_user.username = user_update.username
    
    # 更新头像
    if user_update.avatar is not None:
        db_user.avatar = user_update.avatar
    
    # 更新密码
    if user_update.new_password is not None:
        # 如果要修改密码，需要验证旧密码
        if user_update.old_password is None:
            return None
        if not verify_password(user_update.old_password, db_user.hashed_password):
            return None
        db_user.hashed_password = get_password_hash(user_update.new_password)
    
    db.commit()
    db.refresh(db_user)
    return db_user
