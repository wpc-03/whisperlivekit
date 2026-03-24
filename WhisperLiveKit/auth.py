import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# 配置
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 密码加密上下文 - 使用pbkdf2_sha256避免bcrypt的密码长度限制
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# OAuth2密码承载令牌
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# 用户数据文件路径
USERS_FILE = Path(__file__).parent / "data" / "users.json"


class UserManager:
    """用户管理类"""
    
    def __init__(self):
        """初始化用户管理器"""
        self.users_file = USERS_FILE
        self._ensure_users_file_exists()
    
    def _ensure_users_file_exists(self) -> None:
        """确保用户文件存在，如果不存在则创建默认用户"""
        if not self.users_file.exists():
            # 创建默认用户，用户名: admin, 密码: admin
            # 确保密码不超过72字节
            password = "admin"
            if len(password.encode('utf-8')) > 72:
                password = password[:72]
            
            default_users = [
                {
                    "username": "admin",
                    "hashed_password": pwd_context.hash(password),
                    "disabled": False
                }
            ]
            
            try:
                os.makedirs(self.users_file.parent, exist_ok=True)
                with open(self.users_file, 'w', encoding='utf-8') as f:
                    json.dump(default_users, f, indent=2, ensure_ascii=False)
                logger.info(f"Created default users file at {self.users_file}")
            except Exception as e:
                logger.error(f"Failed to create users file: {e}")
                raise
    
    def get_users(self) -> list:
        """获取所有用户"""
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read users file: {e}")
            return []
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """根据用户名获取用户"""
        users = self.get_users()
        for user in users:
            if user["username"] == username:
                return user
        return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """验证用户"""
        user = self.get_user(username)
        if not user:
            return None
        # 确保密码不超过72字节
        if len(password.encode('utf-8')) > 72:
            password = password[:72]
        if not pwd_context.verify(password, user["hashed_password"]):
            return None
        if user.get("disabled", False):
            return None
        return user


user_manager = UserManager()


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = user_manager.get_user(username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """获取当前活跃用户"""
    if current_user.get("disabled", False):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
