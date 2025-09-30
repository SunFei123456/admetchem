# FastAPI ADMET 化学服务器

基于FastAPI的ADMET（吸收、分布、代谢、排泄、毒性）化学预测服务器。

## 功能特性

- 用户管理的RESTful API
- 用户身份验证和授权
- SQLAlchemy数据库集成
- Pydantic数据验证模型
- 全面的测试套件

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量：
   ```bash
   cp .env.example .env
   # 编辑.env文件配置数据库连接等信息
   ```

3. 创建MySQL数据库：
   ```sql
   CREATE DATABASE admetchem_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```

4. 初始化数据库表：
   ```bash
   python -c "from app.core.database import create_tables; create_tables()"
   ```

5. 启动服务器：
   ```bash
   python run.py
   ```

6. 访问API文档：
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## API端点

- `GET /` - 根端点
- `GET /health` - 健康检查
- `POST /api/v1/users/` - 创建用户
- `GET /api/v1/users/` - 用户列表
- `GET /api/v1/users/{user_id}` - 获取用户信息

## 测试

运行所有测试：
```bash
pytest
```

运行单个测试：
```bash
pytest tests/test_main.py::test_root
```

## 数据库配置

项目使用MySQL数据库，请确保：
1. MySQL服务已启动
2. 创建了`admetchem_db`数据库
3. 在.env文件中正确配置了数据库连接字符串

## 开发指南

详细的开发规范请参考CRUSH.md文件，包含代码风格、测试规范、构建命令等信息。