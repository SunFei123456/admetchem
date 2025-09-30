# ADMET 化学服务器 - 开发指南

## 构建/运行命令
- **启动服务器**: `python run.py`
- **使用Uvicorn启动**: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
- **运行测试**: `pytest`
- **运行单个测试**: `pytest tests/test_main.py::test_root`
- **代码检查**: `flake8 app/ tests/`
- **格式化代码**: `black app/ tests/`
- **类型检查**: `mypy app/`
- **安装依赖**: `pip install -r requirements.txt`
- **初始化数据库**: `python -c "from app.core.database import create_tables; create_tables()"`

## 代码风格规范

### 导入规范
- 使用绝对导入: `from app.core.config import settings`
- 分组导入: 标准库、第三方库、本地应用
- 在组内按字母顺序排序

### 格式化规范
- 使用Black格式化器（默认设置）
- 行长度: 88字符
- 使用双引号表示字符串
- 使用4个空格缩进

### 类型规范
- 为所有函数参数和返回值添加类型提示
- 使用Pydantic模型进行请求/响应验证
- 使用SQLAlchemy模型进行数据库操作

### 命名规范
- **函数**: snake_case (如: `get_user_by_email`)
- **变量**: snake_case (如: `db_user`)
- **类**: PascalCase (如: `UserService`)
- **常量**: UPPER_SNAKE_CASE (如: `API_V1_STR`)
- **文件名**: snake_case (如: `user_service.py`)

### 错误处理
- 使用HTTPException处理API错误
- 提供有意义的错误信息
- 使用适当的HTTP状态码
- 记录错误用于调试

### 数据库规范
- 使用SQLAlchemy ORM
- 使用依赖注入处理数据库会话
- 始终正确关闭数据库会话
- 使用Pydantic模式进行数据验证

### 测试规范
- 为所有服务编写单元测试
- 为API端点编写集成测试
- 使用pytest作为测试框架
- 模拟外部依赖

### 安全规范
- 绝不提交密钥或API密钥
- 使用环境变量进行配置
- 使用bcrypt哈希密码
- 验证所有输入数据


## 数据库初始化说明
- `create_tables()` 内部已自动导入模型（见 `app/core/database.py`），可直接创建所有声明于 `app/models/models.py` 的表。
- 若使用迁移管理，请参考下方 Alembic 流程，避免直接在生产环境使用 `create_all`。

## Alembic 迁移（可选，推荐）
- 初始化：`alembic init alembic`
- 配置：在 `alembic.ini` 设置 `sqlalchemy.url` 为 `.env` 中的 `DATABASE_URL`；或在 `env.py` 中从 `app.core.config` 读取。
- 生成迁移：`alembic revision --autogenerate -m "init"`
- 应用迁移：`alembic upgrade head`

## 质量工具配置（建议）
- 在 `pyproject.toml` 或 `setup.cfg` 中配置：
  - Black：行宽 88，排除 `.venv/`、`alembic/` 等。
  - Flake8：忽略 `E203,W503` 等与 Black 冲突的规则。
  - MyPy：可启用 `pydantic` 插件（Pydantic v2 可选）。

## 测试覆盖率（建议）
- 安装：`pip install pytest-cov`
- 运行：`pytest --cov=app --cov-report=term-missing`

## pre-commit（建议）
- 安装：`pip install pre-commit`
- 初始化：`pre-commit install`
- 配置 `.pre-commit-config.yaml`，挂钩 Black、Flake8、isort、Trivy（可选）等，提交前自动检查与格式化。

## API 约定（摘要）
- 版本化前缀：`/api/v1`（见 `app/core/config.py` 的 `API_V1_STR`）。
- 错误响应：建议统一返回 `{ "detail": str, "code": str, "data": any }` 结构，便于前端处理。

## 目录结构概览（摘要）
- `app/api/`：路由与端点（`app/api/routes.py` 汇总、`app/api/endpoints/users.py`）。
- `app/core/`：配置、数据库（`config.py`、`database.py`）。
- `app/models/`：ORM 与 Pydantic 模型（`models.py`、`schemas.py`）。
- `app/services/`：业务逻辑。
- `tests/`：测试用例。

## 环境与配置（摘要）
- 复制 `.env.example` 至 `.env`，至少设置：
  - `DATABASE_URL`（默认 MySQL，示例见 `app/core/config.py`）
  - `SECRET_KEY`、`ACCESS_TOKEN_EXPIRE_MINUTES`
- 确保 MySQL 服务可用并已创建数据库（详见 `README.md`）。
