@echo off
chcp 65001 >nul
echo ========================================
echo ADMET依赖自动安装脚本
echo ========================================
echo.
echo 检测Python版本...
python --version
echo.

:: 检查Python 3.11是否已安装
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到Python 3.11
    echo.
    echo DGL库需要Python 3.11或更低版本,您当前使用的Python 3.13不支持。
    echo.
    echo 请按以下步骤操作:
    echo 1. 下载Python 3.11.9: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    echo 2. 安装时勾选 "Add Python 3.11 to PATH"
    echo 3. 选择 "Install Now" 或 "Customize installation"
    echo 4. 安装完成后重新运行此脚本
    echo.
    pause
    exit /b 1
)

echo [成功] 检测到Python 3.11
echo.
echo 创建Python 3.11虚拟环境...
py -3.11 -m venv venv_admet

echo.
echo 激活虚拟环境...
call venv_admet\Scripts\activate.bat

echo.
echo 升级pip...
python -m pip install --upgrade pip

echo.
echo ========================================
echo 开始安装依赖 (这需要几分钟时间)
echo ========================================
echo.

echo [1/4] 安装PyTorch (CPU版本)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo [2/4] 安装scikit-learn...
pip install scikit-learn

echo.
echo [3/4] 安装DGL...
pip install dgl -f https://data.dgl.ai/wheels/torch-2.5/repo.html

echo.
echo [4/4] 安装其他依赖...
pip install -r requirements.txt

echo.
echo ========================================
echo 安装完成!
echo ========================================
echo.
echo 启动后端服务:
echo   .\venv_admet\Scripts\activate
echo   python run.py
echo.
echo 或直接运行:
echo   .\venv_admet\Scripts\python.exe run.py
echo.
pause

