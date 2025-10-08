# ADMET 依赖安装指南

## 问题原因

您当前使用的是 **Python 3.13.7**,但 **DGL (Deep Graph Library)** 目前仅支持 Python 3.7-3.11,不支持 Python 3.13。

这导致了以下错误:
```
FileNotFoundError: Could not find module 'C:\Users\...\dgl\libdgl.dll'
```

## 解决方案

### 方案一:一键自动安装(推荐)

1. **运行自动安装脚本**:
   ```powershell
   cd backend
   .\一键安装ADMET依赖.bat
   ```

2. **如果提示未安装Python 3.11**,请:
   - 下载: [Python 3.11.9 (64-bit)](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
   - 安装时**勾选** "Add Python 3.11 to PATH"
   - 安装完成后重新运行 `.\一键安装ADMET依赖.bat`

3. **启动服务**:
   ```powershell
   # 激活虚拟环境
   .\venv_admet\Scripts\activate

   # 运行后端
   python run.py
   ```

   或直接:
   ```powershell
   .\venv_admet\Scripts\python.exe run.py
   ```

---

### 方案二:手动安装

#### 1. 安装Python 3.11

下载并安装: [Python 3.11.9 (64-bit)](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)

**重要**: 安装时勾选 "Add Python 3.11 to PATH"

#### 2. 创建虚拟环境

```powershell
cd backend
py -3.11 -m venv venv_admet
.\venv_admet\Scripts\activate
```

#### 3. 安装依赖

```powershell
# 升级pip
python -m pip install --upgrade pip

# 安装PyTorch (CPU版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装scikit-learn
pip install scikit-learn

# 安装DGL (关键!)
pip install dgl -f https://data.dgl.ai/wheels/torch-2.5/repo.html

# 安装其他依赖
pip install -r requirements.txt
```

#### 4. 运行服务

```powershell
python run.py
```

---

## 常见问题

### Q1: 我已经安装了很多依赖在Python 3.13,不想重装怎么办?

**A**: 使用虚拟环境不会影响您的全局Python 3.13环境。两个版本可以共存:
- Python 3.13: 用于其他项目
- Python 3.11虚拟环境 (`venv_admet`): 仅用于本项目的ADMET功能

### Q2: DGL什么时候支持Python 3.13?

**A**: DGL团队正在开发中,但目前没有明确时间表。建议使用Python 3.11以确保稳定性。

### Q3: 能否只卸载DGL,保留Python 3.13?

**A**: 不行。DGL的C++扩展模块依赖特定Python版本编译,必须使用兼容版本。

### Q4: 虚拟环境占用多少空间?

**A**: 约 3-4 GB (包含PyTorch、DGL等大型库)

---

## 验证安装

安装完成后,运行以下命令验证:

```powershell
# 激活虚拟环境
.\venv_admet\Scripts\activate

# 测试导入
python -c "import torch; import dgl; import sklearn; print('所有依赖安装成功!')"
```

如果没有报错,说明安装成功!

---

## 卸载

如需卸载ADMET依赖:

```powershell
# 删除虚拟环境文件夹
Remove-Item -Recurse -Force venv_admet
```

这不会影响您的全局Python环境。

