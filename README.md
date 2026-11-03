# AILearning

Python 学习项目。当前使用 Python 3.13.7，建议各台电脑使用相同版本。

## 在另一台电脑上开始

先安装 Python、Git 和 VS Code 的 Python 扩展，然后在 PowerShell 执行：

```powershell
git clone git@github.com:LeviMilkYang/AILearning.git
cd AILearning
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

使用 SSH 克隆前，需要将该电脑的 SSH 公钥添加到 GitHub。

用 VS Code 打开项目，执行 `Python: Select Interpreter`，选择 `.venv\Scripts\python.exe`。之后新建终端会自动激活所选环境。

运行示例：

```powershell
.\.venv\Scripts\python.exe main.py
```

## 安装和同步依赖

在项目根目录使用虚拟环境的 Python 安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install 包名
```

安装或升级后，导出依赖清单（使用 UTF-8，避免旧版 PowerShell 重定向产生编码问题）：

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 requirements.txt
```

将代码和 `requirements.txt` 一起提交、推送到 Git。另一台电脑拉取最新代码后执行：

```powershell
git pull
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`.venv` 由每台电脑各自创建，通过 `.gitignore` 排除，不提交到 Git。
