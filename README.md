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

训练和生成可以分两次运行：

```powershell
.\.venv\Scripts\python.exe main.py train --steps 200
.\.venv\Scripts\python.exe main.py generate --prompt "hello" --max-new-tokens 44
```

训练默认使用内置的一句话，也可以通过 `train --corpus-file` 指定 UTF-8 文本。每批从语料中抽取窗口，把目标向后错开一个字符，然后计算交叉熵、反向传播并用 AdamW 更新参数。

训练结束后，`torch.save` 把 `model.state_dict()`、模型配置和字符词表一起写入 `model.pt`。生成进程用 `torch.load` 读取文件，按配置创建同样的模型，用 `load_state_dict()` 恢复权重，再切换到 `eval()`。词表让两个进程使用相同的字符 ID。生成时每次取最后一个位置分数最高的字符，追加到文本后继续预测。提示词只能使用训练词表中的字符。

两个命令都支持 `--checkpoint` 指定同一个文件路径。当前检查点包含生成所需的信息。若要从中断处续训，还需保存并恢复优化器状态和已完成的训练步数。

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
