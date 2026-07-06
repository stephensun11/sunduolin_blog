---
title: "Python 实用入门：把重复工作写成脚本"
date: 2026-07-06T22:20:00+08:00
draft: false
categories: ["CSTech"]
subcategories: ["Python"]
tags: ["Python", "脚本", "自动化", "开发基础"]
---

Python 最适合作为第一门实用脚本语言来学。

它的语法不重，标准库够丰富，写几行代码就能处理文件、整理数据、调用命令、生成报告。对个人博客、AI 实验和日常开发来说，Python 很像一把随手能拿起来的小工具。

这篇文章不追求讲完 Python，而是帮你建立一个能立刻动手的最小知识框架。

## 安装与版本检查

安装完成后，先确认 Python 是否可用。

```bash
python --version
```

有些系统里命令叫 `python3`。

```bash
python3 --version
```

进入交互环境：

```bash
python
```

执行脚本：

```bash
python hello.py
```

一个最小脚本如下。

```python
print("Hello, Python")
```

## 虚拟环境

写 Python 项目时，建议每个项目使用独立虚拟环境。这样不同项目的依赖不会互相污染。

创建虚拟环境：

```bash
python -m venv .venv
```

激活虚拟环境。

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install requests
```

导出依赖：

```bash
pip freeze > requirements.txt
```

下次恢复环境：

```bash
pip install -r requirements.txt
```

## 变量和基本类型

Python 里常见的数据类型很直接。

```python
name = "Alice"
age = 18
score = 95.5
is_active = True

tags = ["AI", "Python", "Blog"]
profile = {"name": name, "age": age}
```

列表适合保存一组有顺序的数据。

```python
numbers = [1, 2, 3]
numbers.append(4)
print(numbers[0])
```

字典适合保存键值对。

```python
article = {
    "title": "Python 实用入门",
    "category": "CSTech",
    "draft": False,
}

print(article["title"])
```

## 条件和循环

条件判断：

```python
score = 86

if score >= 90:
    print("excellent")
elif score >= 60:
    print("passed")
else:
    print("failed")
```

循环列表：

```python
tags = ["Git", "Python", "Linux"]

for tag in tags:
    print(tag)
```

带索引循环：

```python
for index, tag in enumerate(tags, start=1):
    print(index, tag)
```

## 函数

函数的作用是给一段逻辑取名字。

```python
def slugify(title):
    return title.lower().replace(" ", "-")

print(slugify("Python Practical Start"))
```

当脚本变长以后，函数能让代码更容易读，也更容易测试。

## 文件读写

Python 很适合处理文本文件。

读取文件：

```python
from pathlib import Path

path = Path("README.md")
text = path.read_text(encoding="utf-8")
print(text[:100])
```

写入文件：

```python
from pathlib import Path

output = Path("output.txt")
output.write_text("hello\n", encoding="utf-8")
```

遍历目录里的 Markdown 文件：

```python
from pathlib import Path

for path in Path("content").rglob("*.md"):
    print(path)
```

这类脚本在维护博客时特别有用，比如批量检查文章标题、统计标签、生成索引。

## 处理 JSON

很多配置和接口数据都可以用 JSON 表示。

```python
import json

data = {
    "title": "Python 实用入门",
    "tags": ["Python", "脚本"],
}

text = json.dumps(data, ensure_ascii=False, indent=2)
print(text)
```

从文件读取 JSON：

```python
from pathlib import Path
import json

data = json.loads(Path("data.json").read_text(encoding="utf-8"))
print(data["title"])
```

## 写一个小脚本

下面这个脚本会统计 `content/` 目录下有多少 Markdown 文件。

```python
from pathlib import Path

root = Path("content")
files = list(root.rglob("*.md"))

print(f"Markdown 文件数量: {len(files)}")

for path in files:
    print("-", path)
```

保存为 `count_posts.py`，然后运行：

```bash
python count_posts.py
```

这个例子很小，但它已经包含了真实脚本的基本结构：导入模块、定位目录、处理文件、输出结果。

## 命令行参数

脚本写多了以后，不要把路径写死，可以用参数传入。

```python
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", help="要扫描的目录")
args = parser.parse_args()

for path in Path(args.root).rglob("*.md"):
    print(path)
```

运行：

```bash
python scan_md.py content
```

## 总结

Python 入门的关键不是语法背得多，而是尽快写出能帮自己省时间的脚本。

先掌握变量、列表、字典、函数、文件读写和命令行参数。等你能把重复操作写成脚本，Python 就不再是一门“正在学习的语言”，而会变成你每天能用上的工具。
