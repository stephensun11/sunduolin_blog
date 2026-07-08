# Duolin Sun' Blog

这是一个基于 [Hugo](https://gohugo.io/) 搭建的个人静态博客，用于整理和分享 AI 技术、论文解读、行业动态以及计算机基础技术笔记。

## 项目概览

博客当前按栏目组织内容，每个栏目都有独立的内容目录、文章模板和导航入口。

| 栏目 | 内容目录 | 默认子分类 | 说明 |
| --- | --- | --- | --- |
| AiNews | `content/ainews/` | 行业动态、产品发布、政策法规 | AI 行业新闻、产品发布和政策观察 |
| AiTech | `content/aitech/` | LLM基本原理、LLM最新技术、强化学习、部署实践 | AI 技术实践、模型原理和工程部署 |
| CSTech | `content/cstech/` | Git、Python、Linux、tmux | 计算机基础技术和开发经验 |
| PaperReading | `content/paperreading/` | NLP、CV、多模态、RL | 经典与前沿论文解读 |

站点导航、分类和子分类映射主要配置在 `hugo.toml` 中；页面模板与样式位于 `layouts/`。

## 目录结构

```text
.
├── archetypes/          # Hugo 文章模板
├── content/             # Markdown 内容源文件
│   ├── ainews/
│   ├── aitech/
│   ├── cstech/
│   ├── paperreading/
│   └── search/
├── layouts/             # 自定义页面模板和局部模板
│   ├── _default/
│   └── partials/
├── public/              # Hugo 构建输出，可由 nginx 等静态服务托管
├── deploy.sh            # 服务器部署脚本
├── hugo.toml            # Hugo 站点配置
└── README.md
```

## 本地开发

### 1. 安装 Hugo

请先安装 Hugo，并确认命令可用：

```bash
hugo version
```

### 2. 启动本地预览

```bash
hugo server -D
```

启动后访问：

```text
http://localhost:1313
```

`-D` 表示预览草稿文章，适合写作和本地检查。

### 3. 构建静态文件

```bash
hugo
```

构建结果会输出到 `public/` 目录。

## 本地写文章流程

下面这套流程适合在本地 VSCode 里写新文章、预览效果，并把静态页面一起提交到仓库。

### 1. 打开项目并同步最新代码

在 PowerShell 中进入项目目录：

```powershell
cd E:\AI\sunduolin_blog
code .
```

建议先切到 `dev` 分支并拉取最新代码：

```powershell
git switch dev
git pull origin dev
```

### 2. 选择文章栏目

文章源文件都放在 `content/` 下，按栏目选择目录：

| 栏目 | 写入目录 | 可用子分类 |
| --- | --- | --- |
| AiNews | `content/ainews/` | 行业动态、产品发布、政策法规 |
| AiTech | `content/aitech/` | LLM基本原理、LLM最新技术、强化学习、部署实践 |
| CSTech | `content/cstech/` | Git、Python、Linux、tmux |
| PaperReading | `content/paperreading/` | NLP、CV、多模态、RL |

### 3. 创建文章文件

推荐用 Hugo 命令创建文章，这样会自动生成基础 front matter：

```powershell
hugo new aitech/my-new-note.md
```

也可以按栏目创建：

```powershell
hugo new ainews/my-ai-news.md
hugo new aitech/my-llm-note.md
hugo new cstech/my-git-note.md
hugo new paperreading/my-paper-reading.md
```

文件名建议使用英文小写和短横线，例如 `transformer-attention.md`。Hugo 会根据文件名生成文章访问路径。

### 4. 修改 front matter

打开新建的 Markdown 文件，先把顶部 front matter 改好：

```yaml
---
title: "文章标题"
date: 2026-07-08T10:00:00+08:00
draft: true
categories: ["AiTech"]
subcategories: ["LLM基本原理"]
topics: ["Transformer"]
tags: ["LLM", "Transformer"]
---
```

注意：

- `draft: true` 表示草稿，本地预览可见，正式发布前改成 `false`。
- `categories` 必须和栏目一致，例如 `AiTech`、`CSTech`。
- `subcategories` 要使用上表已有子分类，否则导航里可能不容易找到。
- `topics` 是可选字段。目前主要用于 `LLM基本原理` 页面里的三级目录分组。
- `tags` 用来生成标签页，建议 2-5 个即可。

### 5. 编写正文

正文直接使用 Markdown：

```markdown
## 第一节标题

这里写正文。

### 小节标题

- 要点一
- 要点二
```

文章页左侧目录会自动读取 `##`、`###` 等标题生成，不需要手动维护目录。

如果文章里有数学公式，推荐使用当前站点已经验证过的写法：

```html
行内公式：<span class="math-inline">\\(E=mc^2\\)</span>

块级公式：

<div class="math-display">\[
\mathcal{L}(\theta)=\mathbb{E}[r]
\]</div>
```

### 6. 添加图片

图片建议放到 `static/images/栏目/文章名/` 下，例如：

```text
static/images/aitech/my-new-note/architecture.png
```

在 Markdown 里这样引用：

```markdown
![架构图](../../images/aitech/my-new-note/architecture.png)
```

使用这种相对路径后，部署到服务器和本地直接打开 `public/index.html` 都更稳。

### 7. 本地预览

写作时启动本地预览：

```powershell
hugo server -D
```

浏览器打开：

```text
http://localhost:1313
```

检查重点：

- 首页是否出现新文章。
- 栏目页和子分类页是否能找到文章。
- 文章左侧目录是否正常跳转。
- 图片是否能显示。
- 公式、表格、代码块是否渲染正常。

### 8. 发布前构建

确认文章没问题后，把 front matter 中的草稿状态改成：

```yaml
draft: false
```

然后重新构建静态文件：

```powershell
hugo --cleanDestinationDir
```

构建结果会输出到 `public/`，这个仓库目前会把 `public/` 一起提交，服务器更新后能直接拿到最新静态页面。

### 9. 提交并推送

查看改动：

```powershell
git status
```

提交到 `dev`：

```powershell
git add -A
git commit -m "Add new blog article"
git push origin dev
```

确认没问题后合并到 `main`：

```powershell
git switch main
git pull origin main
git merge dev
git push origin main
```

如果服务器部署脚本拉取的是 `main`，推送 `main` 后再到服务器执行更新即可。

## 分类与导航

本项目启用了三类 taxonomy：

```toml
[taxonomies]
  category = "categories"
  subcategory = "subcategories"
  tag = "tags"
```

添加新栏目时，通常需要同步更新三处：

1. 在 `content/` 下创建栏目目录，并添加 `_index.md`。
2. 在 `hugo.toml` 的 `[[menu.main]]` 中添加导航入口。
3. 在 `hugo.toml` 的 `[params.section_subcategories]` 中维护栏目和子分类映射。

添加新子分类时，需要确保文章 front matter 中的 `subcategories` 与 `hugo.toml` 中的导航和映射保持一致。

## 部署

仓库包含 `deploy.sh`，用于服务器侧部署。脚本会执行以下操作：

1. 进入站点目录 `/var/www/myblog/myblog-site`。
2. 如有需要，初始化 Git 仓库并绑定远程地址。
3. 从远程仓库拉取 `main` 或 `master` 的最新内容。
4. 执行 `hugo` 构建站点。
5. 将 `public/` 目录权限设置为 `755`。

默认远程仓库地址可以通过环境变量覆盖：

```bash
GIT_REPO="https://github.com/yourusername/myblog.git" ./deploy.sh
```

服务器上的 nginx 或其他 Web 服务应指向：

```text
/var/www/myblog/myblog-site/public
```

## 常用命令

```bash
# 本地预览，包括草稿
hugo server -D

# 构建生产静态文件
hugo

# 创建 AiTech 文章
hugo new aitech/article-slug.md

# 创建论文解读文章
hugo new paperreading/paper-slug.md
```

## 维护提示

- `public/` 是构建产物，修改站点内容时优先编辑 `content/`、`layouts/` 和 `hugo.toml`。
- 样式目前主要内联在 `layouts/` 下的模板文件中。
- 修改菜单、栏目或子分类后，建议重新运行 `hugo server -D` 检查导航、侧边栏和 taxonomy 页面。
