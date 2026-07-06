# 我的算法博客

这是一个基于 [Hugo](https://gohugo.io/) 搭建的个人静态博客，用于整理和分享 AI 技术、论文解读、行业动态以及计算机基础技术笔记。

## 项目概览

博客当前按栏目组织内容，每个栏目都有独立的内容目录、文章模板和导航入口。

| 栏目 | 内容目录 | 默认子分类 | 说明 |
| --- | --- | --- | --- |
| AiNews | `content/ainews/` | 行业动态、产品发布、政策法规 | AI 行业新闻、产品发布和政策观察 |
| AiTech | `content/aitech/` | LLM基本原理、LLM最新技术、部署实践 | AI 技术实践、模型原理和工程部署 |
| CSTech | `content/cstech/` | Git、Python | 计算机基础技术和开发经验 |
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

## 写作流程

使用对应栏目的 archetype 创建文章：

```bash
hugo new ainews/my-ai-news.md
hugo new aitech/my-llm-note.md
hugo new cstech/my-git-note.md
hugo new paperreading/my-paper-reading.md
```

新文章会自动带上基础 front matter，例如：

```yaml
---
title: "文章标题"
date: 2026-07-05T10:00:00+08:00
draft: true
categories: ["AiTech"]
subcategories: ["LLM基本原理"]
tags: ["Transformer", "PyTorch"]
---
```

写作完成后，将 `draft` 改为 `false`，再运行 `hugo` 构建发布版本。

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
