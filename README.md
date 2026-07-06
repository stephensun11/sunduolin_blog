# 我的算法博客

基于Hugo搭建的个人博客，专门用于分享AI技术、论文解读和技术心得。

## 博客栏目

- **AiNews**: 分享AI行业最新动态和新闻
- **PaperReading**: 深度解读经典和前沿论文
- **AiTech**: 技术实践和代码分享
- 可轻松添加更多栏目

## 本地开发工作流

### 1. 克隆仓库到本地
```bash
git clone https://github.com/yourusername/myblog.git
cd myblog
```

### 2. 添加新文章
```bash
# 在content/posts/目录下创建新的Markdown文件
# 文件头部需要包含frontmatter

---
title: "文章标题"
date: 2026-07-05T10:00:00+08:00
draft: false
categories: ["AiNews"]
tags: ["AI", "标签"]
---

文章内容...
```

### 3. 本地预览
```bash
hugo server -D
# 访问 http://localhost:1313
```

### 4. 提交并推送到GitHub
```bash
git add .
git commit -m "添加新文章"
git push origin main
```

### 5. 服务器自动更新
GitHub Actions会自动构建并部署到服务器。

或者手动在服务器上执行：
```bash
/var/www/myblog/myblog-site/deploy.sh
```

## 服务器配置

### 目录结构
```
/var/www/myblog/
└── myblog-site/
    ├── content/          # 博客内容
    │   └── posts/        # 文章存放目录
    ├── layouts/          # 自定义模板
    ├── public/           # 构建输出 (由nginx提供)
    ├── hugo.toml         # Hugo配置文件
    └── deploy.sh         # 部署脚本
```

### nginx配置
nginx已配置为提供 `/var/www/myblog/myblog-site/public` 目录的内容。

## 添加新栏目

1. 在文章的frontmatter中添加新的分类：
```yaml
categories: ["你的新栏目"]
```

2. 在 `hugo.toml` 中添加导航菜单项：
```toml
[[menu.main]]
  name = "你的新栏目"
  weight = 6
  url = "/categories/你的栏目英文名/"
```

## 技术栈

- **Hugo**: 静态网站生成器
- **Nginx**: Web服务器
- **GitHub**: 代码托管和CI/CD
- **Markdown**: 文章格式

## 常用命令

```bash
# 构建站点
hugo

# 启动本地开发服务器
hugo server -D

# 创建新文章
hugo new posts/your-article.md
```

## 自定义样式

所有样式都在 `layouts/` 目录下的HTML模板中内联定义，可以直接修改CSS样式。