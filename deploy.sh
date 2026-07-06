#!/bin/bash

# 博客部署脚本
# 用途：从GitHub拉取最新内容，构建Hugo站点，部署到nginx

set -e

# 配置
SITE_DIR="/var/www/myblog/myblog-site"
PUBLIC_DIR="$SITE_DIR/public"
GIT_REPO="${GIT_REPO:-https://github.com/yourusername/myblog.git}"

# 切换到站点目录
cd "$SITE_DIR"

# 检查是否已经是git仓库
if [ ! -d ".git" ]; then
    echo "初始化Git仓库..."
    git init
    git remote add origin "$GIT_REPO"
fi

# 拉取最新内容
echo "从GitHub拉取最新内容..."
git fetch origin
git reset --hard origin/main 2>/dev/null || git reset --hard origin/master 2>/dev/null || true

# 构建Hugo站点
echo "构建Hugo站点..."
hugo

# 设置权限
echo "设置文件权限..."
chmod -R 755 "$PUBLIC_DIR"

# 重启nginx (可选)
echo "部署完成！"
echo "博客已更新: http://$(hostname -I | awk '{print $1}')"