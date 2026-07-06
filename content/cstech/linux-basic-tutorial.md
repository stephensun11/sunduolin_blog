---
title: "Linux 简单教学：从目录、文件到常用命令"
date: 2026-07-06T22:25:00+08:00
draft: false
categories: ["CSTech"]
subcategories: ["Linux"]
tags: ["Linux", "Shell", "命令行", "服务器"]
---

学 Linux 不需要一开始就理解内核、发行版和系统调用。

对大多数开发者来说，第一目标很实际：能登录服务器，能找到文件，能看日志，能启动程序，能判断一个问题大概出在哪里。先把这些基本动作练熟，Linux 就不会显得那么陌生。

这篇文章是一份简单教学，按日常使用顺序整理常用命令。

## 先理解目录

Linux 的文件系统从根目录 `/` 开始。

常见目录可以先这样理解。

```text
/home      普通用户的家目录
/root      root 用户的家目录
/etc       系统和软件配置
/var       日志、缓存、运行时数据
/usr       系统程序和库
/tmp       临时文件
/opt       额外安装的软件
```

查看当前目录：

```bash
pwd
```

列出当前目录文件：

```bash
ls
ls -l
ls -a
```

切换目录：

```bash
cd /var/log
cd ~
cd ..
```

`~` 表示当前用户的家目录，`..` 表示上一级目录。

## 文件和目录操作

创建目录：

```bash
mkdir notes
mkdir -p projects/blog/content
```

创建空文件：

```bash
touch hello.txt
```

复制文件：

```bash
cp hello.txt backup.txt
```

复制目录：

```bash
cp -r content content-backup
```

移动或重命名：

```bash
mv old.txt new.txt
mv file.txt notes/
```

删除文件：

```bash
rm old.txt
```

删除目录要更谨慎：

```bash
rm -r old-folder
```

`rm -rf` 很危险。它不会把文件放进回收站，执行前一定确认路径。

## 查看文件内容

查看整个文件：

```bash
cat README.md
```

分页查看：

```bash
less README.md
```

查看文件前几行和后几行：

```bash
head README.md
tail README.md
tail -f app.log
```

`tail -f` 很适合看实时日志。服务正在运行时，新的日志会不断追加显示。

搜索文件内容：

```bash
grep "error" app.log
grep -R "TODO" .
```

如果系统安装了 `rg`，也可以用它搜索，速度通常更快。

```bash
rg "error"
```

## 权限基础

Linux 文件有读、写、执行三类权限。

查看权限：

```bash
ls -l
```

你会看到类似这样的内容：

```text
-rw-r--r--  1 user user  1200 Jul  6 README.md
```

给脚本增加执行权限：

```bash
chmod +x deploy.sh
```

切换到管理员权限执行命令：

```bash
sudo systemctl status nginx
```

不要随手在不理解的命令前加 `sudo`。权限越大，误操作的代价越高。

## 进程和端口

查看当前进程：

```bash
ps aux
```

搜索某个进程：

```bash
ps aux | grep nginx
```

查看端口占用：

```bash
ss -tulpn
```

结束进程：

```bash
kill PID
```

如果普通结束失败，再考虑：

```bash
kill -9 PID
```

`kill -9` 是强制结束，适合最后再用。

## 服务管理

很多服务器程序会交给 systemd 管理，比如 nginx、mysql、docker。

查看服务状态：

```bash
systemctl status nginx
```

启动服务：

```bash
sudo systemctl start nginx
```

重启服务：

```bash
sudo systemctl restart nginx
```

设置开机自启：

```bash
sudo systemctl enable nginx
```

查看服务日志：

```bash
journalctl -u nginx -f
```

## 网络和下载

测试网络连通：

```bash
ping example.com
```

查看当前机器 IP：

```bash
ip addr
```

下载文件：

```bash
curl -O https://example.com/file.txt
wget https://example.com/file.txt
```

测试 HTTP 响应：

```bash
curl -I https://example.com
```

## 一个日常排查流程

假设你部署了一个网站，但页面打不开，可以按这个顺序看。

```bash
# 1. 服务是否在运行
systemctl status nginx

# 2. 端口是否在监听
ss -tulpn | grep 80

# 3. 本机访问是否正常
curl -I http://127.0.0.1

# 4. 日志里有没有错误
journalctl -u nginx -n 50
```

这个流程不一定解决所有问题，但能帮你快速判断问题在服务、端口、配置还是网络。

## 总结

Linux 入门不需要一次学完所有命令。

先掌握目录切换、文件操作、日志查看、权限、进程、服务管理这几组动作。只要能在服务器上稳定完成这些操作，你就已经具备了继续学习部署、自动化和系统维护的基础。

命令行真正的价值不是看起来专业，而是让你能清楚地控制一台机器。
