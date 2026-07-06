---
title: "Tmux 实用入门：让终端会话不再怕断线"
date: 2026-07-06T22:00:00+08:00
draft: false
categories: ["CSTech"]
subcategories: ["tmux"]
tags: ["tmux", "Linux", "终端", "效率工具"]
---

如果你经常通过 SSH 登录服务器，tmux 是一个非常值得尽早掌握的工具。

它解决的问题很朴素：终端窗口关掉了、网络断了、笔记本合上了，服务器上的任务还能不能继续跑？没有 tmux 时，很多命令会跟着 SSH 会话一起结束；有了 tmux，会话可以留在服务器后台，等你重新登录后再接回去。

这篇文章参考了阮一峰老师的《[Tmux 使用教程](https://www.ruanyifeng.com/blog/2019/10/tmux.html)》，但会按照我日常使用服务器和开发环境的习惯，整理成一份更偏实战的入门笔记。

## tmux 到底在管什么

理解 tmux，先记住三个概念。

**Session** 是会话。你可以把它理解成一组长期存在的终端工作区。即使终端窗口关闭，session 也可以继续留在后台。

**Window** 是窗口。一个 session 里可以有多个 window，比如一个用来跑服务，一个用来查看日志，一个用来编辑文件。

**Pane** 是窗格。一个 window 可以继续拆成多个 pane，比如左边跑程序，右边看日志，下面开一个临时命令行。

所以 tmux 的层级大致是：

```text
session
└── window
    └── pane
```

日常使用时，不需要一开始就记住所有命令。先把 session 用熟，就已经能解决 80% 的断线焦虑。

## 安装 tmux

不同系统的安装方式略有差别。

```bash
# Ubuntu / Debian
sudo apt-get install tmux

# CentOS / Fedora
sudo yum install tmux

# macOS
brew install tmux
```

安装完成后，确认版本：

```bash
tmux -V
```

## 最小可用流程

第一次使用 tmux，我建议只记这一组命令。

```bash
# 新建一个名为 work 的会话
tmux new -s work
```

进入 tmux 后，可以像普通终端一样执行命令。比如跑一个长时间任务：

```bash
python train.py
```

如果你要临时离开，不要直接关任务，而是把会话挂到后台。

快捷键是：

```text
Ctrl+b，然后按 d
```

这里容易误解的一点是：不是一直按着 `Ctrl+b+d`。正确操作是先按 `Ctrl+b`，松开，再按 `d`。

回到普通终端后，可以查看后台有哪些 tmux 会话：

```bash
tmux ls
```

重新接回刚才的会话：

```bash
tmux attach -t work
```

这就是 tmux 最核心的价值：任务还在那里，终端只是重新连了回去。

## 会话管理

给 session 起名字很重要。不要依赖默认编号，否则会话多了以后很难分清哪个是哪个。

```bash
# 新建会话
tmux new -s blog

# 查看所有会话
tmux ls

# 接入会话
tmux attach -t blog

# 重命名会话
tmux rename-session -t blog writing

# 杀掉会话
tmux kill-session -t writing
```

常用快捷键也不多。

| 快捷键 | 作用 |
| --- | --- |
| `Ctrl+b d` | 分离当前会话 |
| `Ctrl+b s` | 列出并切换会话 |
| `Ctrl+b $` | 重命名当前会话 |

如果你只在服务器上跑任务，掌握这些就已经够用了。

## 窗口管理

当一个 session 里需要做几件事时，可以创建多个 window。

比如我会把一个项目拆成这样：

```text
0: editor   编辑文件
1: server   启动服务
2: logs     查看日志
3: shell    临时命令
```

常用快捷键如下。

| 快捷键 | 作用 |
| --- | --- |
| `Ctrl+b c` | 创建新窗口 |
| `Ctrl+b n` | 切到下一个窗口 |
| `Ctrl+b p` | 切到上一个窗口 |
| `Ctrl+b 0` 到 `Ctrl+b 9` | 切到指定编号窗口 |
| `Ctrl+b ,` | 重命名当前窗口 |
| `Ctrl+b w` | 从列表中选择窗口 |

命令行方式也可以做到同样的事情。

```bash
# 创建新窗口
tmux new-window

# 创建指定名称的窗口
tmux new-window -n logs

# 重命名当前窗口
tmux rename-window server
```

快捷键更适合交互使用，命令更适合写进脚本里。

## 窗格管理

pane 适合临时对照信息，比如一边跑测试，一边看日志。

| 快捷键 | 作用 |
| --- | --- |
| `Ctrl+b %` | 左右分屏 |
| `Ctrl+b "` | 上下分屏 |
| `Ctrl+b 方向键` | 在窗格之间移动 |
| `Ctrl+b x` | 关闭当前窗格 |
| `Ctrl+b z` | 当前窗格全屏，再按一次恢复 |
| `Ctrl+b q` | 显示窗格编号 |

也可以用命令创建窗格。

```bash
# 上下分屏
tmux split-window

# 左右分屏
tmux split-window -h
```

窗格很好用，但不要贪多。一个屏幕里塞太多 pane，最后往往比多个 window 更难看清楚。我的习惯是：长期任务放 window，临时对照用 pane。

## 一个推荐的日常工作流

假设我要在服务器上维护一个博客项目，可以这样开始：

```bash
tmux new -s blog
```

进入后先重命名第一个窗口：

```text
Ctrl+b，然后按 ,
```

把它命名为 `editor`。再新建一个窗口跑本地预览：

```text
Ctrl+b，然后按 c
```

命名为 `server`，然后执行：

```bash
hugo server -D --bind 0.0.0.0
```

如果需要看日志，再开一个 `logs` 窗口。临时对比配置时，可以在当前窗口里左右分屏。

离开服务器时：

```text
Ctrl+b，然后按 d
```

下次回来：

```bash
tmux attach -t blog
```

这个流程的好处是，工作现场不会因为一次 SSH 断线而消失。

## 常见坑

**第一，前缀键要先按再松开。** tmux 默认前缀键是 `Ctrl+b`。很多新手卡住，是因为把它当成组合键一直按着。

**第二，不要忘记给会话命名。** `tmux new -s name` 比直接输入 `tmux` 更清晰。

**第三，退出 pane 不等于结束 session。** 在 pane 里输入 `exit` 只会关闭当前 shell 或窗格。真正要杀掉整个会话，用 `tmux kill-session -t name`。

**第四，服务器重启后 tmux 会话也会消失。** tmux 能抵抗终端断开，但不能让进程跨系统重启继续存在。重要任务还是要配合 systemd、supervisor、nohup 或任务队列。

## 总结

tmux 不复杂，复杂的是它给了太多命令。

入门时只需要抓住一条主线：新建会话、分离会话、查看会话、接回会话。

```bash
tmux new -s work
tmux ls
tmux attach -t work
tmux kill-session -t work
```

等这几个命令变成肌肉记忆，再慢慢加上 window 和 pane。到那时，tmux 就不再是一个需要背快捷键的工具，而是你在服务器上保留工作现场的方式。
