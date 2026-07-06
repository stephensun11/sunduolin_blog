---
title: "Git 实用入门：从提交到分支协作"
date: 2026-07-06T22:10:00+08:00
draft: false
categories: ["CSTech"]
subcategories: ["Git"]
tags: ["Git", "版本控制", "开发工具"]
---

Git 最重要的能力不是“保存代码”，而是让你敢于修改代码。

很多人刚开始学 Git，会被一堆命令吓住：`add`、`commit`、`branch`、`merge`、`rebase`、`stash`。其实入门时不需要把它们全部背下来。先理解 Git 在记录什么，再记住一条稳定的日常工作流，就能覆盖大多数开发场景。

## Git 在解决什么问题

写代码最怕两件事。

第一是不知道自己改了什么。今天改一点，明天改一点，三天后发现程序坏了，却不知道是哪一步引入的问题。

第二是不敢尝试。想重构一个函数，想换一种实现方式，但又担心改坏以后回不去。

Git 的价值就在这里：它把项目的每一次关键变化记录成一个提交。你可以查看差异，可以回到过去，也可以开一个分支大胆试验。

## 四个基本区域

理解 Git，先记住四个区域。

```text
working tree  工作区：你正在编辑的文件
staging area  暂存区：准备进入下一次提交的改动
repository    本地仓库：已经保存为 commit 的历史
remote        远程仓库：GitHub / GitLab 等远端副本
```

日常命令基本都围绕这四个区域移动文件状态。

查看当前状态：

```bash
git status
```

查看具体改动：

```bash
git diff
```

把改动放入暂存区：

```bash
git add file.md
git add .
```

提交到本地仓库：

```bash
git commit -m "Add tmux guide"
```

推送到远程仓库：

```bash
git push
```

## 第一次使用 Git

第一次使用时，先配置提交身份。

```bash
git config --global user.name "你的名字"
git config --global user.email "you@example.com"
```

克隆已有仓库：

```bash
git clone https://github.com/user/project.git
cd project
```

如果是一个本地新项目：

```bash
git init
git add .
git commit -m "Initial commit"
```

## 最常用的工作流

我推荐把日常工作固定成这几步。

```bash
# 1. 开始前先同步
git pull

# 2. 查看当前状态
git status

# 3. 编辑代码后查看差异
git diff

# 4. 暂存需要提交的文件
git add .

# 5. 提交
git commit -m "Describe the change"

# 6. 推送
git push
```

这条流程看起来普通，但能避免很多混乱。尤其是 `git status` 和 `git diff`，不要嫌它们啰嗦。真正救命的往往就是这两个命令。

## 分支的意义

分支不是高级功能，它是 Git 最好用的部分。

假设你要写一篇新文章，或者改一块不确定的代码，可以先开分支：

```bash
git switch -c feature/git-guide
```

在分支上提交完以后，切回主分支并合并：

```bash
git switch main
git merge feature/git-guide
```

推送新分支：

```bash
git push -u origin feature/git-guide
```

查看所有分支：

```bash
git branch
git branch -a
```

删除已经合并的本地分支：

```bash
git branch -d feature/git-guide
```

分支让你可以把“正在做的事”和“稳定版本”分开。这个习惯一旦建立，代码修改会轻松很多。

## 撤销和修正

Git 里最容易紧张的是“我改错了怎么办”。先记几个安全命令。

丢弃某个文件的未提交改动：

```bash
git restore file.md
```

取消暂存，但保留文件修改：

```bash
git restore --staged file.md
```

修改最近一次提交信息：

```bash
git commit --amend -m "Better commit message"
```

创建一个反向提交来撤销历史提交：

```bash
git revert commit_hash
```

`revert` 比 `reset --hard` 更适合已经推送到远程的提交，因为它不会强行改写公共历史。

## 写好提交信息

提交信息不需要文学性，但要让未来的自己看得懂。

坏例子：

```text
update
fix
change files
```

好一点：

```text
Add tmux practical guide
Rename cstech subcategory to tmux
Fix local static links
```

提交信息最好回答一个问题：这次提交改变了什么？

## 常用命令速查

```bash
# 查看状态和历史
git status
git log --oneline --decorate -5

# 查看差异
git diff
git diff --staged

# 暂存与提交
git add .
git commit -m "Message"

# 分支
git switch -c feature/name
git switch main
git merge feature/name

# 远程
git pull
git push
git fetch origin
```

## 总结

Git 入门不应该从背命令开始，而应该从建立安全感开始。

先用好 `status`、`diff`、`add`、`commit`、`push`，再用分支隔离不同任务。等这些命令变成习惯以后，再慢慢学习 `rebase`、`stash`、`cherry-pick` 也不迟。

真正的目标不是“会 Git”，而是让你改代码时知道自己站在哪里，也知道怎么回来。
