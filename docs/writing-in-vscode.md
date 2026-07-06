# VSCode 写作工作流

这个博客可以直接在 VSCode 里完成新文章创建、预览和构建。

## 推荐入口

打开命令面板：

```text
Ctrl+Shift+P
```

选择：

```text
Tasks: Run Task
```

常用任务：

| 任务 | 用途 |
| --- | --- |
| `Blog: New Post` | 根据输入创建一篇 Markdown 草稿 |
| `Blog: Preview (drafts)` | 启动 Hugo 本地预览，包含草稿 |
| `Blog: Build public` | 重新生成 `public/` 静态目录 |

## 新建文章

运行 `Blog: New Post` 后，依次输入：

| 输入项 | 示例 |
| --- | --- |
| section | `cstech` |
| slug | `linux-process-notes` |
| title | `Linux 进程排查笔记` |
| subcategory | `Linux` |
| tags | `Linux,Shell,服务器` |

脚本会在对应目录生成 Markdown 文件，并自动填好 Hugo front matter。

## Snippets

在 Markdown 文件里输入下面前缀可以快速插入模板：

| snippet | 用途 |
| --- | --- |
| `hugo-cstech` | CSTech 文章 front matter |
| `hugo-ai` | AiTech / AiNews / PaperReading 文章 front matter |
| `codeblock` | Markdown 代码块 |

## 发布前检查

发布前建议检查三件事：

1. `draft` 已改为 `false`。
2. `categories` 和 `subcategories` 与 `hugo.toml` 中的配置一致。
3. 运行 `Blog: Build public` 后，确认 `public/` 中生成了对应页面。
