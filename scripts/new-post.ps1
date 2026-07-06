param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("ainews", "aitech", "cstech", "paperreading")]
    [string]$Section,

    [Parameter(Mandatory = $true)]
    [string]$Slug,

    [Parameter(Mandatory = $true)]
    [string]$Title,

    [string]$Subcategory = "",
    [string]$Tags = ""
)

$ErrorActionPreference = "Stop"

$categoryBySection = @{
    ainews       = "AiNews"
    aitech       = "AiTech"
    cstech       = "CSTech"
    paperreading = "PaperReading"
}

if ([string]::IsNullOrWhiteSpace($Subcategory)) {
    throw "Subcategory is required. Examples: Git, Python, Linux, tmux, NLP."
}

$safeSlug = $Slug.Trim().ToLowerInvariant() -replace "\s+", "-" -replace "[^a-z0-9\-_]", ""
if ([string]::IsNullOrWhiteSpace($safeSlug)) {
    throw "Slug must contain at least one letter, number, dash, or underscore."
}

$targetDir = Join-Path "content" $Section
$targetPath = Join-Path $targetDir "$safeSlug.md"

if (Test-Path -LiteralPath $targetPath) {
    throw "Article already exists: $targetPath"
}

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

$tagItems = @()
if (-not [string]::IsNullOrWhiteSpace($Tags)) {
    $tagItems = $Tags.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}

$tagText = ($tagItems | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }) -join ", "
$now = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
$category = $categoryBySection[$Section]

$content = @"
---
title: "$Title"
date: $now
draft: true
categories: ["$category"]
subcategories: ["$Subcategory"]
tags: [$tagText]
---

## Goal

What problem should this article solve?

## Draft

Start writing here.

## Summary

Wrap up the article in a few sentences.
"@

Set-Content -LiteralPath $targetPath -Value $content -Encoding UTF8
Write-Host "Created $targetPath"
