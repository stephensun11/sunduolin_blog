# LLM Learning Path Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Audit the existing AiTech knowledge base, fill the most important gaps in an end-to-end LLM learning path with at least five rigorous articles, verify the rendered Hugo site, and publish the result to `origin/main`.

**Architecture:** Keep the existing Hugo taxonomy and article layout. Add focused long-form Markdown articles to established AiTech subcategories, introduce only the missing `LLM 对齐` and `LLM 评测` subcategories, and regenerate the tracked `public/` output so both Hugo deployment and direct static hosting receive the same content.

**Tech Stack:** Hugo 0.121.2, Goldmark Markdown, MathJax, Git.

---

### Task 1: Audit the current LLM curriculum

**Files:**
- Inspect: `content/aitech/*.md`
- Inspect: `hugo.toml`

**Step 1:** Inventory titles, subcategories, topics, and article depth.

**Step 2:** Map existing material to the sequence: foundations, architecture, pretraining, fine-tuning, alignment, evaluation, inference, and application engineering.

**Step 3:** Record the highest-impact gaps and select articles that do not duplicate existing content.

### Task 2: Fill training and architecture gaps

**Files:**
- Create: `content/aitech/pretraining-data-pipeline.md`
- Create: `content/aitech/llm-training-optimization.md`
- Create: `content/aitech/peft-lora-qlora.md`
- Create: `content/aitech/mixture-of-experts.md`

**Step 1:** Explain data collection, filtering, deduplication, mixture design, tokenization, packing, and contamination controls.

**Step 2:** Explain loss construction, AdamW, schedules, precision, gradient control, checkpointing, and stability diagnosis.

**Step 3:** Derive LoRA and explain QLoRA, adapter placement, memory accounting, merging, and evaluation.

**Step 4:** Explain sparse MoE routing, capacity, load balancing, communication cost, and serving trade-offs.

### Task 3: Fill post-training, evaluation, and deployment gaps

**Files:**
- Create: `content/aitech/post-training-alignment.md`
- Create: `content/aitech/llm-evaluation-methodology.md`
- Create: `content/aitech/llm-quantization-deployment.md`
- Create: `content/aitech/rag-system-design.md`

**Step 1:** Connect SFT, preference data, reward models, DPO, RLHF, and reasoning-oriented RL into one correct pipeline.

**Step 2:** Build a reproducible evaluation methodology covering capability, behavior, system metrics, contamination, and uncertainty.

**Step 3:** Explain weight and activation quantization, calibration, GPTQ/AWQ/SmoothQuant, KV-cache precision, and deployment validation.

**Step 4:** Explain retrieval, chunking, indexing, reranking, generation, citations, evaluation, and production failure modes.

### Task 4: Integrate navigation and learning order

**Files:**
- Modify: `hugo.toml`

**Step 1:** Add `LLM 对齐` and `LLM 评测` to the AiTech subcategory map and menu.

**Step 2:** Assign every new article a stable topic so the tertiary navigation is generated automatically.

**Step 3:** Confirm each taxonomy URL and breadcrumb resolves.

### Task 5: Build and verify

**Files:**
- Regenerate: `public/`

**Step 1:** Run `hugo --cleanDestinationDir` and require a successful build.

**Step 2:** Check all new generated pages, local links, fragment targets, raw Markdown/LaTeX leakage, and taxonomy membership.

**Step 3:** Serve `public/` locally and inspect representative desktop and mobile pages for TOC behavior, formulas, tables, and overflow.

### Task 6: Publish to main

**Files:**
- Commit all intentional source and generated changes.

**Step 1:** Fetch `origin` and verify the local branch is based on the latest `origin/main`.

**Step 2:** Review the final diff and commit with a scoped message.

**Step 3:** Push the verified commit to `origin/main` and confirm the remote branch points to it.
