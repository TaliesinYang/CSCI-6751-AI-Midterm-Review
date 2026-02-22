# CSCI 6751 — Artificial Intelligence | Midterm Review Kit

FDU 研究生 AI 课程复习资料。包含 5 周课堂笔记、往年真题（含解答）、公式速查表、AI 辅助复习提示词。

## Quick Start — 怎么用这个 repo 复习

1. 打开 `generated/AI-Midterm-Study-Plan.md` — 里面有完整的复习计划、文件索引、和 7 个 AI 提示词
2. 按计划顺序刷真题：GD → Fuzzy → MCQ → 思维导图扫盲
3. 复制提示词喂给任何 AI（Claude / ChatGPT / Copilot / Codex），AI 会用苏格拉底式提问帮你练习

## Exam Info

| Item | Detail |
|------|--------|
| Course | CSCI 6751 V1 — Artificial Intelligence |
| Midterm | Feb 24, 2026 |
| Format | Computation 50pts (50min) + MCQ 50pts (15min) |
| Coverage | Week01–Week05 |

### Past Exam Pattern (stable across semesters)

- **Q1 (25pts)**: Gradient Descent — multivariate + L2 regularization, hand-compute 1 iteration
- **Q2 (25pts)**: Fuzzy Logic — trapezoidal membership functions, multi-input, firing strengths + centroid defuzzification
- **MCQ (50pts, 10 questions)**: GD vs Normal Equation, L1 vs L2, matrix dimensions, overfitting/underfitting, evaluation metrics

## Repo Structure

```
.
├── README.md                          ← you are here
├── index.yaml                         ← course config (weeks, topics, file mappings)
│
├── generated/                         ← AI-generated study materials
│   ├── AI-Midterm-Study-Plan.md       ← ⭐ START HERE: plan + file index + AI prompts
│   ├── CSCI-6751-Midterm-Review.canvas ← ⭐ exam-focused mind map (Obsidian Canvas)
│   ├── CSCI-6751-Knowledge-Map.canvas  ← full course knowledge map
│   │
│   └── Week01–05/                     ← per-week generated notes
│       ├── notes.md                   ← detailed lecture notes
│       ├── Reminders.md               ← exam-focused review checklist
│       ├── qa-summary.md              ← classroom Q&A summary
│       ├── vocabulary.md              ← key terms (EN/CN bilingual)
│       ├── WeekXX-MindMap.md          ← topic mind map (markdown)
│       ├── WeekXX-Canvas.canvas       ← topic mind map (Obsidian Canvas)
│       └── transcripts/               ← lecture transcript text files
│
├── materials/                         ← original course materials
│   ├── slides/                        ← lecture slides (PDF/PPTX) + Jupyter notebooks
│   ├── handouts/                      ← syllabus + Python fundamentals notebook
│   └── references/                    ← ⭐ exam prep resources
│       ├── Past-Exam-Questions.pdf    ← past exam compilation with solutions
│       ├── Exam-Formula-Cheatsheet.pdf ← formula quick reference for exam
│       ├── Fuzzy-Logic-Tutorial.pdf   ← fuzzy logic detailed tutorial
│       ├── Gradient-Descent-Tutorial.pdf
│       ├── Linear-Algebra-Basics.pdf
│       ├── Matrix-Inverse-Practice.pdf
│       └── past-exams/               ← ⭐ complete past exams with answer keys
│           ├── Midterm-*-Solutions.pdf (computation, 2 groups)
│           ├── Midterm-MCQ-*-Solutions.pdf (MCQ, 2 groups)
│           ├── Quiz-*-Solutions.pdf (Quiz#1, 2 groups)
│           └── Quiz#2-*-Solutions.pdf (Quiz#2, 2 groups)
│
└── .gitignore
```

## Week-by-Week Topics

| Week | Date | Topic | Key Exam Content |
|------|------|-------|-----------------|
| 01 | Jan 14 | AI Introduction | Turing Test, Expert Systems, **Fuzzy Logic** (trimf, trapmf, centroid) |
| 02 | Jan 20 | ML Fundamentals | Supervised vs Unsupervised, Regression vs Classification, Encoding |
| 03 | Jan 27 | Linear Regression | **Normal Equation** θ=(XᵀX)⁻¹Xᵀy, **Gradient Descent** 6-step hand calc |
| 04 | Feb 03 | Overfitting & Regularization | Bias/Variance, **L1 (Lasso) vs L2 (Ridge)**, Lambda tuning |
| 05 | Feb 10 | Evaluation Metrics | MAE/MSE/RMSE/R², **Confusion Matrix**, Precision/Recall/F1, Cross-Validation |

## For AI Agents — Context Prompt

> If you are an AI agent helping a student review, start by reading `generated/AI-Midterm-Study-Plan.md`. It contains the complete study plan, file paths for all materials, and 7 copy-paste prompts for Socratic-style review sessions. The most important exam topics are Gradient Descent (Q1, 25pts) and Fuzzy Logic (Q2, 25pts) — these appear every semester.

## Key Formulas (quick reference)

```
Gradient Descent:     θ_new = θ_old - η · ∂J/∂θ
Normal Equation:      θ = (XᵀX)⁻¹Xᵀy
L2 Regularized Cost:  J = (1/n)Σ(ŷ-y)² + λΣθⱼ²
L2 Gradient:          ∂J/∂θⱼ = (2/n)Σ(ŷᵢ-yᵢ)·xⱼᵢ + 2λθⱼ
2×2 Matrix Inverse:   A⁻¹ = (1/(ad-bc)) × [d,-b;-c,a]
Trapezoidal MF:       trapmf(a,b,c,d): 0 / (x-a)/(b-a) / 1 / (d-x)/(d-c) / 0
Centroid Defuzz:      Output = Σ(FS×Output) / ΣFS
Precision:            TP / (TP + FP)
Recall:               TP / (TP + FN)
F1:                   2 × P × R / (P + R)
```

## License

Educational use only. Course materials belong to FDU and the instructor.
