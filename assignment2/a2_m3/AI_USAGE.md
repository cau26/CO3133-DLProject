# AI Usage Disclosure — Assignment 2 (M3 Final)

**Course:** CO3133 - Deep Learning and Its Applications (Semester 261)  
**Group:** G-M10

| Item | Record |
|---|---|
| Tool / user | Google Antigravity, ChatGPT, Claude / Trần Gia Lâm (2352670), Nguyễn Hữu Cầu (2352129) |
| Role of AI | Assisted in extending the modular `a2/` pipeline for Official VOC2007 Test Set evaluation, implementing the Train/Val-vs-Test overlap audit, packaging the Colab execution notebook, and compiling the M3 final report, slides, and video script from real Colab outputs. |
| Scope | Extended `a2/data.py`, `a2/train.py`, `a2/report.py`, `run_a2.py`, and `configs/a2.json` to support `VOCtest_06-Nov-2007.tar`; integrated verified Val and Official Test metrics into `assignment2-final.md`, `output/latex/report.tex`, `G-M10_A2_Report.pdf`, `G-M10_A2_Slides.pdf`, and `VIDEO_SCRIPT_VI.md`. |
| Experiment review | Verified that checkpoints selected strictly on Validation (`best.pt`: `ssdlite320` ep 17, `fasterrcnn_frozen` ep 11, `fasterrcnn_finetune` ep 12) are evaluated once on Official Test (`test.txt`, `4,952` images) without hyperparameter retuning, and `2,944` `difficult=1` test boxes are ignored per VOC2007 protocol. |
| Evidence used | All reported numbers come strictly from `outputs/a2_m2_a100/` (A100 pilots/M2) and `outputs/a2_m3_a100/` (L4 Val + Official Test: `test_comparison.md`, `val_vs_test_comparison.csv`, `test_overlap_check.json`, `*/test_metrics.json`, `*/test_error_taxonomy.json`). |
| Reproducibility | Recorded `seed = 42`, `source_sha256 = ec1ffd3db1b5f01da4d4319c61527148e157e73e6023be295c1fc8febd806d3f`, `split_sha256 = c60294fe4e914b27399ff065c9e246dcd3428440307c97cebf2febe139c2bd6c`, executed notebook `notebooks/A2_M3_Colab_Run.ipynb`, and Google Drive checkpoint backup. |
| Representative prompts | "Extend a2_m2 into a2_m3 with single-shot Official VOC2007 Test evaluation and Train/Test overlap check"; "Integrate real Colab L4 Official Test Set results into the HCMUT LaTeX report, slides, and Vietnamese video script without inventing numbers." |
| Team verification | Nguyễn Hữu Cầu executed `notebooks/A2_M3_Colab_Run.ipynb` on Google Colab NVIDIA L4, archived checkpoints on Google Drive, and verified all Official Test Set numbers; Trần Gia Lâm verified the methodology, controlled study interpretation, and final report consistency. |
| Final responsibility | Group G-M10 takes full responsibility for all source code, experimental execution, quantitative results, and submitted deliverables. |
