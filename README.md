# BlockBlast AI — Bài tập lớn Nhập môn Trí tuệ Nhân tạo

## Thông tin môn học

|                      |                                            |
| -------------------- | ------------------------------------------ |
| **Tên môn học**      | Nhập môn Trí tuệ Nhân tạo                  |
| **Mã môn học**       | CO3061                                     |
| **Học kỳ / Năm học** | Học kỳ II / 2025 – 2026 (CQ_HK252)         |
| **Lớp**              | A01                                        |
| **Trường**           | Đại học Bách Khoa, Đại học Quốc gia TP.HCM |
| **Khoa**             | Khoa học & Kỹ thuật Máy tính               |
| **GVHD**             | TS. Trương Vĩnh Lân                        |

## Thành viên nhóm

| STT | Họ và tên       | MSSV    | Email                          |
| --- | --------------- | ------- | ------------------------------ |
| 1   | Phạm Ngọc Long  | 2211894 | long.phamngoc2607@hcmut.edu.vn |
| 2   | Nguyễn Ngọc Duy | 2210522 |                                |
| 3   | Võ Quốc Phong   | 2352913 |                                |
| 4   | Nguyễn Tấn Đạt  | 2352234 |                                |

## Mục tiêu bài tập lớn

Vận dụng kiến thức của môn Nhập môn Trí tuệ Nhân tạo để xây dựng một hệ thống AI hoàn chỉnh giải bài toán **Block Placement (Block Blast)** trên bàn cờ 8×8: ở mỗi lượt, hệ thống nhận 4 khối Tetrimino và phải tìm thứ tự / vị trí đặt sao cho tối đa hoá số ô bị xoá khi hoàn thành hàng hoặc cột. Hệ thống tích hợp đầy đủ năm thành phần (modules) theo yêu cầu của đề bài:

- **Module A — Biểu diễn & Tìm kiếm.** Mô hình hoá `(State, Action, Goal, Cost)` và áp dụng **Best-first Search** cho từng lượt.
- **Module B — Heuristic.** Hàm chi phí `Cost = w₁·ΔHoles + w₂·ΔArea − w₃·ΔLinesCleared + K` và đánh giá node `Eval(n) = g(n) + λ·|remaining_blocks|`.
- **Module C — Biểu diễn & Suy luận tri thức.** Lớp tri thức bằng **logic vị từ** (`Empty`, `Remain`, `Placeable`, `Covers`, `Line`, `OneAway`, `Fillable`, `Trapped`, `Completes`) tích hợp trực tiếp vào search: cắt nhánh các trạng thái có ô bị kẹt và cộng bonus cho action hoàn thành hàng/cột gần đầy.
- **Module D — Mạng Bayes / Xác suất.** **Bayesian Network** với hai CPD `P(ValidMoveLevel | Density, Fragmentation, LineClear)` và `P(StuckRisk | ValidMoveLevel, Fragmentation)`, ước lượng bằng tần suất trên dataset rollout (không smoothing).
- **Module E — Học máy.** **Decision Tree** + **Random Forest** dự đoán chất lượng action `(state, action)` dựa trên teacher-signal sinh từ Best-first Search, sau đó dẫn dắt search bằng `best_first_search_ml`.

## Liên kết

|                     |                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------- |
| **Colab notebook**  | <https://colab.research.google.com/drive/1-7p8F6Q2b_-ekDuZOjuHKjkqIsp6AZJb?usp=sharing> |
| **Báo cáo (Prism)** | <https://prism.openai.com/?u=775e7ad3-fc69-4c37-8f4c-107910d359d6&pg=1&m=main.tex&d=7>  |

## Cấu trúc thư mục

```
BlockBlast-AI-Project/
├── README.md                       // tài liệu này
├── requirements.txt                // các thư viện Python cần thiết
├── main.py                         // CLI runner: demo / multi-turn / bayes
├── .gitignore
├── notebooks/
│   └── BlockBlast.ipynb            // notebook hoàn chỉnh, chạy Run-all được
├── modules/                        // mã nguồn được tách theo module
│   ├── __init__.py                 // gom các symbol thường dùng
│   ├── blocks.py                   // Section 1: Tetrimino + rotations (TETRIMINOS)
│   ├── board.py                    // Section 2.1: clear_full_lines, components, bbox
│   ├── state.py                    // Section 2.2: State, Node
│   ├── search.py                   // Section 2.3 + 3 + 4: best_first_search (KR layer baked in)
│   ├── kr_layer.py                 // Module C — Chương 4: predicates
│   ├── ml_features.py              // Module E (data side): teacher-signal labelling
│   ├── ml_search.py                // Module E (runtime side): best_first_search_ml
│   └── bayes_risk.py               // Module D — Chương 5: Bayes Network rollout
├── reports/
│   └── README.md                   // chỗ để bỏ report.pdf
└── features/
    └── README.md                   // chỗ để bỏ dataset.csv, blockblast_model.pkl, outputs/bayes/*.csv
```

Notebook trong `notebooks/` là **toàn bộ** hệ thống trong một file duy nhất (Colab Run-all được). Thư mục `modules/` là phiên bản tách module để tái sử dụng / kiểm thử / import từ Python script khác — nội dung trùng với notebook nhưng được tổ chức theo trách nhiệm rõ ràng.

## Hướng dẫn chạy

### Tuỳ chọn 1 — Chạy trên Google Colab (khuyến nghị, đúng yêu cầu đề)

1. Mở Colab notebook: <https://colab.research.google.com/drive/1-7p8F6Q2b_-ekDuZOjuHKjkqIsp6AZJb?usp=sharing>.
2. `Runtime -> Run all`. Notebook tự cài thư viện cần thiết và chạy đầy đủ năm modules từ trên xuống.
3. Không cần mount Google Drive. Mọi dữ liệu được sinh ra trong môi trường Colab.

### Tuỳ chọn 2 — Chạy local (Jupyter)

```bash
# 1. Clone repo và chuyển vào thư mục
git clone <repo-url>
cd BlockBlast-AI-Project

# 2. Tạo môi trường ảo và cài thư viện
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt

# 3. Khởi động Jupyter và mở notebooks/BlockBlast.ipynb
jupyter notebook notebooks/BlockBlast.ipynb
```

Sau đó chạy `Cell → Run All`. Một số lưu ý về thời gian chạy:

- **Module D (Bayes)** mặc định sinh `n_samples=50_000` rollout — mất vài phút trên CPU phổ thông. Nếu chỉ muốn smoke-test, sửa `n_samples=2_000` ở cuối cell Module D.
- **Module E (ML)** mặc định `N_STATES=500, TEACHER_BUDGET=1000` — cũng vài phút. Sau lần chạy đầu sẽ ghi `dataset.csv` và `blockblast_model.pkl`, lần sau có thể bỏ qua bước sinh dataset bằng cách `joblib.load('blockblast_model.pkl')`.

### Tuỳ chọn 3 — Chạy nhanh từ CLI bằng `main.py`

```bash
python main.py demo                    # one-turn KR ablation (~30 s)
python main.py multi-turn              # 3-turn fixed game, baseline search
python main.py bayes --n-samples 2000  # Bayes rollout experiment (~1 phút)
python main.py --help                  # liệt kê subcommand
```

`main.py` chỉ là wrapper gọi vào `modules/`; nó không thay thế notebook.

### Tuỳ chọn 4 — Sử dụng modules từ Python script

```python
import numpy as np
from modules import State, best_first_search, print_solution

initial = State(
    board=np.zeros((8, 8), dtype=int),
    available_blocks=['O', 'I_0', 'T_180', 'L_90'],
    current_score=0,
)
result = best_first_search(initial, max_expansions=1000)
print_solution(result)
```

Để chạy ablation **không** dùng lớp tri thức (so với Module C):

```python
result_baseline = best_first_search(
    initial, max_expansions=1000,
    prune_trapped=False, completion_bonus=0.0,
)
```

## Yêu cầu thư viện

Đã liệt kê trong `requirements.txt`. Tóm tắt:

- `numpy` — biểu diễn bàn cờ và phép toán vector hoá.
- `pandas` — DataFrame cho dataset Module D / E.
- `scikit-learn` — `DecisionTreeClassifier`, `RandomForestClassifier`, các metric, `StratifiedKFold` cross-validation.
- `matplotlib` — biểu đồ confusion matrix, feature importance, F1 per fold.
- `ipywidgets` — slider hiển thị từng bước đặt block trong cell visualization cuối notebook.
- `joblib` — lưu và nạp lại model đã train.

## Dataset

Toàn bộ dữ liệu (training set Module E + Module D rollout) **được sinh trực tiếp trong notebook** — không có dataset ngoài. Các file kết quả trung gian (`dataset.csv`, `blockblast_model.pkl`, `outputs/bayes/*.csv`) được sinh ra khi chạy notebook và được liệt kê trong `.gitignore`. Khi nộp bài, đặt vào `features/` nếu giảng viên muốn xem feature đã trích xuất.

## Phân công công việc

Tham khảo Phụ lục A của báo cáo PDF.
