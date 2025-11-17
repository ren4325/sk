import csv
import os
import math

IN_CSV = "estimated_value.csv"

def angle_diff_deg(est, truth):
    """角度差を -180〜+180 に正規化"""
    d = est - truth
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


# CSV 読み込み
if not os.path.exists(IN_CSV):
    raise FileNotFoundError(f"{IN_CSV} が見つかりません")

with open(IN_CSV, newline="", encoding="utf-8") as f:
    reader = list(csv.reader(f))

current_set = None

for row in reader[2:]:
    if len(row) < 8:
        continue

    set_str, base_name, filename, idx_str, sx, sy, s_est_str, rot_est_str = row

    try:
        set_no  = int(set_str)
        idx     = int(idx_str)
        s_est   = float(s_est_str)
        rot_est = float(rot_est_str)
    except:
        continue

    # ---- Set 見出し ----
    if current_set != set_no:
        current_set = set_no
        print(f"\n========== Set {set_no} ==========")

    # ---- 真値 ---

    if idx == set_no - 2:
            s_true = 1 / (1.2 * 1.2)
            rot_true = 3.0 * 2.0       
    elif idx == set_no - 1:
            s_true = 1 / 1.2
            rot_true = 3.0
    elif idx == set_no:
            s_true = 1.0
            rot_true = 0.0
    elif idx == set_no + 1:
            s_true = 1.2
            rot_true = -3.0 * 1.0
    elif idx == set_no + 2:
            s_true = 1.2 * 1.2
            rot_true = -3.0 * 2.0


    # ---- 誤差 ----
    s_err = s_est - s_true
    rot_err = angle_diff_deg(rot_est, rot_true)

    # ---- 二乗誤差 ----
    s_sq  = s_err ** 2
    rot_sq = rot_err ** 2

    # ---- 二乗誤差のルート（絶対誤差）----
    s_abs  = abs(s_err)
    rot_abs = abs(rot_err)

    # ---- 出力 ----
    print(f"[index {idx}] {filename}")
    print(f"  scale: 真値={s_true:.6f} | 推定={s_est:.6f}")
    print(f"         誤差={s_err:.6f} | 二乗誤差={s_sq:.6f} | √二乗誤差={s_abs:.6f}")
    print(f"  rot  : 真値={rot_true:.3f}° | 推定={rot_est:.3f}°")
    print(f"         誤差={rot_err:.3f}° | 二乗誤差={rot_sq:.3f} | √二乗誤差={rot_abs:.3f}°")
