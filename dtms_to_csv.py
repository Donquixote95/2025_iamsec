# dtms_to_csv.py ─ '주' 헤더만 사용 + 세로 병합셀 처리 + 여단(unit) 배제
# ▶ 단일 라벨 vs 공통(병합) 라벨 자동 판정: IoU 기반
import argparse, re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pdfplumber, pandas as pd
from collections import defaultdict

# ===== 활동 라벨 동의어 =====
ACTIVITY_ALIASES: Dict[str, str] = {
    "전평": "전술훈련평가", "전측": "전투력측정",
    "마일": "마일즈훈련", "마일즈": "마일즈훈련",
    "동원": "동원훈련", "WMD": "WMD대응훈련",
    "유격": "유격훈련", "포탄": "포탄사격훈련",
    "도시": "도시지역작전훈련", "도시지역작전훈련": "도시지역작전훈련",
    "부임초기지휘통제기구훈련": "부임초기지휘통제기구훈련",
    "혹한기훈련": "혹한기훈련",
    "공용화기집체교육": "공용화기집체교육",
    "전투지휘검열": "전투지휘검열",
    "전투진지공사": "전투진지공사",
    "재입영": "동원훈련(재입영)",
}

# ===== 파라미터 =====
# (1) 열 스냅/세로 병합 관련
X_GAP = 12             # 같은 '주' 열로 간주할 x 허용치
X_COLUMN_WINDOW = 60   # 주 열 근처 후보 허용 범위
Y_MERGE_GAP = 8        # 세로 병합 시 y 간격 허용

# (2) 행 배정(단일 vs 공통) 판정용
SINGLE_ROW_FACTOR = 1.5
MIN_IOU_SINGLE    = 0.45
MIN_IOU_MULTI     = 0.20   # ⬅︎ 0.25 → 0.20 로 완화
MIN_ROWS_FOR_MULTI = 2

MULTI_EDGE_PAD_FACTOR = 0.35  # ⬅︎ 공통 라벨일 때, 행 높이의 35%만큼 위/아래 패딩

@dataclass
class Token:
    text: str; x0: float; x1: float; top: float; bottom: float; page: int
    @property
    def xc(self): return (self.x0 + self.x1)/2
    @property
    def yc(self): return (self.top + self.bottom)/2

@dataclass
class RowWindow:
    unit: str; page: int; y_top: float; y_bottom: float

@dataclass
class WeekAnchor:
    week: int; page: int; x_center: float

@dataclass
class ActivityCluster:
    page: int; week: int; x_center: float
    y_top: float; y_bottom: float
    text: str

# ===== 유틸 =====
def load_tokens(pdf_path: str) -> List[Token]:
    toks: List[Token] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pagenum, page in enumerate(pdf.pages, start=1):
            for w in page.extract_words(use_text_flow=True, keep_blank_chars=False) or []:
                txt = w["text"].strip()
                if not txt: continue
                toks.append(Token(txt, w["x0"], w["x1"], w["top"], w["bottom"], pagenum))
    return toks

def is_brigade(text: str) -> bool:
    return bool(re.search(r"\b\S+여단\b", text))

def normalize_unit_text(t: str) -> Optional[str]:
    t = t.strip()
    if "여단" in t: return None
    if re.search(r"\b\d+\s*대대\b", t):
        num = re.search(r"\d+", t).group(0)
        return f"{num}대대"
    if "포병대대" in t: return "포병대대"
    if "군수지원대대" in t: return "군수지원대대"
    if re.search(r"본부/?\s*직할", t): return "본부/직할"
    return None

def normalize_activity_text(t: str) -> Optional[str]:
    raw  = " ".join(t.split())
    join = raw.replace(" ", "")
    if raw in ACTIVITY_ALIASES:  return ACTIVITY_ALIASES[raw]
    if join in ACTIVITY_ALIASES: return ACTIVITY_ALIASES[join]
    for k, v in ACTIVITY_ALIASES.items():
        if k in raw or k in join:
            return v
    return None

# ===== (1) '주' 헤더만 읽어서 주 앵커 생성 =====
def find_week_anchors(tokens: List[Token]) -> List[WeekAnchor]:
    by_page_band: Dict[Tuple[int,int], List[Token]] = {}
    for t in tokens:
        band = int(t.top // 6)
        by_page_band.setdefault((t.page, band), []).append(t)

    def band_with_label(label: str, page: int) -> Optional[int]:
        for (pg, band), toks in by_page_band.items():
            if pg != page: continue
            if any(x.text == label for x in toks):
                return band
        return None

    anchors: List[WeekAnchor] = []
    for pg in sorted(set(t.page for t in tokens)):
        band_week = band_with_label("주", pg)
        if band_week is None: continue
        week_toks = [t for t in by_page_band[(pg, band_week)] if t.text.isdigit()]
        for t in sorted(week_toks, key=lambda x: x.xc):
            w = int(t.text)
            if w > 0:
                anchors.append(WeekAnchor(week=w, page=pg, x_center=t.xc))
    return anchors

# ===== (2) 부대별 행 경계(RowWindow) =====
def build_row_windows(tokens: List[Token]) -> List[RowWindow]:
    xs_all = [t.xc for t in tokens]
    if not xs_all: return []
    xs = sorted(xs_all)
    x_cut = xs[max(0, int(len(xs)*0.35) - 1)]  # 좌측 35%를 부대명 영역으로 가정

    # 부대명 토큰 수집
    marks: Dict[int, List[Tuple[str, Token]]] = {}
    for t in tokens:
        u = normalize_unit_text(t.text)
        if u and t.xc <= x_cut:
            marks.setdefault(t.page, []).append((u, t))

    windows: List[RowWindow] = []
    for pg, pairs in marks.items():
        if not pairs: continue
        pairs.sort(key=lambda p: p[1].yc)
        # 페이지별 행 간 간격의 중앙값으로 높이 추정
        ys = [p[1].yc for p in pairs]
        deltas = [ys[i+1]-ys[i] for i in range(len(ys)-1)]
        row_h = (sorted(deltas)[len(deltas)//2] if deltas else 24)  # fallback 24pt
        half = row_h/2.0

        for _, m in pairs:
            windows.append(RowWindow(
                unit=normalize_unit_text(m.text) or m.text,
                page=pg,
                y_top=m.yc - half,
                y_bottom=m.yc + half
            ))
    return windows

# ===== (3) 활동 클러스터링: 같은 주 열에서 세로로 이어진 텍스트 묶기 =====
def build_activity_clusters(tokens: List[Token], anchors: List[WeekAnchor]) -> List[ActivityCluster]:
    clusters: List[ActivityCluster] = []
    for pg in sorted(set(t.page for t in tokens)):
        a_on_page = [a for a in anchors if a.page == pg]
        if not a_on_page: continue
        toks_on_page = [t for t in tokens if t.page == pg]

        # 페이지 기준 글자 높이 중앙값 기반 y 병합 간격
        heights = [abs(t.bottom - t.top) for t in toks_on_page]
        y_gap = max(6, (sorted(heights)[len(heights)//2] if heights else 6))

        def is_candidate(tok: Token) -> bool:
            txt = tok.text.strip()
            if not txt or txt in {"주","월","일"}: return False
            if txt.isdigit(): return False
            if normalize_unit_text(txt): return False
            if is_brigade(txt): return False
            dmin = min(abs(tok.xc - a.x_center) for a in a_on_page)
            if dmin > X_COLUMN_WINDOW: return False
            return bool(re.search(r"[가-힣A-Za-z]", txt))

        cand = [t for t in toks_on_page if is_candidate(t)]
        if not cand: continue

        def closest_idx(xc: float) -> int:
            d = [abs(xc - a.x_center) for a in a_on_page]
            i = min(range(len(d)), key=lambda k: d[k])
            return i if d[i] <= X_COLUMN_WINDOW else -1

        enriched = [(closest_idx(t.xc), t) for t in cand]
        enriched = [z for z in enriched if z[0] != -1]
        enriched.sort(key=lambda z: (z[0], z[1].yc))

        for idx, t in enriched:
            anchor = a_on_page[idx]
            merged = False
            for c in clusters:
                if c.page != pg or c.week != anchor.week: continue
                if abs(c.x_center - anchor.x_center) > X_GAP: continue
                # y 겹침/근접 판정
                overlap = not (t.bottom < c.y_top or t.top > c.y_bottom)
                touch   = (t.top <= c.y_bottom + Y_MERGE_GAP) and (t.top >= c.y_top - 2*Y_MERGE_GAP)
                if overlap or touch:
                    c.y_top = min(c.y_top, t.top)
                    c.y_bottom = max(c.y_bottom, t.bottom)
                    c.text += " " + t.text
                    merged = True
                    break
            if not merged:
                clusters.append(ActivityCluster(
                    page=pg, week=a_on_page[idx].week, x_center=a_on_page[idx].x_center,
                    y_top=t.top, y_bottom=t.bottom, text=t.text
                ))
    return clusters

# ===== (4) IoU 기반 단일/공통 배정 =====
def extract(tokens: List[Token], anchors: List[WeekAnchor], brigade_hint: Optional[str]) -> pd.DataFrame:
    toks_sorted = sorted(tokens, key=lambda t: (t.page, t.top, t.x0))
    brigade = brigade_hint or next((t.text for t in toks_sorted if is_brigade(t.text)), None)

    windows = build_row_windows(toks_sorted)
    clusters = build_activity_clusters(toks_sorted, anchors)

    # 페이지별 median row height
    from collections import defaultdict
    rows_by_page: Dict[int, List[RowWindow]] = defaultdict(list)
    for rw in windows:
        rows_by_page[rw.page].append(rw)
    med_row_h: Dict[int, float] = {}
    for pg, rws in rows_by_page.items():
        hs = [rw.y_bottom - rw.y_top for rw in rws]
        hs.sort()
        med_row_h[pg] = hs[len(hs)//2] if hs else 24.0

    # IoU 계산 함수 (단일/공통 분리)
    def iou_with_row_strict(c_top: float, c_bot: float, rw: RowWindow) -> float:
        inter = max(0.0, min(c_bot, rw.y_bottom) - max(c_top, rw.y_top))
        denom = (rw.y_bottom - rw.y_top)
        return inter / denom if denom > 0 else 0.0

    def iou_with_row_padded(c_top: float, c_bot: float, rw: RowWindow, pad: float) -> float:
        # 공통(멀티) 라벨일 때만 행창에 상하 패딩 적용
        y0 = rw.y_top - pad
        y1 = rw.y_bottom + pad
        inter = max(0.0, min(c_bot, y1) - max(c_top, y0))
        denom = (rw.y_bottom - rw.y_top)
        return inter / denom if denom > 0 else 0.0

    rows = []
    for c in clusters:
        label_raw  = " ".join(c.text.split())
        label_join = label_raw.replace(" ", "")
        act = normalize_activity_text(label_raw) or normalize_activity_text(label_join)
        if not act:
            continue

        page_rows = rows_by_page.get(c.page, [])
        if not page_rows:
            continue

        # 단일/공통 판정
        h_row_med = med_row_h.get(c.page, 24.0)
        h_cluster = c.y_bottom - c.y_top

        # 1) 엄격 IoU로 각 행과 겹침 계산
        overlaps_strict = [(rw, iou_with_row_strict(c.y_top, c.y_bottom, rw)) for rw in page_rows]
        overlaps_strict.sort(key=lambda z: z[1], reverse=True)

        overlapped_rows_strict = [(rw, r) for rw, r in overlaps_strict if r >= MIN_IOU_MULTI]
        is_multi_by_height = (h_cluster > SINGLE_ROW_FACTOR * h_row_med)
        is_multi_by_rows   = (len(overlapped_rows_strict) >= MIN_ROWS_FOR_MULTI)
        is_multi = is_multi_by_height or is_multi_by_rows

        if not is_multi:
            # ▶ 단일 라벨: 가장 큰 1개 행에만 (엄격 IoU)
            if overlaps_strict and overlaps_strict[0][1] >= MIN_IOU_SINGLE:
                best_rw = overlaps_strict[0][0]
                rows.append({"brigade": brigade, "unit": best_rw.unit, "activity": act, "week": int(c.week)})
        else:
            # ▶ 공통 라벨: 가장 위/아래 행이 살짝 비껴가는 문제 보정 → 패딩 IoU로 재평가
            # 행 높이 기반 패딩(가장 보수적으로 median 사용)
            pad = max(6.0, MULTI_EDGE_PAD_FACTOR * h_row_med)

            overlaps_padded = [(rw, iou_with_row_padded(c.y_top, c.y_bottom, rw, pad)) for rw in page_rows]
            overlaps_padded.sort(key=lambda z: z[1], reverse=True)

            for rw, r in overlaps_padded:
                if r >= MIN_IOU_MULTI:
                    rows.append({"brigade": brigade, "unit": rw.unit, "activity": act, "week": int(c.week)})

    df = pd.DataFrame(rows).drop_duplicates(subset=["brigade","unit","activity","week"]).reset_index(drop=True)
    cols = ["brigade","unit","activity","week"]
    for col in cols:
        if col not in df.columns:
            df[col] = None
    return df[cols].sort_values(["unit","week","activity"]).reset_index(drop=True)

# ===== (5) 실행 =====
def run(input_pdf: str, output_csv: str, brigade_hint: Optional[str]) -> None:
    tokens  = load_tokens(input_pdf)
    anchors = find_week_anchors(tokens)
    if not anchors:
        print("ERROR: '주' 헤더를 찾지 못했습니다."); return
    df = extract(tokens, anchors, brigade_hint)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_csv} (rows={len(df)})")

def main():
    ap = argparse.ArgumentParser(description="DTMS-like PDF → CSV ('주' 헤더, 세로 병합셀, 여단 제외, IoU 배정)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--brigade", type=str, default=None)
    args = ap.parse_args()
    run(args.input, args.output, args.brigade)

if __name__ == "__main__":
    main()