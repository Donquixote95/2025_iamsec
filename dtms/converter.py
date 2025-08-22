# dtms/converter.py
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict
from io import BytesIO

import pandas as pd
import pdfplumber

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
X_GAP = 12
X_COLUMN_WINDOW = 60
Y_MERGE_GAP = 8

SINGLE_ROW_FACTOR = 1.5
MIN_IOU_SINGLE    = 0.45
MIN_IOU_MULTI     = 0.20
MIN_ROWS_FOR_MULTI = 2
MULTI_EDGE_PAD_FACTOR = 0.35

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

def _is_brigade(text: str) -> bool:
    return bool(re.search(r"\b\S+여단\b", text))

def _normalize_unit_text(t: str) -> Optional[str]:
    t = t.strip()
    if "여단" in t: return None
    if re.search(r"\b\d+\s*대대\b", t):
        num = re.search(r"\d+", t).group(0)
        return f"{num}대대"
    if "포병대대" in t: return "포병대대"
    if "군수지원대대" in t: return "군수지원대대"
    if re.search(r"본부/?\s*직할", t): return "본부/직할"
    return None

def _normalize_activity_text(t: str) -> Optional[str]:
    raw  = " ".join(t.split())
    join = raw.replace(" ", "")
    if raw in ACTIVITY_ALIASES:  return ACTIVITY_ALIASES[raw]
    if join in ACTIVITY_ALIASES: return ACTIVITY_ALIASES[join]
    for k, v in ACTIVITY_ALIASES.items():
        if k in raw or k in join: return v
    return None

def _load_tokens(pdf: Union[str, BytesIO]) -> List[Token]:
    toks: List[Token] = []
    with pdfplumber.open(pdf) as pdfobj:
        for pagenum, page in enumerate(pdfobj.pages, start=1):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
            for w in words:
                txt = w.get("text", "").strip()
                if not txt: continue
                toks.append(Token(txt, w["x0"], w["x1"], w["top"], w["bottom"], pagenum))
    return toks

def _find_week_anchors(tokens: List[Token]) -> List[WeekAnchor]:
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

def _build_row_windows(tokens: List[Token]) -> List[RowWindow]:
    xs_all = [t.xc for t in tokens]
    if not xs_all: return []
    xs = sorted(xs_all)
    x_cut = xs[max(0, int(len(xs)*0.35) - 1)]
    marks: Dict[int, List[Tuple[str, Token]]] = {}
    for t in tokens:
        u = _normalize_unit_text(t.text)
        if u and t.xc <= x_cut:
            marks.setdefault(t.page, []).append((u, t))
    windows: List[RowWindow] = []
    for pg, pairs in marks.items():
        if not pairs: continue
        pairs.sort(key=lambda p: p[1].yc)
        ys = [p[1].yc for p in pairs]
        deltas = [ys[i+1]-ys[i] for i in range(len(ys)-1)]
        row_h = (sorted(deltas)[len(deltas)//2] if deltas else 24)
        half = row_h/2.0
        for _, m in pairs:
            windows.append(RowWindow(
                unit=_normalize_unit_text(m.text) or m.text,
                page=pg, y_top=m.yc - half, y_bottom=m.yc + half
            ))
    return windows

def _build_activity_clusters(tokens: List[Token], anchors: List[WeekAnchor]) -> List[ActivityCluster]:
    clusters: List[ActivityCluster] = []
    for pg in sorted(set(t.page for t in tokens)):
        a_on_page = [a for a in anchors if a.page == pg]
        if not a_on_page: continue
        toks_on_page = [t for t in tokens if t.page == pg]
        heights = [abs(t.bottom - t.top) for t in toks_on_page]
        y_gap = max(6, (sorted(heights)[len(heights)//2] if heights else 6))
        def is_candidate(tok: Token) -> bool:
            txt = tok.text.strip()
            if not txt or txt in {"주","월","일"}: return False
            if txt.isdigit(): return False
            if _normalize_unit_text(txt): return False
            if _is_brigade(txt): return False
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
                    page=pg, week=anchor.week, x_center=anchor.x_center,
                    y_top=t.top, y_bottom=t.bottom, text=t.text
                ))
    return clusters

def _extract(tokens: List[Token], anchors: List[WeekAnchor], brigade_hint: Optional[str]) -> pd.DataFrame:
    toks_sorted = sorted(tokens, key=lambda t: (t.page, t.top, t.x0))
    brigade = brigade_hint or next((t.text for t in toks_sorted if _is_brigade(t.text)), None)
    windows = _build_row_windows(toks_sorted)
    clusters = _build_activity_clusters(toks_sorted, anchors)
    rows_by_page: Dict[int, List[RowWindow]] = defaultdict(list)
    for rw in windows: rows_by_page[rw.page].append(rw)
    med_row_h: Dict[int, float] = {}
    for pg, rws in rows_by_page.items():
        hs = [rw.y_bottom - rw.y_top for rw in rws]
        hs.sort()
        med_row_h[pg] = hs[len(hs)//2] if hs else 24.0
    def iou_strict(c_top: float, c_bot: float, rw: RowWindow) -> float:
        inter = max(0.0, min(c_bot, rw.y_bottom) - max(c_top, rw.y_top))
        denom = (rw.y_bottom - rw.y_top)
        return inter / denom if denom > 0 else 0.0
    def iou_padded(c_top: float, c_bot: float, rw: RowWindow, pad: float) -> float:
        y0 = rw.y_top - pad
        y1 = rw.y_bottom + pad
        inter = max(0.0, min(c_bot, y1) - max(c_top, y0))
        denom = (rw.y_bottom - rw.y_top)
        return inter / denom if denom > 0 else 0.0
    rows = []
    for c in clusters:
        label_raw  = " ".join(c.text.split())
        label_join = label_raw.replace(" ", "")
        act = _normalize_activity_text(label_raw) or _normalize_activity_text(label_join)
        if not act: continue
        page_rows = rows_by_page.get(c.page, [])
        if not page_rows: continue
        h_row_med = med_row_h.get(c.page, 24.0)
        h_cluster = c.y_bottom - c.y_top
        strict = [(rw, iou_strict(c.y_top, c.y_bottom, rw)) for rw in page_rows]
        strict.sort(key=lambda z: z[1], reverse=True)
        overlapped = [(rw, r) for rw, r in strict if r >= MIN_IOU_MULTI]
        is_multi = (h_cluster > SINGLE_ROW_FACTOR * h_row_med) or (len(overlapped) >= MIN_ROWS_FOR_MULTI)
        if not is_multi:
            if strict and strict[0][1] >= MIN_IOU_SINGLE:
                best = strict[0][0]
                rows.append({"brigade": brigade, "unit": best.unit, "activity": act, "week": int(c.week)})
        else:
            pad = max(6.0, MULTI_EDGE_PAD_FACTOR * h_row_med)
            padded = [(rw, iou_padded(c.y_top, c.y_bottom, rw, pad)) for rw in page_rows]
            for rw, r in padded:
                if r >= MIN_IOU_MULTI:
                    rows.append({"brigade": brigade, "unit": rw.unit, "activity": act, "week": int(c.week)})
    df = pd.DataFrame(rows).drop_duplicates(subset=["brigade","unit","activity","week"]).reset_index(drop=True)
    cols = ["brigade","unit","activity","week"]
    for col in cols:
        if col not in df.columns: df[col] = None
    return df[cols].sort_values(["unit","week","activity"]).reset_index(drop=True)

# === 외부 사용 API ===
def parse_pdf_to_df(pdf_input: Union[str, bytes, BytesIO], brigade: Optional[str]=None) -> pd.DataFrame:
    """
    pdf_input: 파일경로(str) 또는 바이트/BytesIO
    """
    if isinstance(pdf_input, (bytes, bytearray)):
        pdf_obj = BytesIO(pdf_input)
    elif isinstance(pdf_input, BytesIO):
        pdf_obj = pdf_input
    else:
        pdf_obj = pdf_input  # path
    tokens  = _load_tokens(pdf_obj)
    anchors = _find_week_anchors(tokens)
    if not anchors:
        # 빈 DF 반환(헤더만)
        return pd.DataFrame(columns=["brigade","unit","activity","week"])
    return _extract(tokens, anchors, brigade)

def save_csv(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False, encoding="utf-8-sig")