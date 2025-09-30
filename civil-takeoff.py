#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
civil_takeoff_autoscale_plus.py  (Ubuntu/OCI ready)

All-in-one, CPU-only takeoff pipeline:
  • Per-page SCALE auto-detection (vector text → OCR title block fallback)
  • Asphalt area (SF) via OpenCV segmentation (binary + morphology)
  • Subtractions by keyworded exclusion masks (CONCRETE, SIDEWALK, BUILDING, LANDSCAPE…)
  • Optional texture rejection (avoid cross-hatch landscaping/structures)
  • Parking stall counting via Hough lines → row grouping
  • ADA count via OCR (keywords)
  • Outputs: totals.csv, meta.json, overlays/

Usage:
  python civil_takeoff_autoscale_plus.py "/path/to/site.pdf" --dpi 220 --outdir out \
         [--pages 1,3-6] [--no-ocr] [--config job.json]
"""

from __future__ import annotations
import os, re, json, math, argparse, logging
from typing import Dict, Optional, List, Tuple
from collections import Counter

import numpy as np
import cv2
import fitz  # PyMuPDF
from PIL import Image

# ---------- OCR (optional) ----------
try:
    import pytesseract
    HAS_TESS = True
except Exception:
    HAS_TESS = False

# On Ubuntu, tesseract is on PATH after: sudo apt install -y tesseract-ocr
# No hard-coded Windows path.

# ---------- Defaults (overridable via --config job.json) ----------
DEFAULTS = {
    "dpi": 220,

    # Asphalt segmentation & cleaning
    "asphalt_min_sf": 200.0,
    "hole_fill_sf": 40.0,
    "morph_kernel": 5,
    "morph_iter": 2,

    # Stall detection / grouping
    "stall_line_min_len_in": 14,
    "stall_line_max_len_in": 80,
    "stall_merge_ang_deg": 10,
    "stall_merge_gap_in": 30,
    "stall_row_min_count": 5,
    "stall_width_ft_default": 9.0,
    "stall_count_mode": "separators",  # "separators" or "row_length"

    # Exclusions
    "exclude_margin_ft": 6.0,
    "ignore_keywords": [
        "CONCRETE","SIDEWALK","WALK","CURB","BUILDING","STRUCTURE",
        "LANDSCAPE","TURF","GRASS","MULCH","PLANTER","HATCH"
    ],

    # Texture rejection (edge density)
    "texture_reject": True,
    "texture_window_px": 21,
    "texture_edge_thresh": 80.0,

    # ADA OCR
    "ada_keywords": ["ADA","ACCESSIBLE"]
}

# ---------- Utilities ----------
def feet_per_pixel(feet_per_inch: float, dpi: int) -> float:
    return feet_per_inch / float(dpi)

def feet_to_pixels(feet: float, fpp: float) -> float:
    return feet / max(1e-9, fpp)

def inches_to_pixels(inches: float, dpi: int) -> float:
    return inches * dpi / 96.0  # slight slack

# ---------- SCALE parsing ----------
QUOTE_IN  = r'["”″]|\bin\b'
PRIME_FT  = r"[’'′]|\bft\b|\bfeet\b"
PATTERNS = [
    rf'\bSCALE\b[^0-9a-zA-Z]*1\s*(?:{QUOTE_IN})\s*=\s*([0-9]+)\s*(?:{PRIME_FT})',
    rf'\b1\s*(?:{QUOTE_IN})\s*=\s*([0-9]+)\s*(?:{PRIME_FT})',
    r'\bSCALE\b[^0-9a-zA-Z]*1\s*(?:IN|INCH(?:ES)?)\s*=\s*([0-9]+)\s*(?:FT|FEET)\b',
    r'\b1\s*(?:IN|INCH(?:ES)?)\s*=\s*([0-9]+)\s*(?:FT|FEET)\b',
]
RATIO_PATS = [
    r'\bSCALE\b[^0-9a-zA-Z]*1\s*:\s*([0-9]+(?:\.[0-9]+)?)\b',
    r'\b1\s*:\s*([0-9]+(?:\.[0-9]+)?)\b',
]

def _normalize_text(s: str) -> str:
    s = s.replace("\u201d", '"').replace("\u201c", '"').replace("\u2033", '"')
    s = s.replace("\u2032", "'").replace("\u2019", "'")
    s = re.sub(r'\s+', ' ', s)
    return s.lower()

def parse_scale_feet_per_inch(text: str) -> Optional[float]:
    t = _normalize_text(text or "")
    for pat in PATTERNS:
        m = re.search(pat, t)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    for pat in RATIO_PATS:
        m = re.search(pat, t)
        if m:
            try:
                ratio = float(m.group(1))  # 1:ratio (inches)
                return ratio / 12.0        # feet per inch
            except Exception:
                pass
    if re.search(r'\b(as\s*noted|varies)\b', t):
        return None
    return None

# ---------- OCR (title block region) ----------
def ocr_title_block(page: fitz.Page, dpi: int) -> str:
    if not HAS_TESS:
        return ""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    w, h = img.size
    crop = img.crop((int(w*0.6), int(h*0.6), w, h)).convert("L")  # bottom-right 40%
    bw = crop.point(lambda x: 0 if x < 180 else 255, "1").convert("L")
    try:
        return pytesseract.image_to_string(bw)
    except Exception:
        return ""

# ---------- Detect per-page SCALE (feet/inch) ----------
def detect_scales(pdf_path: str, dpi: int, pages: List[int]) -> Dict[int, Optional[float]]:
    out: Dict[int, Optional[float]] = {}
    doc = fitz.open(pdf_path)
    for i in pages:
        page = doc[i]
        try:
            txt = page.get_text("text")
        except Exception:
            txt = ""
        fpi = parse_scale_feet_per_inch(txt)
        if not fpi:
            tb = ocr_title_block(page, dpi)
            fpi = parse_scale_feet_per_inch(tb)
        out[i] = fpi
    doc.close()
    return out

# ---------- Rasterize page ----------
def render_rgb(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img.copy()

# ---------- Text blocks + bboxes ----------
def page_text_blocks(page: fitz.Page) -> List[Tuple[str, Tuple[float,float,float,float]]]:
    out = []
    for b in page.get_text("blocks"):
        if len(b) >= 5 and isinstance(b[4], str):
            x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
            out.append((txt, (x0, y0, x1, y1)))
    return out

# ---------- Asphalt segmentation ----------
def asphalt_binary(rgb: np.ndarray, morph_kernel: int, morph_iter: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    th = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 25, 5)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, morph_kernel), max(1, morph_kernel)))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=max(1, morph_iter))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    return th

def texture_mask(rgb: np.ndarray, window_px: int, edge_thresh: float) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    win = max(3, window_px | 1)
    local = cv2.boxFilter(edges.astype(np.float32), ddepth=-1, ksize=(win, win), normalize=True)
    local = cv2.normalize(local, None, 0, 255, cv2.NORM_MINMAX)
    return (local > edge_thresh).astype(np.uint8)

def fill_and_filter(mask: np.ndarray, fpp: float, min_sf: float, hole_fill_sf: float) -> np.ndarray:
    if fpp <= 0:
        return np.zeros_like(mask)
    px_per_sf = 1.0 / (fpp * fpp)
    min_px = int(round(min_sf * px_per_sf))
    hole_px = int(round(hole_fill_sf * px_per_sf))
    ksz = max(1, int(round(math.sqrt(max(1, hole_px)))))
    ksz = min(31, ksz)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    num, labels = cv2.connectedComponents(closed)
    keep = np.zeros_like(mask)
    for lab in range(1, num):
        area = int((labels == lab).sum())
        if area >= max(1, min_px):
            keep[labels == lab] = 255
    return keep

def subtract_exclusions(mask: np.ndarray, exclusions: List[Tuple[int,int,int,int]]) -> np.ndarray:
    out = mask.copy()
    for (x0,y0,x1,y1) in exclusions:
        x0 = max(0, min(out.shape[1]-1, x0))
        x1 = max(0, min(out.shape[1]-1, x1))
        y0 = max(0, min(out.shape[0]-1, y0))
        y1 = max(0, min(out.shape[0]-1, y1))
        out[y0:y1+1, x0:x1+1] = 0
    return out

def compute_asphalt_sf(mask: np.ndarray, fpp: float) -> float:
    return float((mask > 0).sum()) * (fpp * fpp)

# ---------- Exclusions from text ----------
def build_exclusions_from_text(page: fitz.Page, dpi: int, fpp: float,
                               ignore_keywords: List[str],
                               exclude_margin_ft: float) -> List[Tuple[int,int,int,int]]:
    rects: List[Tuple[int,int,int,int]] = []
    blocks = page_text_blocks(page)
    margin_px = int(round(feet_to_pixels(exclude_margin_ft, fpp))) if fpp > 0 else 0
    scale = dpi / 72.0
    kws = {k.upper() for k in ignore_keywords}
    for txt, (x0,y0,x1,y1) in blocks:
        T = (txt or "").upper()
        if any(kw in T for kw in kws):
            rx0 = int(x0 * scale) - margin_px
            ry0 = int(y0 * scale) - margin_px
            rx1 = int(x1 * scale) + margin_px
            ry1 = int(y1 * scale) + margin_px
            rects.append((rx0, ry0, rx1, ry1))
    # OCR title-block carveout (if nothing else)
    if not rects and HAS_TESS:
        try:
            tb = ocr_title_block(page, dpi)
            if any(kw.lower() in (tb or "").lower() for kw in ignore_keywords):
                doc_rect = page.rect
                rx0 = int(doc_rect.x1 * 0.6 * scale)
                ry0 = int(doc_rect.y1 * 0.6 * scale)
                rects.append((rx0, ry0, int(doc_rect.x1*scale), int(doc_rect.y1*scale)))
        except Exception:
            pass
    return rects

# ---------- Stall detection ----------
def detect_stall_segments(rgb: np.ndarray, dpi: int,
                          min_len_in: float, max_len_in: float) -> List[Tuple[Tuple[int,int,int,int], float]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    min_len_px = int(round(inches_to_pixels(min_len_in, dpi)))
    max_len_px = int(round(inches_to_pixels(max_len_in, dpi)))
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35,
                            minLineLength=max(8, min_len_px), maxLineGap=12)
    out = []
    if lines is None:
        return out
    for l in lines[:,0,:]:
        x1,y1,x2,y2 = map(int, l)
        L = math.hypot(x2-x1, y2-y1)
        if L < min_len_px or L > max_len_px:
            continue
        ang = (math.degrees(math.atan2(y2-y1, x2-x1)) + 180.0) % 180.0
        out.append(((x1,y1,x2,y2), ang))
    return out

def group_rows(lines: List[Tuple[Tuple[int,int,int,int], float]],
               dpi: int, merge_ang_deg: float, merge_gap_in: float) -> List[List[Tuple[Tuple[int,int,int,int], float]]]:
    if not lines:
        return []
    rows: List[List[Tuple[Tuple[int,int,int,int], float]]] = []
    gap_px = int(round(inches_to_pixels(merge_gap_in, dpi)))
    for seg, ang in lines:
        x1,y1,x2,y2 = seg
        mx,my = (x1+x2)//2, (y1+y2)//2
        placed = False
        for row in rows:
            _, base_ang = row[0]
            dang = min(abs(ang-base_ang), 180-abs(ang-base_ang))
            if dang <= merge_ang_deg:
                near = any(abs(((sx1+sx2)//2)-mx) <= gap_px and
                           abs(((sy1+sy2)//2)-my) <= gap_px for (sx1,sy1,sx2,sy2),_ in row)
                if near:
                    row.append((seg, ang)); placed = True; break
        if not placed:
            rows.append([(seg, ang)])
    return rows

def count_stalls(rows: List[List[Tuple[Tuple[int,int,int,int], float]]],
                 mode: str, stall_width_ft: float, fpp: float) -> int:
    stalls = 0
    for r in rows:
        if len(r) < 2: continue
        if mode == "separators":
            stalls += max(0, len(r)-1)
        else:
            (x1,y1,x2,y2), ang = r[0]
            theta = math.radians(ang)
            ux, uy = math.cos(theta), math.sin(theta)
            pts = []
            for (a,b,c,d), _ in r:
                pts.extend([(a,b),(c,d)])
            proj = [p[0]*ux + p[1]*uy for p in pts]
            span_px = float(max(proj) - min(proj))
            span_ft = span_px * fpp
            if stall_width_ft > 0:
                stalls += int(round(span_ft / stall_width_ft))
    return int(stalls)

# ---------- ADA OCR ----------
def detect_ada(page: fitz.Page, dpi: int, ada_keywords: List[str]) -> int:
    if not HAS_TESS:
        return 0
    T = _normalize_text(ocr_title_block(page, dpi))
    cnt = 0
    for kw in ada_keywords:
        cnt += len(re.findall(rf'\b{re.escape(kw.lower())}\b', T))
    return min(cnt, 20)

# ---------- Overlays ----------
def draw_overlay(rgb: np.ndarray,
                 asphalt_mask_kept: np.ndarray,
                 exclusions: List[Tuple[int,int,int,int]],
                 lines: List[Tuple[Tuple[int,int,int,int], float]],
                 rows: List[List[Tuple[Tuple[Tuple[int,int,int,int], float]]]],
                 info_text: List[str]) -> np.ndarray:
    out = rgb.copy()
    if asphalt_mask_kept is not None and asphalt_mask_kept.size:
        tint = np.zeros_like(out)
        tint[asphalt_mask_kept>0] = (60, 200, 80)
        out = cv2.addWeighted(out, 1.0, tint, 0.35, 0)
    for (x0,y0,x1,y1) in exclusions:
        cv2.rectangle(out, (x0,y0), (x1,y1), (255,60,60), 2)
    for (x1,y1,x2,y2), _ in lines:
        cv2.line(out, (x1,y1), (x2,y2), (80,240,255), 1, cv2.LINE_AA)
    palette = [(255,80,80),(80,255,80),(80,120,255),(255,200,80),(180,90,255)]
    for i,row in enumerate(rows):
        col = palette[i % len(palette)]
        for (x1,y1,x2,y2), _ in row:
            cv2.line(out, (x1,y1), (x2,y2), col, 2, cv2.LINE_AA)
    y = 24
    for t in info_text:
        cv2.putText(out, t, (18,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, t, (18,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        y += 26
    return out

# ---------- Pages list ----------
def parse_pages(spec: str, n_pages: int) -> List[int]:
    if not spec:
        return list(range(n_pages))
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a,b = part.split("-",1)
            out.extend(list(range(int(a)-1, int(b))))
        else:
            out.append(int(part)-1)
    return [p for p in out if 0 <= p < n_pages]

# ---------- Main processing ----------
def process(pdf_path: str, outdir: str, cfg: dict, pages_spec: str, ocr_enabled: bool):
    os.makedirs(outdir, exist_ok=True)
    overlays_dir = os.path.join(outdir, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)

    dpi = int(cfg["dpi"])
    doc = fitz.open(pdf_path)
    pages = parse_pages(pages_spec, len(doc))

    # 1) Auto-scale per page
    logging.info("Detecting per-page SCALE…")
    scales = detect_scales(pdf_path, dpi, pages)
    valid = [v for v in scales.values() if v]
    common_fpi = Counter(valid).most_common(1)[0][0] if valid else None

    totals = {"ASPHALT_SF": 0.0, "STALLS": 0, "ADA": 0}
    meta_pages = []

    for idx in pages:
        page = doc[idx]
        rgb  = render_rgb(page, dpi)
        fpi  = scales[idx] if scales[idx] else common_fpi
        fpp  = feet_per_pixel(fpi, dpi) if fpi else 0.0

        # 2) Asphalt mask
        raw = asphalt_binary(rgb, cfg["morph_kernel"], cfg["morph_iter"])

        # 3) Texture rejection
        if cfg.get("texture_reject", True):
            tex = texture_mask(rgb, cfg.get("texture_window_px", 21),
                               cfg.get("texture_edge_thresh", 80.0))
            raw[tex>0] = 0

        # 4) Exclusions from text
        excl = build_exclusions_from_text(page, dpi, fpp,
                                          cfg["ignore_keywords"],
                                          cfg["exclude_margin_ft"])
        asphalt = subtract_exclusions(raw, excl)

        # 5) Fill / filter by SF
        asphalt = fill_and_filter(asphalt, fpp, cfg["asphalt_min_sf"], cfg["hole_fill_sf"])
        asphalt_sf = compute_asphalt_sf(asphalt, fpp)

        # 6) Stalls
        lines = detect_stall_segments(rgb, dpi,
                                      cfg["stall_line_min_len_in"],
                                      cfg["stall_line_max_len_in"])
        rows = group_rows(lines, dpi, cfg["stall_merge_ang_deg"], cfg["stall_merge_gap_in"])
        stall_width_ft = cfg.get("stall_width_ft_default", 9.0)
        stall_count = count_stalls(rows, cfg.get("stall_count_mode","separators"),
                                   stall_width_ft, fpp)

        # 7) ADA
        ada = detect_ada(page, dpi, cfg.get("ada_keywords", ["ADA","ACCESSIBLE"])) if ocr_enabled else 0

        # 8) Accumulate + overlay
        totals["ASPHALT_SF"] += asphalt_sf
        totals["STALLS"]     += stall_count
        totals["ADA"]        += ada

        info = [
            f"Page {idx+1}",
            f"Scale: {fpi:.4f} ft/in" if fpi else "Scale: (not found)",
            f"Asphalt: {asphalt_sf:,.2f} SF",
            f"Stalls: {stall_count}",
            f"ADA (OCR): {ada if ocr_enabled else 0}",
        ]
        overlay = draw_overlay(rgb, asphalt, excl, lines, rows, info)
        out_png = os.path.join(overlays_dir, f"page_{idx+1:03d}.png")
        cv2.imwrite(out_png, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        meta_pages.append({
            "page": idx+1,
            "fpi": fpi,
            "feet_per_pixel": fpp,
            "asphalt_sf": asphalt_sf,
            "stalls": stall_count,
            "ada": ada if ocr_enabled else 0,
            "overlay_png": os.path.relpath(out_png, outdir),
            "exclusion_boxes": excl
        })
        logging.info("Page %d done | Asphalt %.2f SF | Stalls %d | ADA %d",
                     idx+1, asphalt_sf, stall_count, ada)

    # 9) Write outputs
    with open(os.path.join(outdir, "totals.csv"), "w", encoding="utf-8") as f:
        f.write("ASPHALT_SF,STALLS,ADA\n")
        f.write(f"{totals['ASPHALT_SF']:.2f},{totals['STALLS']},{totals['ADA']}\n")

    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "src": os.path.abspath(pdf_path),
            "dpi": dpi,
            "scales_fpi": {str(k+1): scales[k] for k in pages},
            "totals": totals,
            "pages": meta_pages
        }, f, indent=2)

    logging.info("==== TAKEOFF COMPLETE ====")
    logging.info("ASPHALT_SF: %,.2f", totals["ASPHALT_SF"])
    logging.info("STALLS    : %d", totals["STALLS"])
    logging.info("ADA       : %d", totals["ADA"])
    logging.info("Outputs   : %s", outdir)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Automated civil takeoff (auto-scale, exclusions, stalls, ADA).")
    ap.add_argument("src", help="Path to plan PDF")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--dpi", type=int, default=DEFAULTS["dpi"])
    ap.add_argument("--pages", default="", help='1-based pages e.g. "1,3-5" (default: all)')
    ap.add_argument("--config", default="", help="Optional JSON to override defaults")
    ap.add_argument("--no-ocr", action="store_true", help="Disable OCR (faster; ADA=0; OCR scale fallback off)")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    lvl = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    if not os.path.isfile(args.src):
        logging.error("File not found: %s", args.src)
        raise SystemExit(2)

    cfg = DEFAULTS.copy()
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            user = json.load(f)
        cfg.update(user)
    cfg["dpi"] = int(args.dpi)

    process(args.src, args.outdir, cfg, args.pages, ocr_enabled=(not args.no_ocr))

if __name__ == "__main__":
    main()
