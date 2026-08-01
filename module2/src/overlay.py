"""
src/overlay.py — generic OpenCV rendering, driven by an Exercise DisplaySpec.

These functions are exercise-agnostic: they render whatever the active
exercise's get_display_spec() produced.  The exercise decides the content
(labels, units, which zone channel colours which body region); overlay.py only
knows how to paint it.

Public functions:
  draw_text()       — text with a black shadow outline (legible on any backdrop)
  draw_pose()       — skeleton, dots, torso line, key joints (from a DisplaySpec)
  draw_aura()       — blurred coloured glow around skeleton regions
  draw_overlay()    — top-left rep badge
  render_info_bar() — the info panel below the video (columns + rep + overall)
  render_banner()   — minimal banner for Null/Rest states
  annotate_frame()  — the frame-only composition used by the offline analyser
  compose_display() — frame + info bar stacked, as the desktop window shows it
"""

import cv2
import numpy as np

from config import (
    BAR_H, POSE_CONNECTIONS, KEY_JOINTS,
    IDX_SHOULDER_L, IDX_SHOULDER_R, IDX_HIP_L, IDX_HIP_R,
)
from src.pose_utils import get_px, midpoint
from src.zones import get_zone_color


# =============================================================================
# TEXT HELPER
# =============================================================================

def draw_text(frame, text, pos, scale=0.6, color=(255, 255, 255), thickness=1):
    """
    Draw text with a black shadow underneath so it is legible on any background.

    The shadow is drawn first (slightly thicker, black), then the coloured text
    is drawn on top — this gives a clean outline effect without extra geometry.
    """
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color,     thickness,     cv2.LINE_AA)


# =============================================================================
# POSE / SKELETON
# =============================================================================

def draw_pose(frame, lm, spec, w, h):
    """
    Draw skeleton, landmark dots, torso line, and key joints from a DisplaySpec.

    Constant-colour connections + green dots, a torso centre-line in
    spec.torso_color, and key joints coloured by spec.joint_colors (gold default).
    """
    for a, b in POSE_CONNECTIONS:
        if a < len(lm) and b < len(lm):
            ax, ay = int(lm[a].x * w), int(lm[a].y * h)
            bx, by = int(lm[b].x * w), int(lm[b].y * h)
            cv2.line(frame, (ax, ay), (bx, by), (0, 180, 255), 2)

    for lm_pt in lm:
        cx, cy = int(lm_pt.x * w), int(lm_pt.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 220, 0), -1)

    sh_mid = midpoint(get_px(lm, IDX_SHOULDER_L, w, h), get_px(lm, IDX_SHOULDER_R, w, h))
    hi_mid = midpoint(get_px(lm, IDX_HIP_L, w, h), get_px(lm, IDX_HIP_R, w, h))
    cv2.line(frame, sh_mid, hi_mid, spec.torso_color, 4)

    for label, idx in KEY_JOINTS.items():
        if idx < len(lm):
            cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
            dot_color = spec.joint_colors.get(idx, (255, 230, 0))
            cv2.circle(frame, (cx, cy), 7, dot_color, -1)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 0), 1)
            draw_text(frame, label, (cx + 9, cy + 5), scale=0.38, color=dot_color)


# =============================================================================
# AURA
# =============================================================================

def conns_is_limb(conns):
    """Heuristic: the catch-all 'other' region is long; limb/torso regions short."""
    return len(conns) <= 5


def draw_aura(frame, lm, aura_regions, w, h, overall_red=False):
    """Generic aura glow: paint each (connections, colour) region, blur, blend."""
    aura = np.zeros_like(frame)

    def aura_thick(color):
        # Red regions get a wider brush (red BGR ~ (80, 80, 255)).
        return 28 if color == (80, 80, 255) else 20

    for conns, color in aura_regions:
        thick = aura_thick(color) if conns_is_limb(conns) else 16
        for a, b in conns:
            if a < len(lm) and b < len(lm):
                ax, ay = int(lm[a].x * w), int(lm[a].y * h)
                bx, by = int(lm[b].x * w), int(lm[b].y * h)
                cv2.line(aura, (ax, ay), (bx, by), color, thick)

    glow = cv2.GaussianBlur(aura, (41, 41), 0)
    alpha = 0.72 if overall_red else 0.50
    cv2.addWeighted(frame, 1.0, glow, alpha, 0, frame)


# =============================================================================
# REP BADGE OVERLAY
# =============================================================================

def draw_overlay(frame, overlay):
    """Generic rep badge in the top-left corner from a DisplaySpec.overlay dict."""
    if overlay is None:
        return
    phase_str = overlay["phase"]
    count = overlay["rep_count"]
    phase_col = overlay["phase_color"]
    last_text = overlay.get("last_text")

    count_str = str(count)
    (cw, ch), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
    (pw, _), _ = cv2.getTextSize(phase_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    # last_text MUST be measured too.  It was previously left out of the width
    # calculation, so the quality line spilled outside the translucent panel
    # whenever it was the longest line — by ~31 px for squat and ~53 px for the
    # bicep curl, whose per-arm wording is longer.  That overflow, not the font
    # size, is what made the badge look cramped.
    lw = (cv2.getTextSize(last_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0]
          if last_text else 0)
    box_w = max(cw + 60, pw + 20, lw + 20, 150)
    # Line baselines are spaced 26 px / 24 px apart rather than 18 px / 23 px, so
    # the three lines read as separate rows instead of one block.
    box_h = 118 if last_text else 92

    x1, y1 = 8, 8
    overlay_img = frame.copy()
    cv2.rectangle(overlay_img, (x1, y1), (x1 + box_w, y1 + box_h), (15, 15, 15), -1)
    cv2.rectangle(overlay_img, (x1, y1), (x1 + box_w, y1 + box_h), (60, 60, 60), 1)
    cv2.addWeighted(overlay_img, 0.65, frame, 0.35, 0, frame)

    draw_text(frame, count_str, (x1 + 10, y1 + 56), scale=1.8, color=(255, 255, 255), thickness=3)
    draw_text(frame, "REPS", (x1 + 14 + cw, y1 + 52), scale=0.55, color=(160, 160, 160))
    draw_text(frame, phase_str, (x1 + 10, y1 + 82), scale=0.52, color=phase_col)
    if last_text:
        draw_text(frame, last_text, (x1 + 10, y1 + 106), scale=0.46,
                  color=overlay.get("last_color", (200, 200, 200)))


# =============================================================================
# INFO BAR
# =============================================================================

def render_info_bar(spec, w):
    """Generic info bar from a DisplaySpec (columns + rep row + overall row)."""
    bar = np.zeros((BAR_H, w, 3), dtype=np.uint8)
    bar[0:2, :] = (55, 55, 55)

    c_x = [12, w // 3 + 8, 2 * w // 3 + 8]
    DATA_BOTTOM, SEP_Y1, REP_ROW_Y, SEP_Y2, OVERALL_Y = 112, 118, 136, 150, 168
    cv2.line(bar, (w // 3, 4), (w // 3, DATA_BOTTOM), (55, 55, 55), 1)
    cv2.line(bar, (2 * w // 3, 4), (2 * w // 3, DATA_BOTTOM), (55, 55, 55), 1)
    cv2.line(bar, (4, SEP_Y1), (w - 4, SEP_Y1), (55, 55, 55), 1)
    cv2.line(bar, (4, SEP_Y2), (w - 4, SEP_Y2), (55, 55, 55), 1)
    ly = [16, 40, 64, 88]

    for col_idx, cells in enumerate(spec.columns[:3]):
        for row_idx, cell in enumerate(cells[:4]):
            draw_text(bar, cell.text, (c_x[col_idx], ly[row_idx]), 0.50, cell.color)

    if spec.rep_row is not None:
        rr = spec.rep_row
        draw_text(bar, f"Phase: {rr['phase']}", (c_x[0], REP_ROW_Y), 0.50, rr["phase_color"])
        rep_text = f"Reps: {rr['rep_count']}"
        (rtw, _), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        draw_text(bar, rep_text, ((w - rtw) // 2, REP_ROW_Y), 0.55, (255, 255, 255))
        if rr.get("last_text"):
            draw_text(bar, rr["last_text"], (c_x[2], REP_ROW_Y), 0.48,
                      rr.get("last_color", (200, 200, 200)))

    if spec.overall is not None:
        text = spec.overall["text"]
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        draw_text(bar, text, ((w - tw) // 2, OVERALL_Y), 0.58, spec.overall["color"])
    return bar


def _wrap_to_width(text, w, scale, thickness, margin=24):
    """
    Greedy word-wrap so every line fits inside the frame width.

    Measured with the real font metrics rather than a character count, because
    Hershey glyphs are not fixed-width.  A single word longer than the limit is
    emitted on its own line rather than dropped.
    """
    limit = max(1, w - margin)
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        (tw, _), _ = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw <= limit or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def render_banner(w, text, h=BAR_H):
    """
    Minimal banner for Null/Rest states (no info bar, no rep overlay).

    Wraps to as many lines as the width needs; a banner that already fits stays
    on one line at exactly its previous baseline, so short banners are unchanged.
    """
    bar = np.zeros((h, w, 3), dtype=np.uint8)
    bar[0:2, :] = (55, 55, 55)
    scale, thickness, line_h = 0.7, 2, 30
    lines = _wrap_to_width(text, w, scale, thickness)
    y0 = h // 2 + 8 - (len(lines) - 1) * line_h // 2
    for i, line in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        draw_text(bar, line, ((w - tw) // 2, y0 + i * line_h), scale,
                  (220, 220, 220), thickness=thickness)
    return bar


# =============================================================================
# FRAME COMPOSITION
# =============================================================================
# Two consumers compose a frame from the same DisplaySpec, and they need
# different amounts of it:
#
#   * the desktop window can only show pixels, so it stacks the info bar under
#     the annotated frame and displays one image (`compose_display`);
#   * the web analyser encodes only the annotated FRAME and sends the info bar
#     to the browser as data, where it renders as live HTML beside the video.
#
# `annotate_frame` is the shared half, so both paths draw exactly the same
# picture and cannot drift apart.

def annotate_frame(frame, spec, lm, zones, pose_detected, w, h) -> None:
    """
    Draw this frame's visual feedback in place: aura, skeleton, torso line, key
    joints, rep badge, and the overall-zone border.

    No-op when there is no pose or no zones yet, which is what leaves the raw
    frame visible during calibration.
    """
    if not (pose_detected and zones):
        return
    overall_red = zones["overall"] == "red"
    draw_aura(frame, lm, spec.aura_regions, w, h, overall_red=overall_red)
    draw_pose(frame, lm, spec, w, h)
    draw_overlay(frame, spec.overlay)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1),
                  get_zone_color(zones["overall"]), 8 if overall_red else 5)


def compose_display(frame, spec, w, banner=None):
    """
    Stack the info bar (or the Null/Rest banner) beneath an annotated frame and
    return the single image the OpenCV window shows.

    `spec` is None while no exercise is active, in which case `banner` supplies
    the Null/Rest prompt.
    """
    if spec is None:
        return np.vstack([frame, render_banner(w, banner or "")])
    return np.vstack([frame, render_info_bar(spec, w)])
