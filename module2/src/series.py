"""
webapp/series.py — which numeric metrics each exercise charts in the browser.

This is PRESENTATION configuration for the web layer only.  It selects a few of
the metrics an exercise already computes and states the thresholds those metrics
are judged against, so a chart can draw the decision boundary next to the trace.

Every number here is READ FROM `config.py` — none is redefined.  Re-deriving a
threshold (which several of them still need, per their config comments) changes
the chart automatically.

An exercise with no entry simply gets no charts; the zone timeline, rep table,
info bar and cue log are all generic and keep working.
"""

import config

# A chart = one panel: several traces sharing a y-axis, plus optional horizontal
# reference bands.
#   series : (feature_key, label)              — feature_key indexes the dict
#                                                Exercise.compute_features returns
#   bands  : (value, label, severity)          — severity in {"yellow", "red", "gate"}
#            "gate" marks a rep-counter state boundary rather than a fault line.

CHART_SPECS = {
    "Squat": [
        {
            "title": "Knee angle",
            "unit": "deg",
            "series": [("knee_angle_l_deg", "Left"), ("knee_angle_r_deg", "Right")],
            "bands": [(config.KNEE_STANDING, "stand", "gate"),
                      (config.KNEE_DESCENDING, "desc", "gate"),
                      (config.KNEE_BOTTOM, "bottom", "gate")],
        },
        {
            "title": "Trunk deviation from baseline",
            "unit": "deg",
            "series": [("trunk_lean_dev_deg", "Deviation")],
            "bands": [(config.TRUNK_DEV_YELLOW, "yellow", "yellow"),
                      (config.TRUNK_DEV_RED, "red", "red"),
                      (-config.TRUNK_DEV_YELLOW, "yellow", "yellow"),
                      (-config.TRUNK_DEV_RED, "red", "red")],
        },
        {
            "title": "Hip depth (lower = deeper)",
            "unit": "x femur",
            "series": [("norm_hip_depth", "Depth ratio")],
            "bands": [(config.DEPTH_YELLOW, "yellow", "yellow"),
                      (config.DEPTH_RED, "red", "red")],
        },
        {
            "title": "Knee valgus (normalised offset)",
            "unit": "x hip width",
            "series": [("norm_knee_offset_l", "Left"), ("norm_knee_offset_r", "Right")],
            "bands": [(config.VALGUS_L_YELLOW, "L yel", "yellow"),
                      (config.VALGUS_L_RED, "L red", "red"),
                      (config.VALGUS_R_YELLOW, "R yel", "yellow"),
                      (config.VALGUS_R_RED, "R red", "red")],
        },
    ],
    "BicepCurl": [
        {
            "title": "Elbow angle",
            "unit": "deg",
            "series": [("elbow_angle_l_deg", "Left"), ("elbow_angle_r_deg", "Right")],
            "bands": [(config.CURL_EXTENDED, "extend", "gate"),
                      (config.CURL_CURLING, "curling", "gate"),
                      (config.CURL_CONTRACTED, "contract", "gate")],
        },
        {
            "title": "Elbow drift from baseline",
            "unit": "x shoulder width",
            "series": [("elbow_drift_l", "Left"), ("elbow_drift_r", "Right")],
            "bands": [(config.ELBOW_DRIFT_YELLOW, "yellow", "yellow"),
                      (config.ELBOW_DRIFT_RED, "red", "red")],
        },
        {
            "title": "Body swing (trunk deviation)",
            "unit": "deg",
            "series": [("trunk_deviation_deg", "Deviation")],
            "bands": [(config.BODY_SWING_YELLOW, "yellow", "yellow"),
                      (config.BODY_SWING_RED, "red", "red")],
        },
    ],
    "ShoulderPress": [
        {
            "title": "Elbow angle",
            "unit": "deg",
            "series": [("elbow_angle_l_deg", "Left"), ("elbow_angle_r_deg", "Right")],
            "bands": [(config.PRESS_LOCKED, "locked", "gate"),
                      (config.PRESS_PRESSING, "press", "gate"),
                      (config.PRESS_RACKED, "racked", "gate")],
        },
        {
            "title": "Elbow flare from racked baseline",
            "unit": "x shoulder width",
            "series": [("elbow_flare_l", "Left"), ("elbow_flare_r", "Right")],
            "bands": [(config.PRESS_ELBOW_FLARE_YELLOW, "yellow", "yellow"),
                      (config.PRESS_ELBOW_FLARE_RED, "red", "red")],
        },
        {
            "title": "Body swing (trunk deviation)",
            "unit": "deg",
            "series": [("trunk_deviation_deg", "Deviation")],
            "bands": [(config.PRESS_BODY_SWING_YELLOW, "yellow", "yellow"),
                      (config.PRESS_BODY_SWING_RED, "red", "red")],
        },
    ],
}


def charts_for(exercise_name: str) -> list:
    """Chart definitions for one exercise, in browser-ready form."""
    return [
        {
            "title": c["title"],
            "unit": c["unit"],
            "series": [{"key": k, "label": lbl} for k, lbl in c["series"]],
            "bands": [{"value": round(v, 4), "label": lbl, "severity": sev}
                      for v, lbl, sev in c["bands"]],
        }
        for c in CHART_SPECS.get(exercise_name, [])
    ]


def metric_keys(exercise_name: str) -> list:
    """Every feature key the charts need, deduplicated and order-stable."""
    keys = []
    for chart in CHART_SPECS.get(exercise_name, []):
        for key, _label in chart["series"]:
            if key not in keys:
                keys.append(key)
    return keys
