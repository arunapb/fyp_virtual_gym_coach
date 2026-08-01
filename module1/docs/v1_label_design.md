# Module 1 — Version 1 Label Design Note

**Date:** 2026-05-01
**Scope:** Version 1 supports one target exercise only — **squat**.
**Team agreement required:** Everyone annotating or labeling data must follow these definitions exactly.
If two people label the same situation differently, the model will learn conflicting rules.

---

## 1. The Three States

These are the only valid state labels in version 1.

### NULL
> The system does not consider a workout session to be happening right now.

Use NULL when:
- No person is detected in the frame.
- A person is in the frame but is clearly not working out (e.g. standing around, walking, adjusting equipment, sitting on a bench before starting).
- A person is doing a movement that is **not** a squat and is not recovering after a squat set. Because squat is the only target exercise in v1, all other exercises count as NULL.
- A rest period has gone on for too long with no sign of continuing (timeout has expired — see Section 3).

NULL means: **"from the system's point of view, no exercise session is currently open."**

---

### REST
> A session is open, the person just finished a squat set, and the system is waiting to see if another set will begin.

Use REST when:
- The person just completed one or more squat reps and is now standing still, catching their breath, drinking water, looking at their phone, or resting in any way.
- The pause is **short enough** that it is reasonable to expect another squat set to follow.
- The timeout has **not yet expired**.

REST means: **"the workout session is paused, but it is still open."**

Key rule: REST can only follow ACTIVE. You cannot go directly from NULL to REST.
If a session has never opened (no squat reps have happened yet), inactivity is NULL, not REST.

---

### ACTIVE
> The person is currently performing squat repetitions.

Use ACTIVE when:
- The person is in the middle of a squat: bending their knees, lowering their hips, coming back up.
- The movement clearly matches the squat pattern (not just any leg movement).

ACTIVE means: **"a squat set is happening right now."**

In version 1, ACTIVE implicitly means **squat**. There is no separate exercise label.
If the movement is not a squat, it is not ACTIVE — it is NULL.

---

## 2. Squat Phase Labels

Phase labels are only assigned to frames that are labeled ACTIVE.
The five phases of a squat in order are:

| Phase | What is happening |
|-------|-------------------|
| **setup** | The person stands tall, feet shoulder-width apart, about to begin the squat. Knees are almost straight. Hips are at their highest position. |
| **descent** | The person is bending their knees and lowering their hips downward. The knee angle is decreasing (getting smaller). |
| **bottom** | The person is at the lowest point of the squat. Knee angle is at its minimum. Hips are at their lowest. This is usually only 1–3 frames. |
| **ascent** | The person is straightening their knees and raising their hips back up. Knee angle is increasing (getting larger). |
| **finish** | The person has returned to a standing position. Knees are nearly straight again. Movement is slowing down. The rep is complete. |

One full squat rep cycles through: **setup → descent → bottom → ascent → finish → (next rep) setup → ...**

> **Important:** Use the exact spelling above in all annotation files and code.
> Do **not** use "up", "down", "squat_down", or any other variation.
> The canonical list is defined in `config.py` as `SQUAT_PHASES`.

---

## 3. The Timeout Rule (Final-Set Problem)

After a squat set ends, the state changes from ACTIVE to REST.
The system waits up to **5 seconds** (`TIMEOUT_SECONDS` in `config.py`) for a new set to begin.

```
Squat set ends
      |
      v
   REST  <------ Timeout clock starts
      |
      |-- New squat reps begin --> ACTIVE (session continues, clock resets)
      |
      |-- 5 seconds pass with no squatting --> NULL (session closes)
```

**The final-set problem:** After the very last squat set of the day, the person just stops.
The system does not know it is the last set, so it first shows REST. After the timeout expires,
it correctly moves to NULL. This is the intended behaviour — do not treat the final rest as a
special case in the labels. Just annotate it as REST until the timeout point, then NULL after.

---

## 4. Corner Cases and Special Situations

### Drinking water between sets

**Label as: REST**

Reasoning: Drinking water is a normal part of rest between sets. The workout session is still open.
The person is expected to return to squatting. As long as the timeout has not expired, this is REST.

If they drink water for so long that the timeout expires, the state becomes NULL from that point.

---

### Unknown or non-target exercise (e.g. push-ups, lunges, bicep curls)

**Label as: NULL**

Reasoning: In version 1, the only target exercise is squat. Any other movement is not a target exercise.
The system should not mark it as ACTIVE.

Exception: If the person does a non-target exercise **immediately between two squat sets** and the
gap is short, you may treat the whole gap as REST. Discuss with the team and be consistent.
Do not mix REST and NULL for the same type of situation across different videos.

---

### Walking to get a weight plate or adjust the barbell

**Label as: NULL** (before the session opens) or **REST** (between sets)

If this happens before the person has done any squats: NULL.
If this happens between squat sets within the session: REST.

---

### Adjusting form or position (e.g. stepping into squat stance)

**Label as: setup** (if already ACTIVE) or **NULL** (if no squat has been done yet)

If the person is clearly preparing for a squat rep (stepping into position, adjusting their feet,
unracking the bar), and a squat is about to happen, label it as ACTIVE with phase = setup.
If they are just wandering around before any squat, label it NULL.

---

### Partial squat reps (person stops halfway)

**Label as: ACTIVE** for the frames where the movement is happening.

Even incomplete reps should be labeled ACTIVE if the squat pattern is visible.
Use descent for the downward phase and ascent for any upward recovery.
If they stop in the bottom position and hold it, label the hold as bottom.

---

### Person walks out of frame mid-set

**Label as: NULL** for the frames where no pose is detected.
**Label as: REST** if the person returns quickly and resumes squatting.

If MediaPipe cannot detect a person, the frame-level data will have NaN landmarks.
Do not label frames with no pose as ACTIVE or REST.

---

### Long pause that looks like rest but person has left the gym

This is a judgment call. Use the timeout rule as your guide:
- If the gap is under 5 seconds and the person returns: REST → ACTIVE
- If the gap is over 5 seconds: REST → NULL (timeout triggers automatically)

For annotation purposes, label the gap as REST and let the state machine handle the timeout.
Do not manually annotate the exact frame where NULL begins — that is calculated automatically.

---

### Multiple people in frame

In version 1, track the most visible person. If multiple people are doing squats simultaneously,
label the session based on the primary subject. Note in the `notes` column which person is being tracked.

---

## 5. Annotation CSV Format

Each video gets one annotation file in `data/annotations/`. Use this column format:

| Column | Type | Example | Rules |
|--------|------|---------|-------|
| `video_id` | string | `video_001` | Match the video filename without extension |
| `start_frame` | int | `0` | Inclusive. First frame of this segment. |
| `end_frame` | int | `149` | Inclusive. Last frame of this segment. |
| `state` | string | `active` | One of: `null`, `rest`, `active` (lowercase) |
| `exercise` | string | `squat` | `squat` if active, `none` otherwise |
| `phase` | string | `descent` | One of the 5 SQUAT_PHASES if active, `none` otherwise |
| `notes` | string | `drinking water` | Optional. Use for edge cases or reviewer comments. |

**Rules:**
- Segments must be continuous. Every frame in the video must belong to exactly one segment.
- No gaps between segments (end_frame of row N + 1 = start_frame of row N+1).
- No overlaps between segments.
- Phase must be `none` when state is `null` or `rest`.
- Phase must be one of the five canonical names when state is `active`.

**Example annotation file for a short session:**

```csv
video_id,start_frame,end_frame,state,exercise,phase,notes
video_001,0,89,null,none,none,person setting up equipment
video_001,90,104,active,squat,setup,unracking bar and stepping into stance
video_001,105,139,active,squat,descent,first rep going down
video_001,140,145,active,squat,bottom,lowest point of first rep
video_001,146,178,active,squat,ascent,coming back up
video_001,179,195,active,squat,finish,standing tall end of first rep
video_001,196,210,active,squat,descent,second rep starting
video_001,211,218,active,squat,bottom,bottom of second rep
video_001,219,248,active,squat,ascent,coming up
video_001,249,269,active,squat,finish,end of second rep
video_001,270,450,rest,none,none,drinking water between sets
video_001,451,460,active,squat,setup,starting second set
video_001,461,999,active,squat,descent,... (continue labeling)
```

---

## 6. Quick Reference — Label Decision Tree

```
Is a person detected in the frame?
   NO  --> NULL

   YES --> Has a squat set ever started in this session?
              NO  --> NULL

              YES --> Is the person currently doing squat reps?
                         YES --> ACTIVE  (assign one of the 5 phases)

                         NO  --> How long has it been since the last squat rep?
                                    Under 5 seconds --> REST
                                    Over 5 seconds  --> NULL  (timeout, session closes)
```

---

## 7. What Everyone on the Team Must Agree On

Before annotation starts, all annotators must agree on:

1. **Where does one squat rep end and the next one begin?**
   Suggested rule: finish ends when the person is fully upright and motion energy drops.
   The next descent starts when hips begin moving downward again.

2. **What counts as "setup" vs "null"?**
   Suggested rule: setup only applies once the person is in position and clearly about to squat
   (bar on shoulders, stance set). Walking to the squat rack is still null.

3. **What is the minimum squat depth to count as a rep?**
   Suggested rule: if the knee angle drops below 130 degrees, count it as a squat.
   Shallow knee bends that do not reach this depth can be labeled as descent + ascent
   without a bottom phase, or excluded entirely if they look like fidgeting.

4. **How to handle unclear or ambiguous frames?**
   Suggested rule: leave a note in the `notes` column and flag for team review before training.
   Do not guess on ambiguous segments — inconsistent labels are worse than missing labels.
