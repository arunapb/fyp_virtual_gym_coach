# AWS Deployment & Production Integration Plan

This document outlines how to transform your local "Module 1" prototype into a production-ready cloud service, how it will integrate with your team's other 3 modules, and how to clean up your project folder.

## User Review Required
> [!IMPORTANT]
> Please review this plan. This is a major architectural step that changes the project from a "desktop script" to a "cloud server application." 

## Open Questions
> [!TIP]
> 1. Do you want me to write the `api.py` wrapper right now so you can test it locally before uploading to AWS?
> 2. Does your frontend team plan to send **video frames (images)** to the backend, or will the frontend run MediaPipe locally and only send the **skeleton coordinates** to AWS? (Sending skeleton coordinates is 100x faster and cheaper for AWS!)

---

## 1. AWS Architecture (Connecting 4 Modules to 1 Frontend)

For a real-time gym coach with 4 AI modules, standard HTTP requests are too slow. You need a persistent, real-time connection.

**Proposed AWS Architecture:**
1. **Frontend Hosting:** Host your React/Angular/Vue frontend on **AWS Amplify** or **AWS S3 + CloudFront**. 
2. **Backend Server (The AI):** Host your AI modules on a powerful **AWS EC2 Instance** (or AWS ECS via Docker). 
3. **Communication (WebSockets):** The frontend connects to the EC2 server using **WebSockets**. As the user exercises, the frontend streams data to the server, and the server instantly replies with `{"state": "active", "exercise": "squat"}`.
4. **Routing:** If you have 4 modules, you can build one central `api.py` (using a framework like **FastAPI**) that imports all 4 modules and routes the data to the correct AI depending on what the user selected on the frontend.

---

## 2. Project Folder Cleanup (What to Keep vs. Remove)

When you deploy to AWS, you pay for storage and bandwidth. You **must** remove all training data and development scripts. 

### ✅ Files to KEEP (Upload to AWS)
These are the only files required to run the AI on a server:
* `src/` (Your core logic: `features.py`, `state_rules.py`, `exercise_rules.py`, `state_machine.py`)
* `models/` (The trained `.pt` weight files)
* `config.py` (Constants and thresholds)
* `requirements.txt` (We need to update this to include `torch`, `fastapi`, `uvicorn`, etc.)
* **[NEW]** `api.py` (A new file we will create to replace `app.py`. It will receive web requests instead of reading local `.mp4` files).

### 🗑️ Files to ARCHIVE (Move to a separate folder on your PC)
These files are for research, training, and testing only. Do **NOT** put them on AWS, but keep them backed up locally for emergencies.
* `data/` (Raw videos, CSVs, NumPy windows - *This takes up massive space!*)
* `training/` (All data building, splitting, and training scripts)
* `testing/` (Test `.mp4` clips)
* `app.py` & `verify_pipeline.py` (AWS EC2 does not have a computer monitor, so scripts using `cv2.imshow()` to draw windows will immediately crash the server).
* `*.zip` (Colab backup files)
* `*.llc` and markdown reports.

---

## 3. Proposed Next Steps
If you approve this plan, I will:
1. Update your `requirements.txt` with the exact server dependencies.
2. Write a production-ready `api.py` using **FastAPI** that wraps your `src/` logic.
3. Give you the exact command to move your unwanted files into a safe `Training_Archive` folder on your desktop.
