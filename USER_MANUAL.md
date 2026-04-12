# FairPlay Review System — User Manual

This document explains how to use **FairPlay Review System**: a cricket **LBW (leg-before-wicket) fair-play review** product backed by computer vision (ball, batsman, and wicket detection) and a secure API. Your team may access it through a mobile or web app that talks to this API; the steps below describe **what you do in the product**, aligned with how the system works.

---

## 1. What this app is for

FairPlay Review helps you:

- **Organize matches** you are reviewing (teams, venue, date, status).
- **Configure wicket positions** (near and far stumps) so geometry-based LBW analysis is meaningful for that ground clip.
- **Upload a delivery clip** and state the **on-field umpire’s original decision** (`OUT` or `NOT OUT`).
- Receive an **AI-assisted LBW suggestion** (with confidence when available), optional **review card imagery**, and a **stored review record** you can revisit later.
- Get **notifications** (for example when a long-running match is auto-completed by the system).

The system is **assistive**: final judgment may still rest with human reviewers and competition rules. Results can be **inconclusive** when the pipeline cannot determine a clear outcome.

---

## 2. Before you start

### 2.1 Account

You need a registered account (email and password). Signing up creates your profile and issues an **access token** used for all protected actions.

**Typical actions**

- **Sign up** — create your account.
- **Log in** — start a session; the client usually stores your token securely.
- **Change password** — while logged in, or using email + old password if the client supports recovery-style flows.

### 2.2 What you need for video review

- A **short video** of the relevant delivery (format supported by the pipeline; commonly MP4).
- The **original umpire decision** for that delivery: **`OUT`** or **`NOT OUT`** (required for full analysis).
- Optional: a **still image** of the pitch/wicket area for **automatic wicket configuration** (see section 4).

### 2.3 Match status vocabulary

The app accepts friendly labels; the server normalizes them. Examples:

| You might see / enter | Stored as      |
|----------------------|----------------|
| Upcoming             | `scheduled`    |
| Live                 | `in_progress`  |
| Completed / Finished | `completed`  |
| Cancelled / Canceled | `cancelled`    |
| Postponed            | `postponed`    |

Use **in progress** while you are actively working on a match (uploads, analysis, heartbeats).

---

## 3. Core workflow (recommended order)

1. **Log in**
2. **Create a match** (name, teams, venue, date, status)
3. **Set wicket configuration** (auto from an image and/or manual boxes) — **recommended before analyzing video** for reliable LBW geometry
4. **Run video analysis** on a clip for that match, with **original decision** `OUT` or `NOT OUT`
5. **Open your reviews** to see stored outcomes, analysis text, and links to processed video / review card assets
6. **Use match heartbeat** if you keep a match “live” for a long session (section 5)

Exact screen names depend on your client app, but the logic above matches the backend.

---

## 4. Matches

### 4.1 Creating a match

Provide at least:

- **Name** — label for the fixture or session  
- **Teams** — usually as a single string such as `Team A vs Team B` (the system can split this for storage)  
- **Date** — flexible formats are accepted and normalized  
- **Status** — often `upcoming` / `live` / `completed` from the UI  

Matches are **owned by your account**; you only see and edit **your** matches.

### 4.2 Listing and opening a match

You can list all your matches and open one by ID to see details (teams, venue, dates, completion metadata if relevant).

### 4.3 Updating a match

You can change name, teams, venue, date, or status — for example, mark a match **completed** when your review session is done.

### 4.4 Deleting a match

Removes that match **for your account** according to server rules (ensure the client confirms destructive actions).

---

## 5. Long sessions: heartbeat and auto-completion

### 5.1 Heartbeat

While a match is **in progress**, the client can send a **heartbeat** periodically. That **refreshes activity** on the match so the system knows you are still working.

### 5.2 Automatic completion after inactivity

If a match stays **in progress** but **no activity** is recorded for an extended period (default **24 hours**), the system may **automatically mark it completed** and record that it was completed by the system. You may receive a **notification** when that happens.

You can also run a **maintenance** action (where exposed in the client) to apply the same stale-match completion logic on demand.

---

## 6. Wicket configuration

Wicket configuration tells the analysis **where the near and far wickets are** in the image plane. This matters for LBW-related geometry.

### 6.1 Checking configuration

You can fetch the current configuration for a match. The response includes:

- **`configured`** — whether the setup is considered complete (typically **`true` when the far wicket box is present**; near box may sometimes be absent and the config can still be usable)  
- **`status`** — such as idle, processing, or error states during auto-detection  
- **`near_box` / `far_box`** — bounding regions when available  
- Paths or metadata for **annotated preview images** when the pipeline produces them  

### 6.2 Automatic configuration

Upload a **still image** (multipart field `image_file`; some clients may still use legacy `video_file`). The server saves the file and runs **wicket detection in the background**. Poll the GET configuration endpoint until processing finishes or an error is shown.

You can often tune **detection confidence** and **display/debug** options if your client exposes them.

### 6.3 Manual configuration

If auto-detection is wrong or unavailable, you (or an operator) can **manually set** near and far wicket boxes so the match is marked configured.

---

## 7. Video analysis (LBW review)

### 7.1 What you submit

- **Match ID** — the match this clip belongs to  
- **Video file** — the delivery or replay clip  
- **Original decision** — required: **`OUT`** or **`NOT OUT`** (the on-field call you are reviewing)

Optional parameters (when exposed) control detector behavior, for example:

- Confidence thresholds for **person**, **bat**, **pads**, **wicket**  
- **IoU** and **consecutive frames** for batsman/bat association  
- **FPS**, **preprocessing**, and **on-screen debug display** during processing  

Defaults are chosen for a balance of speed and quality; change them only if you understand the trade-offs.

### 7.2 What you get back

The analysis response includes:

- **`decision`** — suggested LBW outcome: **`OUT`** or **`NOT OUT`** when determined  
- **`original_decision`** — echo of what you submitted  
- **`confidence`** — model confidence when provided  
- **`review_outcome`** — may be **`inconclusive`** when the system cannot commit to OUT/NOT OUT  

The pipeline may also produce **processed video** and an **LBW review card image**; these are stored in secure storage and linked from your **review** record (see section 8).

### 7.3 Standalone detection tools (optional)

Some deployments expose separate endpoints to run **ball-only**, **batsman-only**, or **wicket-only** detection on a clip for debugging or demos. These do not replace the full LBW review flow unless your product is built around them.

---

## 8. Reviews

A **review** ties together a match, your notes, decisions, and media.

Typical fields include:

- **Match** reference and optional **match name**  
- **Delivery / over** labels for context  
- **Original decision** and **final decision** (`OUT` / `NOT OUT`)  
- **Impact, pitch, wickets** — textual or coded context your app uses  
- **Video URI** and **LBW review card URI** — storage paths or URLs the client resolves for playback  
- **Content** and **analysis** — free text or structured analysis  

You can list all reviews, fetch one by ID, or list reviews **for a specific match**. Updates and deletes are scoped to **your** data.

---

## 9. Notifications

You can:

- **List notifications** (e.g., system events like auto-completed matches)  
- **Mark a notification as read**  
- **Adjust settings**, such as:  
  - **Match alerts**  
  - **Review updates**  
  - **System notifications**  

Turn categories off if you want fewer interruptions.

---

## 10. Profile

Your profile usually includes **name**, **email**, and **avatar**.

- **View profile** — see current details  
- **Update profile** — change name or email where allowed  
- **Upload avatar** — replace profile image  

Some profile actions may duplicate **auth profile** endpoints depending on the client; use whichever your app exposes.

---

## 11. Watching review videos and images

Processed media often lives in **private cloud storage**. The API may expose a **redirect URL** that issues a **short-lived signed link** (for example, about one hour) so your app can stream video or show images securely. If playback stops after a long pause, **refresh the link** by requesting the asset again.

Paths typically look like: `reviews/user_<id>/match_<id>/<filename>`.

---

## 12. Health check

A **health** endpoint is available for operators to confirm the API is running. End users normally do not need this.

---

## 13. Limitations and good practice

- **Video length and resolution** affect processing time and memory; use the shortest clip that still contains the full delivery and impact.  
- **Lighting and angle** strongly affect detection quality; wicket configuration should match the same camera geometry as the delivery clip when possible.  
- **Network**: large uploads need a stable connection; some clients should show upload progress and retry policies.  
- **Security**: never share your password or raw API tokens; use only the official app or documented tools.

---

## 14. Getting help (technical)

- **Interactive API documentation** is available from the running server at `/docs` (Swagger UI), which lists paths, parameters, and schemas.  
- **Administrator / developer setup** (database, environment variables, Supabase) is described in project docs such as `README.md` and `SUPABASE_SETUP_GUIDE.md`, not in this user-focused manual.

---

## Document version

Aligned with **FairPlayReviewSystem API v1.1.x** and the behaviors described in the repository at the time of writing. If your deployment differs (custom thresholds, feature flags, or UI-only steps), follow your local administrator’s guidance.
