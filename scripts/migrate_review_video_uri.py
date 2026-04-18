from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from API.core.supabase_client import supabase_admin_client, supabase_client

REVIEWS_TABLE = "reviews"
REVIEW_VIDEOS_BUCKET = "review-videos"


def _write_client():
    return supabase_admin_client or supabase_client


def _is_windows_local_path(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return False
    # Typical: D:\foo\bar.mp4
    if len(v) >= 3 and v[1:3] == ":\\":
        return True
    # Sometimes: D:/foo/bar.mp4
    if len(v) >= 3 and v[1:3] == ":/":
        return True
    return False


def _upload_mp4_and_get_object_path(*, local_path: Path, user_id: int, match_id: int, review_id: int) -> str:
    if not local_path.exists():
        raise FileNotFoundError(str(local_path))
    payload = local_path.read_bytes()
    if not payload:
        raise RuntimeError("mp4 is empty")

    object_path = f"reviews/user_{user_id}/match_{match_id}/review_{review_id}_{uuid.uuid4().hex}.mp4"
    storage = _write_client().storage.from_(REVIEW_VIDEOS_BUCKET)
    try:
        storage.upload(object_path, payload, {"content-type": "video/mp4", "upsert": "true"})
    except TypeError:
        storage.upload(object_path, payload)
    return object_path


def _fetch_candidate_rows(batch_size: int = 1000) -> List[Dict[str, Any]]:
    # Pull only required columns; filter for non-null video_uri on DB side.
    resp = (
        supabase_client.table(REVIEWS_TABLE)
        .select("id, user_id, match_id, video_uri")
        .not_.is_("video_uri", "null")
        .limit(batch_size)
        .execute()
    )
    rows = resp.data or []
    return [r for r in rows if _is_windows_local_path(str(r.get("video_uri") or ""))]


def _update_row(review_id: int, *, video_uri: Optional[str]) -> None:
    payload: Dict[str, Any] = {"video_uri": video_uri}
    supabase_client.table(REVIEWS_TABLE).update(payload).eq("id", review_id).execute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate reviews.video_uri from local paths to Supabase Storage object paths.")
    parser.add_argument("--dry-run", action="store_true", help="Do not upload or update DB; only print what would happen.")
    parser.add_argument("--clear-missing", action="store_true", help="If local file is missing, set video_uri to null.")
    parser.add_argument("--batch-size", type=int, default=1000, help="How many rows to fetch per run (default: 1000).")
    args = parser.parse_args()

    # Ensure imports work when running from repo root.
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    rows = _fetch_candidate_rows(batch_size=args.batch_size)
    print(f"Found {len(rows)} review rows with local video_uri paths.")
    if not rows:
        return

    migrated = 0
    cleared = 0
    failed = 0

    for r in rows:
        review_id = int(r["id"])
        user_id = int(r["user_id"])
        match_id = int(r["match_id"])
        video_uri = str(r.get("video_uri") or "")
        local_path = Path(video_uri)

        print(f"- review_id={review_id} local={video_uri}")

        if not local_path.exists():
            msg = "missing"
            if args.clear_missing:
                msg = "missing -> will clear video_uri"
                if not args.dry_run:
                    _update_row(review_id, video_uri=None)
                cleared += 1
            else:
                failed += 1
            print(f"  {msg}")
            continue

        try:
            object_path = _upload_mp4_and_get_object_path(
                local_path=local_path,
                user_id=user_id,
                match_id=match_id,
                review_id=review_id,
            )
            print(f"  upload_ok -> {object_path}")
            if not args.dry_run:
                _update_row(review_id, video_uri=object_path)
            migrated += 1
        except Exception as exc:
            failed += 1
            print(f"  upload_failed: {exc}")

    print("Done.")
    print(f"  migrated: {migrated}")
    print(f"  cleared : {cleared}")
    print(f"  failed  : {failed}")


if __name__ == "__main__":
    main()

