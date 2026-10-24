"""Cliplane worker: claims queued jobs from Supabase and runs them.
Sources are files the user uploaded to their own storage folder. The worker never fetches user-supplied URLs."""
import os, shutil, sys, tempfile, time, glob, json, datetime as dt
import requests
import highlights, render

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")  # secret: never log or expose
POLL = int(os.environ.get("POLL_SECONDS", "10"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "3"))
BUCKET = "cl-media"
H = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
DEFAULT_CAPTIONS = {"font": "Anton", "size": 30, "color": "#ffffff", "background": "none",
                    "position_pct": 78, "resolution": "1080x1920"}


def log(msg):
    print(f"{dt.datetime.utcnow().isoformat(timespec='seconds')}Z {msg}", flush=True)


def rest(method, table, params=None, body=None, prefer="return=representation"):
    r = requests.request(method, f"{URL}/rest/v1/{table}", headers={**H, "Prefer": prefer}, params=params,
                         data=json.dumps(body) if body is not None else None, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"Database error {r.status_code} on {table}")
    return r.json() if r.text else []


def download(path, dest):
    with requests.get(f"{URL}/storage/v1/object/{BUCKET}/{path}", headers=H, stream=True, timeout=60) as r:
        if r.status_code != 200:
            raise RuntimeError("Could not download the source file from storage.")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def upload(path, src):
    with open(src, "rb") as f:
        r = requests.post(f"{URL}/storage/v1/object/{BUCKET}/{path}", timeout=300,
                          headers={"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "video/mp4",
                                   "x-upsert": "true"}, data=f)
    if r.status_code >= 300:
        raise RuntimeError(f"Upload of the rendered clip failed ({r.status_code}).")


def transcribe(src, tmp):
    gkey, okey = os.environ.get("GROQ_API_KEY"), os.environ.get("OPENAI_API_KEY")
    if gkey:
        key, url, model = gkey, "https://api.groq.com/openai/v1/audio/transcriptions", os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
    elif okey:
        key, url, model = okey, "https://api.openai.com/v1/audio/transcriptions", "whisper-1"
    else:
        raise RuntimeError("GROQ_API_KEY (free) or OPENAI_API_KEY is not set. Transcription needs a speech-to-text provider key.")
    chunk = 600
    render.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
                "-f", "segment", "-segment_time", str(chunk), "-reset_timestamps", "1", os.path.join(tmp, "a_%03d.mp3")])
    segs, words = [], []
    for i, f in enumerate(sorted(glob.glob(os.path.join(tmp, "a_*.mp3")))):
        with open(f, "rb") as fh:
            r = requests.post(url, timeout=300,
                              headers={"Authorization": f"Bearer {key}"}, files={"file": fh},
                              data=[("model", model), ("response_format", "verbose_json"),
                                    ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")])
        if r.status_code != 200:
            raise RuntimeError(f"Transcription provider returned {r.status_code}.")
        j, off = r.json(), i * chunk
        segs += [{"start": s["start"] + off, "end": s["end"] + off, "text": s["text"].strip()} for s in j.get("segments", [])]
        words += [{"word": w["word"], "start": w["start"] + off, "end": w["end"] + off} for w in j.get("words", [])]
    if not segs:
        raise RuntimeError("No speech was found in this video.")
    return {"segments": segs, "words": words}


def plan_for(user_id):
    p = rest("GET", "cl_profiles", {"id": f"eq.{user_id}", "select": "cl_plans(*)"})
    if not p:
        raise RuntimeError("No profile found for this user.")
    return p[0]["cl_plans"]


def minutes_used(user_id):
    start = dt.datetime.utcnow().replace(day=1, hour=0, minute=0, second=0).isoformat()
    rows = rest("GET", "cl_usage", {"user_id": f"eq.{user_id}", "created_at": f"gte.{start}", "select": "minutes"})
    return sum(float(r["minutes"]) for r in rows)


def run_generation(job, tmp):
    uid = job["user_id"]
    vid = rest("GET", "cl_source_videos", {"id": f"eq.{job['source_video_id']}", "user_id": f"eq.{uid}"})[0]
    if not vid.get("storage_path"):
        raise RuntimeError("This source has no uploaded file. Upload your own video file; "
                           "downloading from YouTube links is not supported.")
    plan = plan_for(uid)
    src = os.path.join(tmp, "src.mp4")
    download(vid["storage_path"], src)
    dur = render.probe_duration(src)
    if minutes_used(uid) + dur / 60 > plan["monthly_minutes"]:
        raise RuntimeError(f"This video would exceed your plan's {plan['monthly_minutes']} processing minutes this month.")
    tr = transcribe(src, tmp)
    rest("PATCH", "cl_source_videos", {"id": f"eq.{vid['id']}"}, {"transcript": tr, "duration_seconds": int(dur)})
    max_s = min(60, plan["max_clip_seconds"])
    clips = highlights.build_clips(tr["segments"], dur, n=3, min_s=min(20, max_s), max_s=max_s)
    if not clips:
        raise RuntimeError("The AI found no suitable moments in this video.")
    rest("DELETE", "cl_clips", {"job_id": f"eq.{job['id']}", "status": "eq.suggested"})
    rest("POST", "cl_clips", [{**c, "user_id": uid, "source_video_id": vid["id"], "job_id": job["id"],
                               "status": "suggested"} for c in clips])
    rest("POST", "cl_usage", {"user_id": uid, "job_id": job["id"], "minutes": round(dur / 60, 2)})
    return {"clips_created": len(clips), "source_seconds": round(dur)}


def run_render(job, tmp):
    uid, clip_id = job["user_id"], (job.get("details") or {}).get("clip_id")
    if not clip_id:
        raise RuntimeError("Render job is missing a clip.")
    clip = rest("GET", "cl_clips", {"id": f"eq.{clip_id}", "user_id": f"eq.{uid}"})[0]
    vid = rest("GET", "cl_source_videos", {"id": f"eq.{clip['source_video_id']}", "user_id": f"eq.{uid}"})[0]
    rest("PATCH", "cl_clips", {"id": f"eq.{clip_id}"}, {"status": "rendering"})
    sets = rest("GET", "cl_caption_settings", {"user_id": f"eq.{uid}", "or": f"(clip_id.eq.{clip_id},clip_id.is.null)",
                                              "order": "clip_id.asc.nullslast"})
    s = {**DEFAULT_CAPTIONS, **(sets[0] if sets else {})}
    plan = plan_for(uid)
    s_, e_ = float(clip["start_seconds"]), float(clip["end_seconds"])
    if e_ - s_ > plan["max_clip_seconds"]:
        raise RuntimeError(f"Clip is longer than your plan allows ({plan['max_clip_seconds']} seconds).")
    src = os.path.join(tmp, "src.mp4")
    download(vid["storage_path"], src)
    if clip.get("caption_text"):
        lines = render.manual_lines(clip["caption_text"], e_ - s_)
    else:
        lines = render.words_to_lines((vid.get("transcript") or {}).get("words", []), s_, e_)
    out = os.path.join(tmp, "out.mp4")
    render.render_clip(src, s_, e_, lines, s, out, tmp, layout=(job.get("details") or {}).get("layout", "blur"))
    path = f"{uid}/clips/{clip_id}.mp4"
    upload(path, out)
    rest("PATCH", "cl_clips", {"id": f"eq.{clip_id}"}, {"status": "ready", "output_path": path})
    return {"output_path": path, "bytes": os.path.getsize(out)}


def process(job):
    tmp = tempfile.mkdtemp(prefix="cl_", dir=os.environ.get("WORK_DIR"))
    try:
        d = run_generation(job, tmp) if job["kind"] == "clip_generation" else run_render(job, tmp)
        rest("PATCH", "cl_jobs", {"id": f"eq.{job['id']}"}, {"status": "completed", "error": None, "details": {**(job.get("details") or {}), **d},
                                                           "updated_at": dt.datetime.utcnow().isoformat()})
        log(f"job {job['id']} completed")
    except Exception as e:  # noqa: BLE001 - report every failure on the job row
        msg = str(e)[:500]
        retry = job["attempts"] < MAX_ATTEMPTS and "is not set" not in msg and "exceed" not in msg
        rest("PATCH", "cl_jobs", {"id": f"eq.{job['id']}"}, {"status": "queued" if retry else "failed", "error": msg,
                                                           "updated_at": dt.datetime.utcnow().isoformat()})
        if job["kind"] == "render" and not retry and (job.get("details") or {}).get("clip_id"):
            rest("PATCH", "cl_clips", {"id": f"eq.{job['details']['clip_id']}"}, {"status": "failed"})
        log(f"job {job['id']} {'will retry' if retry else 'failed'}: {msg}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # always clean temp files


def claim():
    rows = rest("GET", "cl_jobs", {"status": "eq.queued", "order": "created_at.asc", "limit": "1"})
    if not rows:
        return None
    j = rows[0]
    got = rest("PATCH", "cl_jobs", {"id": f"eq.{j['id']}", "status": "eq.queued"},
               {"status": "processing", "attempts": j["attempts"] + 1, "updated_at": dt.datetime.utcnow().isoformat()})
    return got[0] if got else None  # empty means another worker claimed it first


def health_server():
    """Some free hosts need a web port open. Serves a plain OK, nothing else."""
    port = os.environ.get("PORT")
    if not port:
        return
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    class H_(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"cliplane worker ok")
        def log_message(self, *a):
            pass
    threading.Thread(target=HTTPServer(("0.0.0.0", int(port)), H_).serve_forever, daemon=True).start()


def main():
    health_server()
    if not URL or not KEY:
        sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY. See .env.example.")
    stale = (dt.datetime.utcnow() - dt.timedelta(minutes=30)).isoformat()
    rest("PATCH", "cl_jobs", {"status": "eq.processing", "updated_at": f"lt.{stale}"}, {"status": "queued"})
    once = "--once" in sys.argv  # process everything queued, then exit (for scheduled runs)
    log("worker started" + (" (single pass)" if once else ""))
    while True:
        try:
            job = claim()
        except Exception as e:  # noqa: BLE001
            log(f"poll error: {e}")
            job = None
        if job:
            process(job)
        elif once:
            log("queue empty, exiting")
            return
        else:
            time.sleep(POLL)


if __name__ == "__main__":
    main()
