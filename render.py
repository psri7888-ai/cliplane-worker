"""Caption building and FFmpeg rendering. No network access in this module."""
import json, os, subprocess

FONTS = {"Anton", "Bebas Neue", "Montserrat", "Oswald", "Poppins", "Bangers"}
RES = {"1080x1920": (1080, 1920), "720x1280": (720, 1280)}


def run(cmd, cwd=None, timeout=1800):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg/ffprobe failed: " + r.stderr.strip()[-600:])
    return r.stdout


def probe_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path])
    return float(json.loads(out)["format"]["duration"])


def _ass_time(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _ass_color(hex_rgb, alpha=0):
    h = hex_rgb.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _esc(text):
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def words_to_lines(words, clip_start, clip_end, per_line=4):
    """words: [{word,start,end}] in source time -> caption lines in clip time."""
    inside = [w for w in words if w["end"] > clip_start and w["start"] < clip_end]
    lines = []
    for i in range(0, len(inside), per_line):
        grp = inside[i:i + per_line]
        lines.append({
            "start": max(0.0, grp[0]["start"] - clip_start),
            "end": min(clip_end - clip_start, grp[-1]["end"] - clip_start),
            "text": " ".join(w["word"].strip() for w in grp),
        })
    return lines


def manual_lines(text, duration):
    """User-edited caption text: one line per row, spread evenly over the clip."""
    rows = [r.strip() for r in text.splitlines() if r.strip()]
    if not rows:
        return []
    step = duration / len(rows)
    return [{"start": i * step, "end": (i + 1) * step, "text": r} for i, r in enumerate(rows)]


def make_ass(lines, s, w, h):
    font = s.get("font") if s.get("font") in FONTS else "Anton"
    size = int(s.get("size", 30)) * (h / 640)  # editor preview is ~640px tall
    bg = s.get("background", "none")
    boxed = bg != "none"
    text_color = _ass_color(s.get("color", "#ffffff"))
    if boxed and "214,243,75" in bg:
        text_color = _ass_color("#141a33")
    back = _ass_color("#d6f34b", 0x19) if "214,243,75" in bg else _ass_color("#000000", 0x59)
    style = (f"Style: Cap,{font},{size:.0f},{text_color},&H000000FF,&H00000000,{back},"
             f"-1,0,0,0,100,100,0,0,{3 if boxed else 1},{3 if not boxed else 8},0,5,40,40,0,1")
    y = int(h * int(s.get("position_pct", 78)) / 100)
    head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {w}\nPlayResY: {h}\nWrapStyle: 0\n\n"
            "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
            "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
            f"Alignment,MarginL,MarginR,MarginV,Encoding\n{style}\n\n"
            "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    ev = "".join(
        f"Dialogue: 0,{_ass_time(l['start'])},{_ass_time(l['end'])},Cap,,0,0,0,,"
        f"{{\\an5\\pos({w // 2},{y})}}{_esc(l['text'].upper())}\n" for l in lines)
    return head + ev


def render_clip(src, start, end, lines, settings, out_path, workdir, layout="blur"):
    w, h = RES.get(settings.get("resolution", "1080x1920"), RES["1080x1920"])
    ass_name = "caps.ass"
    with open(os.path.join(workdir, ass_name), "w", encoding="utf-8") as f:
        f.write(make_ass(lines, settings, w, h))
    if layout == "crop":
        vf = f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},ass={ass_name}[v]"
    else:  # sharp video centred over a blurred copy of itself
        vf = (f"[0:v]split[a][b];[a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=24:2[bg];"
              f"[b]scale={w}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,ass={ass_name}[v]")
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", os.path.abspath(src),
           "-filter_complex", vf, "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
           os.path.abspath(out_path)]
    run(cmd, cwd=workdir)
    return out_path
