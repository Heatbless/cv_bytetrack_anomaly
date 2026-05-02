"""GUI tool to splice long traffic videos into normal-traffic clips for VAE datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np


@dataclass
class Segment:
    start_frame: int
    end_frame: int

    def normalized(self) -> "Segment":
        if self.start_frame <= self.end_frame:
            return self
        return Segment(self.end_frame, self.start_frame)


class VAESampleSplicerGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VAE Normal Traffic Sample Splicer")
        self.geometry("1200x900")

        self.cap: cv2.VideoCapture | None = None
        self.video_path: Path | None = None
        self.total_frames = 0
        self.fps = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.current_frame = 0

        self.is_playing = False
        self.play_job: str | None = None
        self._ignore_scale_callback = False

        self.start_mark: int | None = None
        self.end_mark: int | None = None
        self.segments: list[Segment] = []

        self.preview_max_width = 960
        self.preview_max_height = 540

        self.video_path_var = tk.StringVar(value="No video loaded")
        self.video_info_var = tk.StringVar(value="Video: -")
        self.frame_info_var = tk.StringVar(value="Frame: -")
        self.marks_var = tk.StringVar(value="Start: - | End: -")
        self.output_dir_var = tk.StringVar(value="")
        self.prefix_var = tk.StringVar(value="normal")
        self.log_window: tk.Toplevel | None = None
        self.log_text: tk.Text | None = None
        self.log_messages: list[str] = []
        self.last_frame_bgr: np.ndarray | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)

        top_split = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        top_split.grid(row=0, column=0, sticky="nsew")

        left_panel = ttk.LabelFrame(top_split, text="Video Preview", padding=8)
        left_panel.grid_columnconfigure(0, weight=1)
        left_panel.grid_rowconfigure(1, weight=1)

        info = ttk.Frame(left_panel)
        info.grid(row=0, column=0, sticky="ew")
        info.grid_columnconfigure(0, weight=1)
        ttk.Label(info, textvariable=self.video_info_var).grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.frame_info_var).grid(row=0, column=1, sticky="e")

        self.preview_label = tk.Label(left_panel, bg="#101010")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        self.preview_label.bind("<Configure>", self._on_preview_resize)

        self.frame_scale = ttk.Scale(
            left_panel,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            command=self._on_seek,
        )
        self.frame_scale.grid(row=2, column=0, sticky="ew")

        right_panel = ttk.LabelFrame(top_split, text="Segments To Export", padding=8)
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(0, weight=1)

        list_wrap = ttk.Frame(right_panel)
        list_wrap.grid(row=0, column=0, sticky="nsew")
        list_wrap.grid_columnconfigure(0, weight=1)
        list_wrap.grid_rowconfigure(0, weight=1)

        self.segment_listbox = tk.Listbox(list_wrap, height=20)
        self.segment_listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(list_wrap, orient=tk.VERTICAL, command=self.segment_listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.segment_listbox.configure(yscrollcommand=scroll.set)

        segment_actions = ttk.Frame(right_panel)
        segment_actions.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(segment_actions, text="Remove Selected", command=self._remove_selected_segment).pack(side=tk.LEFT)
        ttk.Button(segment_actions, text="Clear", command=self._clear_segments).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(segment_actions, text="Save List", command=self._save_segment_list).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(segment_actions, text="Load List", command=self._load_segment_list).pack(side=tk.LEFT, padx=(8, 0))

        top_split.add(left_panel, weight=4)
        top_split.add(right_panel, weight=2)

        controls = ttk.LabelFrame(root, text="Splice Controls", padding=8)
        controls.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        controls.grid_columnconfigure(0, weight=1)

        source_row = ttk.Frame(controls)
        source_row.grid(row=0, column=0, sticky="ew")
        source_row.grid_columnconfigure(1, weight=1)
        ttk.Button(source_row, text="Open Video", command=self._open_video).grid(row=0, column=0, sticky="w")
        ttk.Label(source_row, textvariable=self.video_path_var).grid(row=0, column=1, sticky="w", padx=(10, 10))
        ttk.Button(source_row, text="Show Log Window", command=self._open_log_window).grid(row=0, column=2, sticky="e")

        transport = ttk.Frame(controls)
        transport.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.play_button = ttk.Button(transport, text="Play", command=self._toggle_play)
        self.play_button.pack(side=tk.LEFT)
        ttk.Button(transport, text="-90", command=lambda: self._step_frame(-90)).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(transport, text="-60", command=lambda: self._step_frame(-60)).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(transport, text="-30", command=lambda: self._step_frame(-30)).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(transport, text="-15", command=lambda: self._step_frame(-15)).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(transport, text="+15", command=lambda: self._step_frame(15)).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(transport, text="+30", command=lambda: self._step_frame(30)).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(transport, text="+60", command=lambda: self._step_frame(60)).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(transport, text="+90", command=lambda: self._step_frame(90)).pack(side=tk.LEFT, padx=(4, 0))

        marks = ttk.Frame(controls)
        marks.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(marks, text="Mark Start", command=self._mark_start).pack(side=tk.LEFT)
        ttk.Button(marks, text="Mark End", command=self._mark_end).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(marks, text="Add Segment", command=self._add_segment).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(marks, textvariable=self.marks_var).pack(side=tk.LEFT, padx=(12, 0))

        export_row = ttk.Frame(controls)
        export_row.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        export_row.grid_columnconfigure(1, weight=1)
        ttk.Label(export_row, text="Output Dir:").grid(row=0, column=0, sticky="w")
        ttk.Entry(export_row, textvariable=self.output_dir_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(export_row, text="Browse", command=self._browse_output_dir).grid(row=0, column=2, sticky="e")

        prefix_row = ttk.Frame(controls)
        prefix_row.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(prefix_row, text="File Prefix:").pack(side=tk.LEFT)
        ttk.Entry(prefix_row, textvariable=self.prefix_var, width=20).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(prefix_row, text="Export Segments", command=self._export_segments).pack(side=tk.RIGHT)

    def _log(self, message: str) -> None:
        self.log_messages.append(message)
        if self.log_text is not None and self.log_text.winfo_exists():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

    def _open_log_window(self) -> None:
        if self.log_window is not None and self.log_window.winfo_exists():
            self.log_window.deiconify()
            self.log_window.lift()
            self.log_window.focus_force()
            return

        self.log_window = tk.Toplevel(self)
        self.log_window.title("VAE Splicer Log")
        self.log_window.geometry("900x360")
        self.log_window.protocol("WM_DELETE_WINDOW", self._close_log_window)

        wrap = ttk.Frame(self.log_window, padding=8)
        wrap.pack(fill=tk.BOTH, expand=True)
        wrap.grid_columnconfigure(0, weight=1)
        wrap.grid_rowconfigure(0, weight=1)

        self.log_text = tk.Text(wrap, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

        actions = ttk.Frame(wrap)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT)
        ttk.Button(actions, text="Close", command=self._close_log_window).pack(side=tk.RIGHT)

        self.log_text.configure(state=tk.NORMAL)
        if self.log_messages:
            self.log_text.insert(tk.END, "\n".join(self.log_messages) + "\n")
            self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _close_log_window(self) -> None:
        if self.log_window is not None and self.log_window.winfo_exists():
            self.log_window.destroy()
        self.log_window = None
        self.log_text = None

    def _clear_log(self) -> None:
        self.log_messages.clear()
        if self.log_text is not None and self.log_text.winfo_exists():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.delete("1.0", tk.END)
            self.log_text.configure(state=tk.DISABLED)

    def _on_preview_resize(self, _event: tk.Event) -> None:
        if self.last_frame_bgr is not None:
            self._render_frame(self.last_frame_bgr)

    def _format_time(self, frame_idx: int) -> str:
        if self.fps <= 0:
            return "00:00:00.000"
        total_seconds = frame_idx / self.fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def _update_frame_info(self) -> None:
        if self.total_frames <= 0:
            self.frame_info_var.set("Frame: -")
            return
        self.frame_info_var.set(
            f"Frame {self.current_frame}/{self.total_frames - 1} | Time {self._format_time(self.current_frame)}"
        )

    def _update_marks_label(self) -> None:
        start_text = "-" if self.start_mark is None else f"{self.start_mark} ({self._format_time(self.start_mark)})"
        end_text = "-" if self.end_mark is None else f"{self.end_mark} ({self._format_time(self.end_mark)})"
        self.marks_var.set(f"Start: {start_text} | End: {end_text}")

    def _set_scale_value(self, frame_idx: int) -> None:
        self._ignore_scale_callback = True
        try:
            self.frame_scale.set(frame_idx)
        finally:
            self._ignore_scale_callback = False

    def _open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._load_video(Path(path))

    def _load_video(self, video_path: Path) -> None:
        self._stop_playback()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open video:\n{video_path}")
            return

        self.cap = cap
        self.video_path = video_path

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.current_frame = 0
        self.start_mark = None
        self.end_mark = None
        self.segments.clear()
        self._refresh_segment_list()

        self.video_path_var.set(str(video_path))
        duration_seconds = (self.total_frames / self.fps) if self.fps > 0 else 0.0
        self.video_info_var.set(
            f"FPS: {self.fps:.3f} | Frames: {self.total_frames} | Size: {self.frame_width}x{self.frame_height} | Duration: {duration_seconds/3600:.2f}h"
        )

        self.frame_scale.configure(from_=0, to=max(self.total_frames - 1, 1))
        self._set_scale_value(0)
        self._update_marks_label()
        self._show_frame(0)

        if not self.output_dir_var.get().strip():
            self.output_dir_var.set(str(video_path.parent / "vae_normal_samples"))

        self._log(f"Loaded video: {video_path}")

    def _on_seek(self, value: str) -> None:
        if self._ignore_scale_callback:
            return
        if self.cap is None:
            return
        frame_idx = int(float(value))
        if frame_idx == self.current_frame:
            return
        self._show_frame(frame_idx)

    def _show_frame(self, frame_idx: int) -> None:
        if self.cap is None:
            return

        if self.total_frames > 0:
            frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        else:
            frame_idx = max(0, frame_idx)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            return

        self.current_frame = frame_idx
        self.last_frame_bgr = frame.copy()
        self._set_scale_value(frame_idx)
        self._render_frame(frame)
        self._update_frame_info()

    def _render_frame(self, frame_bgr: np.ndarray) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        max_w = self.preview_label.winfo_width() - 4
        max_h = self.preview_label.winfo_height() - 4
        if max_w < 32 or max_h < 32:
            max_w = self.preview_max_width
            max_h = self.preview_max_height

        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        ppm_header = f"P6 {w} {h} 255 ".encode("ascii")
        ppm_data = ppm_header + frame_rgb.tobytes()
        photo = tk.PhotoImage(data=ppm_data, format="PPM")

        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

    def _toggle_play(self) -> None:
        if self.cap is None:
            return
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        self.is_playing = True
        self.play_button.configure(text="Pause")
        self._play_tick()

    def _stop_playback(self) -> None:
        self.is_playing = False
        self.play_button.configure(text="Play")
        if self.play_job is not None:
            self.after_cancel(self.play_job)
            self.play_job = None

    def _play_tick(self) -> None:
        if not self.is_playing:
            return

        next_frame = self.current_frame + 1
        if self.total_frames > 0 and next_frame >= self.total_frames:
            self._stop_playback()
            return

        self._show_frame(next_frame)
        delay_ms = max(1, int(1000 / max(self.fps, 1.0)))
        self.play_job = self.after(delay_ms, self._play_tick)

    def _step_frame(self, delta: int) -> None:
        if self.cap is None:
            return
        self._stop_playback()
        self._show_frame(self.current_frame + delta)

    def _mark_start(self) -> None:
        if self.cap is None:
            return
        self.start_mark = self.current_frame
        self._update_marks_label()
        self._log(f"Start marked at frame {self.start_mark}")

    def _mark_end(self) -> None:
        if self.cap is None:
            return
        self.end_mark = self.current_frame
        self._update_marks_label()
        self._log(f"End marked at frame {self.end_mark}")

    def _add_segment(self) -> None:
        if self.start_mark is None or self.end_mark is None:
            messagebox.showwarning("Marks missing", "Mark both start and end frames before adding a segment.")
            return

        seg = Segment(self.start_mark, self.end_mark).normalized()
        self.segments.append(seg)
        self.segments.sort(key=lambda s: (s.start_frame, s.end_frame))
        self._refresh_segment_list()
        self._log(f"Added segment: {seg.start_frame} -> {seg.end_frame}")

    def _refresh_segment_list(self) -> None:
        self.segment_listbox.delete(0, tk.END)
        for idx, seg in enumerate(self.segments, start=1):
            duration = (seg.end_frame - seg.start_frame + 1) / max(self.fps, 1.0)
            text = (
                f"{idx:03d} | f{seg.start_frame}-{seg.end_frame} | "
                f"{self._format_time(seg.start_frame)} -> {self._format_time(seg.end_frame)} | {duration:.2f}s"
            )
            self.segment_listbox.insert(tk.END, text)

    def _remove_selected_segment(self) -> None:
        selection = self.segment_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        removed = self.segments.pop(index)
        self._refresh_segment_list()
        self._log(f"Removed segment: {removed.start_frame} -> {removed.end_frame}")

    def _clear_segments(self) -> None:
        if not self.segments:
            return
        if not messagebox.askyesno("Confirm", "Clear all segments?"):
            return
        self.segments.clear()
        self._refresh_segment_list()
        self._log("Cleared all segments")

    def _save_segment_list(self) -> None:
        if self.video_path is None:
            return

        out_path = filedialog.asksaveasfilename(
            title="Save segment list",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="vae_segment_list.json",
        )
        if not out_path:
            return

        payload = {
            "video_path": str(self.video_path),
            "fps": self.fps,
            "total_frames": self.total_frames,
            "segments": [
                {"start_frame": s.start_frame, "end_frame": s.end_frame}
                for s in self.segments
            ],
        }
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log(f"Saved segment list: {out_path}")

    def _load_segment_list(self) -> None:
        in_path = filedialog.askopenfilename(
            title="Load segment list",
            filetypes=[("JSON", "*.json")],
        )
        if not in_path:
            return

        try:
            payload = json.loads(Path(in_path).read_text(encoding="utf-8"))
            loaded = []
            for item in payload.get("segments", []):
                seg = Segment(int(item["start_frame"]), int(item["end_frame"])).normalized()
                loaded.append(seg)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Could not read segment list:\n{exc}")
            return

        self.segments = loaded
        self.segments.sort(key=lambda s: (s.start_frame, s.end_frame))
        self._refresh_segment_list()
        self._log(f"Loaded {len(self.segments)} segments from {in_path}")

    def _browse_output_dir(self) -> None:
        chosen = filedialog.askdirectory(title="Select output folder")
        if chosen:
            self.output_dir_var.set(chosen)

    def _export_segments(self) -> None:
        if self.video_path is None:
            messagebox.showwarning("No video", "Load a video first.")
            return
        if not self.segments:
            messagebox.showwarning("No segments", "Add at least one segment first.")
            return

        output_dir = Path(self.output_dir_var.get().strip() or self.video_path.parent / "vae_normal_samples")
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.prefix_var.get().strip() or "normal"
        prefix = "".join(ch for ch in prefix if ch.isalnum() or ch in ("-", "_"))
        if not prefix:
            prefix = "normal"

        manifest_path = output_dir / f"{prefix}_manifest.csv"

        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "clip_path",
                    "start_frame",
                    "end_frame",
                    "start_sec",
                    "end_sec",
                    "duration_sec",
                ],
            )
            writer.writeheader()

            for idx, seg in enumerate(self.segments, start=1):
                out_name = f"{prefix}_{idx:03d}_f{seg.start_frame}-{seg.end_frame}.mp4"
                out_path = output_dir / out_name
                self._write_segment(seg, out_path)

                start_sec = seg.start_frame / max(self.fps, 1.0)
                end_sec = seg.end_frame / max(self.fps, 1.0)
                duration_sec = (seg.end_frame - seg.start_frame + 1) / max(self.fps, 1.0)
                writer.writerow(
                    {
                        "clip_path": str(out_path),
                        "start_frame": seg.start_frame,
                        "end_frame": seg.end_frame,
                        "start_sec": f"{start_sec:.6f}",
                        "end_sec": f"{end_sec:.6f}",
                        "duration_sec": f"{duration_sec:.6f}",
                    }
                )

                self._log(f"Exported {out_name}")
                self.update_idletasks()

        messagebox.showinfo("Done", f"Exported {len(self.segments)} clips to:\n{output_dir}")
        self._log(f"Manifest written: {manifest_path}")

    def _write_segment(self, seg: Segment, output_path: Path) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen video: {self.video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, seg.start_frame)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.frame_width, self.frame_height))

        frames_to_write = seg.end_frame - seg.start_frame + 1
        for _ in range(max(0, frames_to_write)):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)

        writer.release()
        cap.release()

    def _on_close(self) -> None:
        self._stop_playback()
        self._close_log_window()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.destroy()


def main() -> None:
    app = VAESampleSplicerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
