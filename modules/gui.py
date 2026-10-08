"""Tkinter GUI for Boxxy.

Tkinter is not thread-safe, so all long-running work (model loading, LLM
generation, Playwright browsing) runs in worker threads and communicates with
the UI exclusively through a queue that the Tk main loop polls.
"""

import queue
import re
import sys
import threading
import time
import traceback
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from modules import logger
from modules.models import DEFAULT_MODELS_DIR, discover_models, validate_model
from modules.window_layout import desktop_bounds

DEFAULT_TEST_URL = "https://forms-staging.cdssandbox.xyz/en/form-builder"

HAIKU_PROMPT = [
    {"role": "system", "content": "You are a poet. Reply with only the poem, no title or commentary."},
    {"role": "user", "content": "Write a haiku (5-7-5 syllables, three lines) about black box software testing."},
]

POLL_INTERVAL_MS = 100


def parse_aria_snapshot(snapshot):
    """Convert Playwright's YAML-like ARIA snapshot into nested {label, children} nodes."""
    root = []
    stack = [(-1, root)]

    for raw_line in (snapshot or "").splitlines():
        if not raw_line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        text = raw_line.strip()
        if text.startswith("- "):
            text = text[2:]
        elif text == "-":
            continue
        if text.endswith(":"):
            text = text[:-1]

        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()

        node = {"label": text, "children": []}
        stack[-1][1].append(node)
        stack.append((indent, node["children"]))

    return root


class BoxxyGui:
    def __init__(self, test_url=DEFAULT_TEST_URL, model_path=None, models_dir=DEFAULT_MODELS_DIR):
        self.model_path = model_path
        self.models_dir = Path(models_dir).resolve()
        self.test_url = test_url
        self.main_open = False
        self.events = queue.Queue()
        self.llm = None
        self.model_loading = False
        self.run_thread = None
        self.stop_event = threading.Event()
        self.closed = False

        self.root = tk.Tk()
        self.root.title("Boxxy - Select a model")
        self.gui_bounds, self.browser_bounds = desktop_bounds(
            self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        ).split()
        self._position_window()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        # Exceptions raised inside Tk callbacks are shown in the GUI instead of stderr.
        self.root.report_callback_exception = self._on_tk_exception

        self._build_model_selector()

    def _position_window(self):
        bounds = self.gui_bounds
        self.root.geometry(f"{bounds.width}x{bounds.height}+{bounds.x}+{bounds.y}")
        self.root.update_idletasks()
        title_height = max(0, self.root.winfo_rooty() - self.root.winfo_y())
        height = max(1, bounds.height - title_height)
        self.root.minsize(min(600, bounds.width), min(350, height))
        self.root.geometry(f"{bounds.width}x{height}+{bounds.x}+{bounds.y}")

    def _build_model_selector(self):
        self.selector = ttk.Frame(self.root, padding=20)
        self.selector.pack(fill="both", expand=True)
        ttk.Label(self.selector, text="Select a local model", font=("TkDefaultFont", 16, "bold")).pack(
            anchor="w"
        )
        ttk.Label(self.selector, text=f"Models folder: {self.models_dir}", wraplength=640).pack(
            anchor="w", pady=(8, 16)
        )
        self.model_var = tk.StringVar()
        self.model_picker = ttk.Combobox(self.selector, textvariable=self.model_var, state="readonly")
        self.model_picker.pack(fill="x")
        self.selection_status = tk.StringVar()
        ttk.Label(
            self.selector, textvariable=self.selection_status, wraplength=640, justify="left"
        ).pack(anchor="w", pady=12)
        self.selection_details = self._readonly_text(self.selector, height=6)
        self.selection_details.pack(fill="both", expand=True)
        controls = ttk.Frame(self.selector)
        controls.pack(fill="x", pady=(12, 0))
        ttk.Button(controls, text="Refresh", command=self._refresh_models).pack(side="left")
        ttk.Button(controls, text="Cancel", command=self.on_close).pack(side="right")
        self.open_button = ttk.Button(controls, text="Open Boxxy", command=self._open_main)
        self.open_button.pack(side="right", padx=8)
        self._refresh_models()

    def _refresh_models(self):
        selected = self.model_var.get()
        self.available_models = {}
        try:
            models, rejected = discover_models(self.models_dir)
        except OSError as error:
            self.selection_status.set(f"Unable to read the models folder: {error}")
            self._set_text(self.selection_details, "Create the models folder and download a model, then Refresh.")
        else:
            self.available_models = {str(path.relative_to(self.models_dir)): path for path in models}
            self.selection_status.set(
                f"Found {len(models)} local model(s). Select one to continue."
                if models else "No complete local models found. Download a model, then Refresh."
            )
            self._set_text(
                self.selection_details,
                "Skipped folders:\n" + "\n".join(rejected) if rejected
                else "Local files checked. Compatibility is verified when the selected model loads.",
            )
        names = list(self.available_models)
        self.model_picker.configure(values=names)
        preferred = ""
        if self.model_path:
            preferred_path = Path(self.model_path).resolve()
            if preferred_path.is_relative_to(self.models_dir):
                preferred = str(preferred_path.relative_to(self.models_dir))
        self.model_var.set(selected if selected in names else preferred if preferred in names else names[0] if names else "")
        self.open_button.configure(state="normal" if names else "disabled")

    def _open_main(self):
        if self.main_open:
            return
        path = self.available_models.get(self.model_var.get())
        if path is None:
            self._show_error("Invalid model", "Select a complete model from the list.")
            return
        try:
            validate_model(path)
        except (OSError, ValueError) as error:
            self._refresh_models()
            self._show_error("Invalid model", str(error))
            return
        self.model_path = str(path)
        self.selector.destroy()
        self.root.title(f"Boxxy - {path.name}")
        self._position_window()
        self._build_toolbar(self.test_url)
        self._build_error_bar()
        self._build_status_bar()
        self._build_main_panes()

        self.main_open = True
        logger.add_listener(self._on_log)
        self._install_exception_hooks()
        self.root.after(0, self._poll_events)
        self.root.after(50, self.load_model_async)

    # ------------------------------------------------------------------ layout

    def _build_toolbar(self, test_url):
        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(side="top", fill="x")

        ttk.Label(toolbar, text="Mode 1 URL:").pack(side="left")
        self.url_var = tk.StringVar(value=test_url)
        self.url_entry = ttk.Entry(toolbar, textvariable=self.url_var)
        self.url_entry.pack(side="left", fill="x", expand=True, padx=(4, 8))

        self.start_button = ttk.Button(toolbar, text="Start", command=self.start_run, state="disabled")
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(toolbar, text="Stop", command=self.stop_run, state="disabled")
        self.stop_button.pack(side="left", padx=(4, 0))
        self.reload_button = ttk.Button(toolbar, text="Retry Model Load", command=self.load_model_async)
        # Only shown if model loading fails.

    def _build_error_bar(self):
        self.error_bar = tk.Frame(self.root, bg="#fde2e1", padx=8, pady=4)
        self.error_label = tk.Label(
            self.error_bar, text="", bg="#fde2e1", fg="#9b1c1c", anchor="w", justify="left"
        )
        self.error_label.pack(side="left", fill="x", expand=True)
        tk.Button(self.error_bar, text="Dismiss", command=self._hide_error).pack(side="right")
        # Packed on demand by _show_error.

    def _build_status_bar(self):
        bar = ttk.Frame(self.root, padding=(8, 4), relief="sunken")
        bar.pack(side="bottom", fill="x")

        self.status_var = tk.StringVar(value="Starting…")
        ttk.Label(bar, textvariable=self.status_var, anchor="w").pack(side="top", fill="x")

        self.haiku_var = tk.StringVar(value="")
        self.haiku_label = ttk.Label(
            bar, textvariable=self.haiku_var, anchor="w", justify="left", font=("TkDefaultFont", 12, "italic")
        )
        self.haiku_label.pack(side="top", fill="x")

        self.timing_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.timing_var, anchor="w", foreground="#555").pack(side="top", fill="x")

    def _build_main_panes(self):
        panes = ttk.PanedWindow(self.root, orient="horizontal")
        panes.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))

        # Left: live ARIA tree.
        tree_frame = ttk.Labelframe(panes, text="Active ARIA State (Mode 1)", padding=4)
        self.tree_updated_var = tk.StringVar(value="No snapshot yet.")
        ttk.Label(tree_frame, textvariable=self.tree_updated_var, foreground="#555").pack(side="top", anchor="w")

        tree_inner = ttk.Frame(tree_frame)
        tree_inner.pack(side="top", fill="both", expand=True)
        self.tree = ttk.Treeview(tree_inner, show="tree")
        tree_y = ttk.Scrollbar(tree_inner, orient="vertical", command=self.tree.yview)
        tree_x = ttk.Scrollbar(tree_inner, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_y.set, xscrollcommand=tree_x.set)
        self.tree.column("#0", width=600, stretch=True)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_y.grid(row=0, column=1, sticky="ns")
        tree_x.grid(row=1, column=0, sticky="ew")
        tree_inner.rowconfigure(0, weight=1)
        tree_inner.columnconfigure(0, weight=1)
        panes.add(tree_frame, weight=1)

        # Right: progress block on top, then the activity log / instructions view below.
        right = ttk.PanedWindow(panes, orient="vertical")
        panes.add(right, weight=1)

        progress_frame = ttk.Labelframe(right, text="Boxxy Progress", padding=4)
        self.progress_updated_var = tk.StringVar(value="No progress yet.")
        ttk.Label(progress_frame, textvariable=self.progress_updated_var, foreground="#555").pack(
            side="top", anchor="w"
        )
        self.progress_text = self._readonly_text(progress_frame, height=12)
        self.progress_text.tag_configure("heading", font=("TkDefaultFont", 12, "bold"))
        self.progress_text.tag_configure("done", foreground="#15803d")
        self.progress_text.tag_configure("todo", foreground="#92400e")
        self.progress_text.tag_configure("failure", foreground="#b91c1c")
        self.progress_text.pack(fill="both", expand=True)
        right.add(progress_frame, weight=1)

        output_frame = ttk.Labelframe(right, text="Boxxy Output", padding=4)
        controls = ttk.Frame(output_frame)
        controls.pack(side="top", fill="x", pady=(0, 4))
        self.output_view_var = tk.StringVar(value="log")
        for label, value in (("Activity Log", "log"), ("Current Instructions", "prompt")):
            ttk.Radiobutton(
                controls, text=label, value=value, variable=self.output_view_var, command=self._show_output_view
            ).pack(side="left", padx=(0, 8))
        self.prompt_updated_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.prompt_updated_var, foreground="#555").pack(side="right")

        self.output_body = ttk.Frame(output_frame)
        self.output_body.pack(side="top", fill="both", expand=True)

        self.log_view = ttk.Frame(self.output_body)
        self.log_text = self._readonly_text(self.log_view, height=10)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_configure("time", foreground="#888")
        self.log_text.tag_configure("error", foreground="#b91c1c")
        self.log_text.tag_configure("separator", foreground="#aaa")
        ttk.Button(self.log_view, text="Clear Log", command=self._clear_log).pack(
            side="bottom", anchor="e", pady=(4, 0)
        )

        self.prompt_view = ttk.Frame(self.output_body)
        self.prompt_text = self._readonly_text(self.prompt_view, height=10)
        self.prompt_text.tag_configure("role", foreground="#1d4ed8", font=("TkDefaultFont", 11, "bold"))
        self.prompt_text.pack(fill="both", expand=True)
        self._set_text(self.prompt_text, "No instructions have been sent to the model yet.")

        self._show_output_view()
        right.add(output_frame, weight=2)
        self.root.after_idle(lambda: panes.sashpos(0, panes.winfo_width() // 2))

    @staticmethod
    def _readonly_text(parent, height):
        return ScrolledText(parent, wrap="word", state="disabled", height=height)

    def _show_output_view(self):
        show, hide = (
            (self.prompt_view, self.log_view) if self.output_view_var.get() == "prompt"
            else (self.log_view, self.prompt_view)
        )
        hide.pack_forget()
        show.pack(fill="both", expand=True)

    # --------------------------------------------------------- error handling

    def _install_exception_hooks(self):
        def thread_hook(args):
            if args.exc_type is SystemExit:
                return
            text = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
            self.post("error", f"Unhandled error in thread {getattr(args.thread, 'name', '?')}", text)

        def sys_hook(exc_type, exc_value, exc_tb):
            text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            self.post("error", "Unhandled error", text)

        threading.excepthook = thread_hook
        sys.excepthook = sys_hook

    def _on_tk_exception(self, exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        self._show_error("GUI error", text)

    def _show_error(self, title, details):
        if not self.main_open:
            messagebox.showerror(title, details, parent=self.root)
            return
        try:
            summary = (details or "").strip().splitlines()
            last_line = summary[-1] if summary else ""
            self.error_label.configure(text=f"⚠ {title}: {last_line}"[:500])
            if not self.error_bar.winfo_ismapped():
                self.error_bar.pack(side="top", fill="x", after=self.url_entry.master)
            self._append_log(f"ERROR — {title}\n{details}", tag="error")
        except Exception:
            # Last resort: never let error display itself take down the GUI.
            traceback.print_exc()

    def _hide_error(self):
        self.error_bar.pack_forget()

    # ------------------------------------------------------- thread ↔ UI glue

    def post(self, kind, *payload):
        """Thread-safe: enqueue an event for the UI thread."""
        self.events.put((kind, payload))

    def _on_log(self, message, clear_screen):
        self.post("log", message, clear_screen)

    def _poll_events(self):
        try:
            while True:
                try:
                    kind, payload = self.events.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle_event(kind, payload)
                except Exception:
                    self._show_error(f"Failed to handle '{kind}' event", traceback.format_exc())
        finally:
            if not self.closed:
                self.root.after(POLL_INTERVAL_MS, self._poll_events)

    def _handle_event(self, kind, payload):
        if kind == "log":
            message, clear_screen = payload
            if clear_screen:
                self._append_log("─" * 40, tag="separator")
            self._append_log(message)
        elif kind == "status":
            self.status_var.set(payload[0])
        elif kind == "snapshot":
            self._update_tree(payload[0])
        elif kind == "progress":
            self._update_progress(payload[0])
        elif kind == "prompt":
            self._update_prompt(payload[0])
        elif kind == "error":
            self._show_error(*payload)
        elif kind == "model_loaded":
            llm, load_seconds = payload
            self.llm = llm
            self.model_loading = False
            self.reload_button.pack_forget()
            self.timing_var.set(f"Model loaded in {load_seconds:.2f}s. Generating readiness haiku…")
            self._refresh_buttons()
        elif kind == "model_failed":
            self.model_loading = False
            self.status_var.set("Model failed to load.")
            self.reload_button.pack(side="left", padx=(4, 0))
            self._refresh_buttons()
        elif kind == "haiku":
            haiku, load_seconds, gen_seconds = payload
            self.haiku_var.set(haiku)
            self.timing_var.set(
                f"Model loaded in {load_seconds:.2f}s · Haiku generated in {gen_seconds:.2f}s (after load)"
            )
            self.status_var.set("Boxxy is ready.")
        elif kind == "run_finished":
            self.run_thread = None
            self._refresh_buttons()

    # ----------------------------------------------------------- UI helpers

    def _append_log(self, message, tag=None):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{stamp}] ", ("time",))
        self.log_text.insert("end", f"{message}\n", (tag,) if tag else ())
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    @staticmethod
    def _set_text(widget, segments):
        """Replace a read-only text widget's contents, keeping the scroll position. segments: str or [(text, tag)]."""
        if isinstance(segments, str):
            segments = [(segments, None)]
        top = widget.yview()[0]
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        for text, tag in segments:
            widget.insert("end", text, (tag,) if tag else ())
        widget.configure(state="disabled")
        widget.yview_moveto(top)

    @staticmethod
    def _progress_line_tag(line):
        if line.startswith("[x]"):
            return "done"
        if line.startswith("[ ]"):
            return "todo"
        if line.startswith("On "):
            return "failure"
        if line.startswith(("Progress so far", "Goal:")) or line.endswith(":") or re.match(r"Step \d+ on ", line):
            return "heading"
        return None

    def _update_progress(self, progress):
        segments = [(line + "\n", self._progress_line_tag(line)) for line in (progress or "").splitlines()]
        self._set_text(self.progress_text, segments)
        self.progress_updated_var.set(f"Last updated {datetime.now().strftime('%H:%M:%S')}")

    def _update_prompt(self, messages):
        segments = []
        for index, message in enumerate(messages or [], start=1):
            role = message.get("role", "?") if isinstance(message, dict) else "?"
            content = message.get("content", "") if isinstance(message, dict) else message
            segments.append((f"── {index}. {role} ──\n", "role"))
            segments.append((f"{content}\n\n", None))
        self._set_text(self.prompt_text, segments)
        self.prompt_updated_var.set(
            f"{len(messages or [])} messages · sent {datetime.now().strftime('%H:%M:%S')}"
        )

    def _update_tree(self, snapshot):
        self.tree.delete(*self.tree.get_children())

        def insert(parent, nodes):
            for node in nodes:
                item = self.tree.insert(parent, "end", text=node["label"], open=True)
                insert(item, node["children"])

        insert("", parse_aria_snapshot(snapshot))
        self.tree_updated_var.set(f"Last updated {datetime.now().strftime('%H:%M:%S')}")

    def _refresh_buttons(self):
        running = self.run_thread is not None
        self.start_button.configure(state="normal" if self.llm is not None and not running else "disabled")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.url_entry.configure(state="disabled" if running else "normal")

    # ------------------------------------------------------------ workers

    def load_model_async(self):
        if self.model_loading or self.llm is not None:
            return
        self.model_loading = True
        self.reload_button.pack_forget()
        self.status_var.set(f"Loading model from {self.model_path}…")
        self.haiku_var.set("")
        self.timing_var.set("")
        threading.Thread(target=self._load_model_worker, name="boxxy-model-loader", daemon=True).start()

    def _load_model_worker(self):
        try:
            from modules.llm import LLM

            load_start = time.perf_counter()
            llm = LLM()
            llm.load_model(self.model_path)
            load_seconds = time.perf_counter() - load_start
        except Exception:
            self.post("error", "Model load failed", traceback.format_exc())
            self.post("model_failed")
            return

        self.post("model_loaded", llm, load_seconds)
        self.post("status", "Model loaded. Asking Boxxy for a haiku…")

        try:
            gen_start = time.perf_counter()
            response = llm.gen_text(HAIKU_PROMPT)
            gen_seconds = time.perf_counter() - gen_start
            haiku = (response.get("text") or "").strip() or "(the model returned an empty haiku)"
            self.post("haiku", haiku, load_seconds, gen_seconds)
        except Exception:
            self.post("error", "Haiku generation failed", traceback.format_exc())
            self.post("status", "Model loaded, but the readiness haiku failed.")

    def start_run(self):
        if self.llm is None or self.run_thread is not None:
            return
        url = self.url_var.get().strip()
        if not url:
            self._show_error("Invalid URL", "Please enter a URL for Boxxy to explore.")
            return

        self._hide_error()
        self.stop_event.clear()
        self.run_thread = threading.Thread(
            target=self._run_worker, args=(url,), name="boxxy-mode1", daemon=True
        )
        self._refresh_buttons()
        self.status_var.set(f"Boxxy is exploring {url} …")
        self.run_thread.start()

    def stop_run(self):
        if self.run_thread is None:
            return
        self.stop_event.set()
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopping after the current step…")

    def _run_worker(self, url):
        from modules.mode1 import Mode1, StopRequested

        boxxy = None
        try:
            logger.log(f"Starting Mode 1 on {url}")
            boxxy = Mode1(
                url,
                llm=self.llm,
                on_snapshot=lambda snapshot: self.post("snapshot", snapshot),
                on_progress=lambda progress: self.post("progress", progress),
                # Copy so later mutation in the worker can't race with the UI thread rendering it.
                on_prompt=lambda messages: self.post("prompt", [dict(message) for message in messages]),
                stop_event=self.stop_event,
                browser_bounds=self.browser_bounds,
            )
            boxxy.launch()
            boxxy.explore()
            self.post("status", "Boxxy finished exploring.")
        except StopRequested:
            logger.log("Boxxy stopped by user.")
            self.post("status", "Boxxy stopped.")
        except RecursionError:
            self.post("error", "Boxxy explored too many steps", traceback.format_exc())
            self.post("status", "Boxxy stopped due to an error.")
        except Exception:
            self.post("error", "Boxxy run failed", traceback.format_exc())
            self.post("status", "Boxxy stopped due to an error.")
        finally:
            if boxxy is not None:
                try:
                    boxxy.close()
                except Exception:
                    self.post("error", "Failed to close the browser", traceback.format_exc())
            self.post("run_finished")

    # ------------------------------------------------------------ lifecycle

    def on_close(self):
        self.closed = True
        self.stop_event.set()
        logger.remove_listener(self._on_log)
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        while not self.closed:
            try:
                self.root.mainloop()
                break
            except KeyboardInterrupt:
                self.on_close()
            except Exception:
                # Keep the window alive; report and resume the event loop.
                self._show_error("GUI error", traceback.format_exc())
