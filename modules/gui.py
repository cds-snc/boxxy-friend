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
from dataclasses import asdict
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from modules import logger
from modules.models import DEFAULT_MODELS_DIR, discover_models, validate_model
from modules.window_layout import desktop_bounds
from modules.objective import Objective, refine_objective

DEFAULT_TEST_URL = Objective().url

HAIKU_PROMPT = [
    {"role": "system", "content": "You are a poet. Reply with only the poem, no title or commentary."},
    {"role": "user", "content": "Write a haiku (5-7-5 syllables, three lines) about black box software testing."},
]

POLL_INTERVAL_MS = 100
# Initial guesses only: neither backend exposes granular readiness progress.
ESTIMATED_LOAD_SECONDS = 60
ESTIMATED_HAIKU_SECONDS = 30


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
        self.loading_screen = None
        self.loading_ready = False
        self.run_thread = None
        self.stop_event = threading.Event()
        self.closed = False
        self.approved_objective = None
        self.refining = False
        self.discussion = []
        self.proposal = None

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
        self.root.after(0, self._poll_events)

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
        if self.main_open or self.loading_screen is not None:
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
        self._build_loading_screen()
        self._install_exception_hooks()
        self.root.after(50, self.load_model_async)

    def _build_loading_screen(self):
        self.root.title(f"Boxxy - Preparing {Path(self.model_path).name}")
        self.loading_screen = ttk.Frame(self.root, padding=20)
        self.loading_screen.pack(fill="both", expand=True)
        ttk.Label(
            self.loading_screen, text="Preparing your model", font=("TkDefaultFont", 16, "bold")
        ).pack(anchor="w")
        ttk.Label(
            self.loading_screen, text=Path(self.model_path).name, wraplength=640
        ).pack(anchor="w", pady=(8, 16))
        self.loading_status_var = tk.StringVar(value="Getting ready to load...")
        ttk.Label(
            self.loading_screen, textvariable=self.loading_status_var, wraplength=640
        ).pack(anchor="w")
        self.loading_progress = ttk.Progressbar(self.loading_screen, maximum=100)
        self.loading_progress.pack(fill="x", pady=12)
        self.loading_time_var = tk.StringVar()
        ttk.Label(
            self.loading_screen, textvariable=self.loading_time_var, wraplength=640
        ).pack(anchor="w")
        ttk.Label(
            self.loading_screen,
            text="Progress and time remaining are approximate, not backend measurements. "
                 "Initial estimates: 60s to load, 30s for the haiku; hardware and model size vary.",
            wraplength=640, foreground="#555",
        ).pack(anchor="w", pady=(4, 12))
        results = ttk.Labelframe(self.loading_screen, text="Readiness haiku and benchmark", padding=8)
        results.pack(fill="both", expand=True)
        self.loading_results = self._readonly_text(results, height=10)
        self.loading_results.pack(fill="both", expand=True)
        controls = ttk.Frame(self.loading_screen)
        controls.pack(fill="x", pady=(12, 0))
        self.retry_loading_button = ttk.Button(
            controls, text="Retry", command=self.load_model_async, state="disabled"
        )
        self.retry_loading_button.pack(side="left")
        self.choose_model_button = ttk.Button(
            controls, text="Choose another model", command=self._return_to_selector, state="disabled"
        )
        self.choose_model_button.pack(side="left", padx=8)
        ttk.Button(controls, text="Cancel", command=self.on_close).pack(side="right")
        self.loading_ok_button = ttk.Button(
            controls, text="Okay", command=self._enter_main, state="disabled"
        )
        self.loading_ok_button.pack(side="right", padx=8)

    def _return_to_selector(self):
        if self.model_loading or self.loading_ready:
            return
        self.loading_screen.destroy()
        self.loading_screen = None
        self.root.title("Boxxy - Select a model")
        self._build_model_selector()

    def _enter_main(self):
        if self.main_open or not self.loading_ready or self.model_loading or self.closed:
            return
        self.loading_screen.destroy()
        self.loading_screen = None
        self.root.title(f"Boxxy - {Path(self.model_path).name}")
        self._position_window()
        self._build_toolbar(self.test_url)
        self._build_error_bar()
        self._build_status_bar()
        self._build_main_panes()

        self.main_open = True
        logger.add_listener(self._on_log)
        self._refresh_buttons()

    # ------------------------------------------------------------------ layout

    def _build_toolbar(self, test_url):
        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(side="top", fill="x")

        ttk.Label(toolbar, text="Scope URL:").pack(side="left")
        self.url_var = tk.StringVar(value=test_url)
        self.url_entry = ttk.Entry(toolbar, textvariable=self.url_var)
        self.url_entry.pack(side="left", fill="x", expand=True, padx=(4, 8))
        self.url_var.trace_add("write", self._objective_changed)

        self.start_button = ttk.Button(toolbar, text="Start", command=self.start_run, state="disabled")
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(toolbar, text="Stop", command=self.stop_run, state="disabled")
        self.stop_button.pack(side="left", padx=(4, 0))

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

        self.status_var = tk.StringVar(value="Boxxy is ready.")
        ttk.Label(bar, textvariable=self.status_var, anchor="w").pack(side="top", fill="x")

    def _build_main_panes(self):
        self.workspace = ttk.Notebook(self.root)
        self.workspace.pack(fill="both", expand=True)
        self.definition_tab = ttk.Frame(self.workspace, padding=8)
        self.execution_tab = ttk.Frame(self.workspace)
        self.workspace.add(self.definition_tab, text="Define smoke test")
        self.workspace.add(self.execution_tab, text="Live run")
        self._build_objective_editor()
        panes = ttk.PanedWindow(self.execution_tab, orient="horizontal")
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

    def _build_objective_editor(self):
        self.approved_objective = None
        self.refining = False
        self.discussion = []
        self.proposal = None
        self.objective_vars = {}
        self.objective_entries = []
        fields = ttk.Frame(self.definition_tab)
        fields.pack(fill="x")
        defaults = asdict(Objective())
        for row, (key, label) in enumerate((
            ("title", "Form title"), ("element_count", "Input elements (1-10)"),
            ("max_actions", "Maximum browser actions"), ("max_seconds", "Maximum seconds"),
            ("no_progress_turns", "Consecutive no-progress turns"),
        )):
            ttk.Label(fields, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8))
            variable = tk.StringVar(value=str(defaults[key]))
            entry = ttk.Entry(fields, textvariable=variable)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            variable.trace_add("write", self._objective_changed)
            self.objective_vars[key] = variable
            self.objective_entries.append(entry)
        fields.columnconfigure(1, weight=1)
        self.objective_status = tk.StringVar(value="Review and approve the objective before starting.")
        ttk.Label(self.definition_tab, textvariable=self.objective_status, wraplength=640).pack(
            fill="x", pady=4
        )
        self.objective_preview = self._readonly_text(self.definition_tab, height=10)
        self.objective_preview.pack(fill="both", expand=True)
        self.discussion_text = self._readonly_text(self.definition_tab, height=5)
        self.discussion_text.pack(fill="both", expand=True, pady=4)
        self._set_text(self.discussion_text, "Discuss a change, then review and apply the proposal. "
                       "Types cycle: short answer, long answer, radio buttons. First input is required.")
        controls = ttk.Frame(self.definition_tab)
        controls.pack(fill="x")
        self.request_var = tk.StringVar()
        self.request_entry = ttk.Entry(controls, textvariable=self.request_var)
        self.request_entry.pack(side="left", fill="x", expand=True)
        self.refine_button = ttk.Button(controls, text="Discuss", command=self.discuss_objective)
        self.refine_button.pack(side="left", padx=4)
        self.apply_button = ttk.Button(
            controls, text="Apply proposal", command=self.apply_proposal, state="disabled"
        )
        self.apply_button.pack(side="left")
        self.approve_button = ttk.Button(
            controls, text="Approve objective", command=self.approve_objective
        )
        self.approve_button.pack(side="left", padx=4)
        self._objective_changed()

    def _read_objective(self):
        numbers = {}
        for key in ("element_count", "max_actions", "max_seconds", "no_progress_turns"):
            try:
                numbers[key] = int(self.objective_vars[key].get().strip())
            except ValueError as error:
                raise ValueError(f"{key} must be a whole number.") from error
        return Objective(
            url=self.url_var.get().strip(), title=self.objective_vars["title"].get().strip(),
            **numbers,
        )

    def _objective_changed(self, *_):
        self.approved_objective = None
        self.proposal = None
        if not hasattr(self, "objective_preview"):
            return
        self.apply_button.configure(state="disabled")
        try:
            objective = self._read_objective()
        except ValueError as error:
            self.objective_status.set(str(error))
            self._set_text(self.objective_preview, f"Invalid objective: {error}")
        else:
            self.objective_status.set("Draft changed. Review and approve before starting.")
            self._set_text(self.objective_preview, objective.render())
        self._refresh_buttons()

    def approve_objective(self):
        if self.run_thread is not None or self.refining:
            return
        try:
            self.approved_objective = self._read_objective()
        except ValueError as error:
            self._show_error("Invalid objective", str(error))
            return
        self.proposal = None
        self.apply_button.configure(state="disabled")
        self._set_text(self.objective_preview, self.approved_objective.render())
        self.objective_status.set("Objective approved. Start will run this exact contract.")
        self._refresh_buttons()

    def discuss_objective(self):
        if self.run_thread is not None or self.refining:
            return
        try:
            objective = self._read_objective()
            request = self.request_var.get().strip()
            if not request or len(request) > 2000:
                raise ValueError("Enter a question or requested change of 1-2000 characters.")
        except ValueError as error:
            self._show_error("Cannot refine objective", str(error))
            return
        self.refining = True
        self.approved_objective = None
        self.proposal = None
        self.objective_status.set("Discussing with the model; the browser has not started.")
        self._refresh_buttons()
        threading.Thread(
            target=self._refine_worker, args=(objective, list(self.discussion), request),
            name="boxxy-objective", daemon=True,
        ).start()

    def _refine_worker(self, objective, discussion, request):
        try:
            text, proposal = refine_objective(self.llm, objective, discussion, request)
            self.post("objective_reply", request, text, proposal)
        except Exception:
            self.post("error", "Objective discussion failed", traceback.format_exc())
        finally:
            self.post("refinement_finished")

    def apply_proposal(self):
        if self.proposal is None or self.refining or self.run_thread is not None:
            return
        values = asdict(self.proposal)
        self.url_var.set(values.pop("url"))
        for key, value in values.items():
            self.objective_vars[key].set(str(value))
        self._objective_changed()

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
        if self.loading_screen is not None:
            self._set_text(self.loading_results, f"{title}\n\n{details}")
            return
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
        if self.closed:
            return
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
                if self.model_loading:
                    self._update_loading_progress()
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
        elif kind == "objective_reply":
            request, text, self.proposal = payload
            self.discussion.extend([
                {"role": "user", "content": request},
                {"role": "model", "content": text},
            ])
            self.discussion = self.discussion[-6:]
            self._set_text(self.discussion_text, "\n\n".join(
                f"{message['role']}: {message['content']}" for message in self.discussion
            ))
            if self.proposal is not None:
                current = asdict(self._read_objective())
                changes = "\n".join(
                    f"{key}: {current[key]!r} -> {value!r}"
                    for key, value in asdict(self.proposal).items() if current[key] != value
                )
                self._set_text(self.objective_preview,
                               "PROPOSED CHANGES (not applied)\n" + (changes or "No changes.")
                               + "\n\n" + self.proposal.render())
            self.request_var.set("")
        elif kind == "refinement_finished":
            self.refining = False
            self.objective_status.set("Discussion finished. Apply a proposal or edit, then approve.")
            self._refresh_buttons()
        elif kind == "model_loaded":
            self.loading_phase = "haiku"
            self.loading_phase_start = time.perf_counter()
            self.loading_status_var.set("Step 2 of 2: Generating the readiness haiku...")
            self._set_text(
                self.loading_results,
                f"Model loaded in {payload[0]:.2f}s.\nGenerating the readiness haiku...",
            )
        elif kind == "model_failed":
            self.model_loading = False
            self.loading_status_var.set("Model preparation failed. Retry or choose another model.")
            self.loading_time_var.set(
                f"Elapsed: {time.perf_counter() - self.loading_start:.1f}s"
            )
            self.retry_loading_button.configure(state="normal")
            self.choose_model_button.configure(state="normal")
        elif kind == "model_ready":
            self.llm, haiku, load_seconds, gen_seconds = payload
            self.model_loading = False
            self.loading_ready = True
            self.loading_progress.configure(value=100)
            self.loading_status_var.set("Model ready. Review the results, then press Okay to continue.")
            self.loading_time_var.set("Complete - 100%")
            self._set_text(
                self.loading_results,
                f"{haiku}\n\nBenchmark (one readiness run)\n"
                f"Model load: {load_seconds:.2f}s\n"
                f"Haiku generation: {gen_seconds:.2f}s\n"
                f"Total: {load_seconds + gen_seconds:.2f}s",
            )
            self.loading_ok_button.configure(state="normal")
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
        editing = not running and not getattr(self, "refining", False)
        approved = getattr(self, "approved_objective", None) is not None
        self.start_button.configure(
            state="normal" if self.llm is not None and editing and approved else "disabled"
        )
        self.stop_button.configure(state="normal" if running else "disabled")
        if hasattr(self, "objective_entries"):
            for entry in [self.url_entry, self.request_entry, *self.objective_entries]:
                entry.configure(state="normal" if editing else "disabled")
            for button in (self.refine_button, self.approve_button):
                button.configure(state="normal" if editing else "disabled")
            self.apply_button.configure(
                state="normal" if editing and self.proposal is not None else "disabled"
            )

    # ------------------------------------------------------------ workers

    def _update_loading_progress(self):
        now = time.perf_counter()
        elapsed = now - self.loading_start
        phase_elapsed = now - self.loading_phase_start
        generating = self.loading_phase == "haiku"
        estimate = ESTIMATED_HAIKU_SECONDS if generating else ESTIMATED_LOAD_SECONDS
        progress = (50 if generating else 0) + min(phase_elapsed / estimate, 0.9) * 50
        self.loading_progress.configure(value=progress)
        if phase_elapsed >= estimate:
            remaining = "Taking longer than estimated"
        else:
            seconds = estimate - phase_elapsed + (0 if generating else ESTIMATED_HAIKU_SECONDS)
            remaining = f"Estimated time remaining: ~{seconds:.0f}s"
        self.loading_time_var.set(
            f"Estimated progress: {progress:.0f}% | Elapsed: {elapsed:.1f}s | {remaining}"
        )

    def load_model_async(self):
        if self.closed or self.model_loading or self.loading_ready or self.loading_screen is None:
            return
        self.model_loading = True
        self.loading_start = self.loading_phase_start = time.perf_counter()
        self.loading_phase = "load"
        self.retry_loading_button.configure(state="disabled")
        self.choose_model_button.configure(state="disabled")
        self.loading_ok_button.configure(state="disabled")
        self.loading_status_var.set("Step 1 of 2: Loading the model...")
        self._set_text(self.loading_results, "")
        self._update_loading_progress()
        threading.Thread(target=self._load_model_worker, name="boxxy-model-loader", daemon=True).start()

    def _load_model_worker(self):
        try:
            load_start = time.perf_counter()
            from modules.llm import LLM

            llm = LLM()
            llm.load_model(self.model_path)
            load_seconds = time.perf_counter() - load_start
        except Exception:
            self.post("error", "Model load failed", traceback.format_exc())
            self.post("model_failed")
            return

        self.post("model_loaded", load_seconds)

        try:
            gen_start = time.perf_counter()
            response = llm.gen_text(HAIKU_PROMPT)
            gen_seconds = time.perf_counter() - gen_start
            haiku = (response.get("text") or "").strip()
            if not haiku:
                raise ValueError("The model returned an empty readiness haiku.")
            self.post("model_ready", llm, haiku, load_seconds, gen_seconds)
        except Exception:
            self.post("error", "Haiku generation failed", traceback.format_exc())
            self.post("model_failed")

    def start_run(self):
        if (not self.main_open or self.llm is None or self.run_thread is not None
                or self.refining):
            return
        if self.approved_objective is None:
            self._show_error("Objective not approved", "Review and approve the objective before starting.")
            return
        objective = self.approved_objective
        if objective != self._read_objective():
            self._objective_changed()
            self._show_error("Objective changed", "Approve the edited objective before starting.")
            return

        self._hide_error()
        self.stop_event.clear()
        self.run_thread = threading.Thread(
            target=self._run_worker, args=(objective,), name="boxxy-mode1", daemon=True
        )
        self._refresh_buttons()
        self.workspace.select(self.execution_tab)
        self.status_var.set(f"Boxxy is testing {objective.url} …")
        self.run_thread.start()

    def stop_run(self):
        if self.run_thread is None:
            return
        self.stop_event.set()
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopping after the current step…")

    def _run_worker(self, objective):
        from modules.mode1 import Mode1, StopRequested

        boxxy = None
        try:
            logger.log(f"Starting bounded smoke test on {objective.url}")
            boxxy = Mode1(
                objective.url,
                objective=objective,
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
            self.post("status", f"{boxxy.outcome}: {boxxy.outcome_detail} Results: {boxxy.output_dir}")
        except StopRequested:
            logger.log("Boxxy stopped by user.")
            self.post("status", "Boxxy stopped.")
        except Exception:
            self.post("error", "Boxxy run failed", traceback.format_exc())
            self.post("status", "Boxxy stopped due to an error.")
            if boxxy is not None and boxxy.outcome is None:
                try:
                    boxxy.finish("blocked", "Run failed; see the error details.")
                except OSError:
                    self.post("error", "Failed to save run result", traceback.format_exc())
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
