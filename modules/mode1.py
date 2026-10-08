import json
import re

from modules.logger import log
from modules.window_layout import WindowBounds


class StopRequested(Exception):
    """Raised when the user asks Boxxy to stop exploring."""


class Mode1:
    MAX_NO_PROGRESS_TURNS = 5
    MAX_LISTED_ELEMENTS = 40
    # Progress-log sizing: recent steps are shown in full, older ones as one-liners, then just counted.
    MAX_RECENT_STEPS = 8
    MAX_EARLIER_STEPS = 30
    MAX_RECORDED_FAILURES = 8
    MAX_CHANGE_LINES = 6
    MAX_LINE_LENGTH = 120
    MAX_PLAN_ITEMS = 15
    MAX_FINDINGS = 15
    PROGRESS_TOOL = "update_progress"
    DEFAULT_GOAL = "Please develop a complex form that includes various input types and validation rules on the " \
        "quality of pastas."
    _INTENT_PROPERTY = {
        "type": "string",
        "description": "One short sentence explaining why you are taking this action and which part of the goal "
            "it advances (e.g. \"Name the form so it can be saved\").",
    }
    _ACTIONABLE_ROLES = {
        "button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio", "switch",
        "tab", "menuitem", "menuitemcheckbox", "menuitemradio", "option", "spinbutton", "slider",
    }
    # Matches snapshot lines like: - button "Add form element" [disabled]
    _SNAPSHOT_ELEMENT_RE = re.compile(r'^\s*-\s+([a-z]+)\s+"((?:[^"\\]|\\.)*)"(.*)$')

    def __init__(self, test_url, local_path=None, llm=None, on_snapshot=None, stop_event=None, goal=None,
                 on_progress=None, on_prompt=None, browser_bounds: WindowBounds | None = None):
        # on_progress(text) receives the current progress block; on_prompt(messages) receives each prompt sent
        # to the LLM. Both are optional hooks for UIs and are called from the exploring thread.
        self.on_progress = on_progress
        self.on_prompt = on_prompt
        self.test_url = test_url
        self.goal = goal or self.DEFAULT_GOAL
        self.steps = []
        self.failures = []
        # Agent-maintained notes: a checklist of sub-goals and observed findings (bugs, accessibility issues).
        self.plan = []
        self.findings = []
        self._progress_updated = False
        self.on_snapshot = on_snapshot
        self.stop_event = stop_event
        self._blocked_snapshot = None
        self._blocked_actions = set()
        self._blocked_targets = set()
        self.test_behavior = "You are an automated testing agent mimicking a user utilizing a screen reader. " \
        "Your goal is to test the whole application for accessibility and functionality." \
        "You are testing GC Forms, a product for creating and managing web forms." \
        "Simulate developing a complex web form with various input types and validation rules." \
        "Be curious, explore, and provide feedback on your findings." \
        "Always perform an action until your task is completed." \
        "When you call the perform_click tool, the 'element' argument MUST be copied verbatim from the ARIA " \
        "snapshot you were given, in the exact form <role> \"<accessible name>\" (for example: " \
        "button \"Design a form Start with a blank form.\"). Use the full accessible name exactly as it appears " \
        "in the snapshot, including any text contributed by child elements. Never paraphrase, shorten, or " \
        "reword the accessible name, and never invent an element that is not present in the snapshot. " \
        "To choose a value in a dropdown (a combobox), call perform_select_option with the combobox as 'element' " \
        "and the option's accessible name as 'option'; do not click the option elements of a combobox directly. " \
        "Use the available tools directly whenever an action is needed. Do not write or imitate tool-call " \
        "JSON in your message; the tools are provided through the tool-calling interface. Do not ask the user " \
        "for guidance. A response without a tool call does not mean the task is complete: keep exploring and " \
        "trying reasonable actions until the form is visibly complete or you can identify a concrete blocker. " \
        "Every tool call must include an 'intent' argument: one short sentence saying why you are taking the " \
        "action and which part of the goal it advances. Each turn you receive a 'Progress so far' section with " \
        "the goal, the steps you already completed (with their intent, result and how the page changed) and " \
        "recent failures. Use it to decide what remains to be done: do not redo completed steps and do not " \
        "retry actions that already failed unless the page has changed. " \
        "Keep your own notes with the update_progress tool: at the start, write a plan of the sub-goals needed to " \
        "reach the goal; as you complete sub-goals, mark them done; and record findings such as bugs, " \
        "accessibility problems, confusing labels or unexpected behaviour. update_progress does not interact " \
        "with the page, so call it in the same response as a browser action rather than on its own."

        # Initialize notes to keep track of observations during exploration.
        self.messages = []

        from modules.browser import Browser
        self.browser = Browser(self.test_url, window_bounds=browser_bounds)

        if llm is None:
            from modules.llm import LLM
            llm = LLM()
            llm.load_model(local_path)
        self.llm = llm

        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "perform_click",
                    "description": "Perform a click action on the web page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element": {
                                "type": "string",
                                "description": "The exact line for the target element as it appears verbatim in the "
                                    "ARIA snapshot, in the form <role> \"<accessible name>\" (e.g. button \"Design a "
                                    "form Start with a blank form.\"). Copy the role and full quoted accessible name "
                                    "character-for-character from the snapshot, including any inner text from child "
                                    "elements. Do not paraphrase, summarize, truncate, or invent text that isn't in "
                                    "the snapshot."
                            },
                            "intent": self._INTENT_PROPERTY
                        },
                        "required": ["element", "intent"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "perform_typing",
                    "description": "Perform a typing action on the web page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element": {
                                "type": "string",
                                "description": "The exact line for the target element as it appears verbatim in the "
                                    "ARIA snapshot, in the form <role> \"<accessible name>\" (e.g. button \"Design a "
                                    "form Start with a blank form.\"). Copy the role and full quoted accessible name "
                                    "character-for-character from the snapshot, including any inner text from child "
                                    "elements. Do not paraphrase, summarize, truncate, or invent text that isn't in "
                                    "the snapshot."
                            },
                            "text": {
                                "type": "string",
                                "description": "The text to type into the target element."
                            },
                            "intent": self._INTENT_PROPERTY
                        },
                        "required": ["element", "text", "intent"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "perform_select_option",
                    "description": "Select an option in a dropdown (combobox) on the web page. Works for native "
                        "select elements and custom comboboxes; the dropdown is opened automatically.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element": {
                                "type": "string",
                                "description": "The combobox exactly as it appears in the ARIA snapshot, in the form "
                                    "combobox \"<accessible name>\" (e.g. combobox \"Order list:\"). Copy the "
                                    "accessible name character-for-character from the snapshot."
                            },
                            "option": {
                                "type": "string",
                                "description": "The accessible name of the option to select, copied exactly from the "
                                    "snapshot (e.g. Alphabetically (A-Z))."
                            },
                            "intent": self._INTENT_PROPERTY
                        },
                        "required": ["element", "option", "intent"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": self.PROGRESS_TOOL,
                    "description": "Update your own progress notes. Does not interact with the web page. Use it "
                        "to set a plan of sub-goals, mark sub-goals done, and record findings. Call it together "
                        "with a browser action.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plan": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "The full ordered list of sub-goals needed to reach the goal. "
                                    "Replaces the previous plan; items that keep the same wording stay marked "
                                    "done. Omit to keep the current plan."
                            },
                            "done": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Plan items you have now completed, copied exactly from the plan. "
                                    "Only mark an item done when the progress log shows evidence for it."
                            },
                            "finding": {
                                "type": "string",
                                "description": "One observation worth reporting, such as a bug, an accessibility "
                                    "problem, a confusing label or unexpected behaviour. Mention where it happened."
                            }
                        },
                        "required": []
                    }
                }
            }
        ]

    def launch(self):
        self.browser.open()

    def close(self):
        self.browser.close()

    def _check_stop(self):
        if self.stop_event is not None and self.stop_event.is_set():
            raise StopRequested("Boxxy was asked to stop.")

    def llm_access_content(self, feedback=None):
        self._check_stop()
        content = self.browser.explore_view()
        if content != self._blocked_snapshot:
            self._reset_blocked(content)
        page = self._page_label()

        progress = self._progress_report()
        messages_with_snapshot = self.messages + [
            {"role": "user", "content": progress},
            {"role": "user", "content": f"The following is the current ARIA snapshot of the page {page}:"},
            {"role": "user", "content": content},
            {"role": "user", "content": self._page_guidance(content, feedback)},
        ]

        log("Exploring content...", clear_screen=True)
        if self.on_snapshot is not None:
            self.on_snapshot(content)
        else:
            log(content)
        self._publish(progress, messages_with_snapshot)

        response = self.llm.gen_text(messages_with_snapshot, self.tools_schema)
        self._check_stop()

        successful_action = False
        action_errors = []
        # Apply note updates first so they are kept even when listed after a browser action that ends the turn.
        self._progress_updated = False
        browser_calls = []
        for tool_call in response["tool_calls"]:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            if isinstance(function, dict) and function.get("name") == self.PROGRESS_TOOL:
                log(tool_call)
                self._update_progress(function.get("arguments", {}))
            else:
                browser_calls.append(tool_call)

        for tool_call in browser_calls:
            self._check_stop()
            log(tool_call)
            name = "unknown"
            arguments = {}
            action_failed = False
            result = None
            try:
                function = tool_call.get("function", {})
                name = function.get("name", "unknown")
                arguments = function.get("arguments", {})
                action_signature = self._action_signature(name, arguments)
                if action_signature in self._blocked_actions:
                    log(f"Rejected repeated action: {action_signature}")
                    action_errors.append(
                        f"{action_signature} was not executed: it already failed while the page was in "
                        "this unchanged state. Choose a different visible action."
                    )
                    continue

                if name == "perform_click":
                    result = self.perform_click(arguments)
                elif name == "perform_typing":
                    result = self.perform_typing(arguments)
                elif name == "perform_select_option":
                    result = self.perform_select_option(arguments)
                else:
                    raise ValueError(f"Unsupported tool: {name}")
                successful_action = True
            except Exception as exc:
                action_signature = self._action_signature(name, arguments)
                error = f"{action_signature} failed: {type(exc).__name__}: {exc}"
                log(f"Browser action failed: {error}")
                action_failed = True

            updated_content = self.browser.explore_view()
            state_changed = updated_content != content
            if successful_action:
                self._record_step(name, arguments, result, page, content, updated_content)
            if state_changed:
                content = updated_content
                self._reset_blocked(content)
                if self.on_snapshot is not None:
                    self.on_snapshot(content)
                else:
                    log(content)
                if action_failed:
                    error += " The ARIA snapshot changed during the failed action."
                    action_errors.append(error)
                    self._record_failure(page, arguments, error)
                if successful_action or action_failed:
                    break
            elif action_failed:
                self._blocked_actions.add(action_signature)
                # A bad option shouldn't hide the whole combobox; the exact action is still blocked above.
                if name != "perform_select_option" and isinstance(arguments, dict) and arguments.get("element"):
                    self._blocked_targets.add(arguments["element"].strip().lstrip("- ").strip())
                error += " The ARIA snapshot did not change."
                action_errors.append(error)
                self._record_failure(page, arguments, error)

            if successful_action:
                break

        # Refresh the progress view now so it reflects this turn while the next prompt is being prepared.
        self._publish(self._progress_report())
        return successful_action, action_errors

    def _publish(self, progress=None, messages=None):
        if progress is not None and self.on_progress is not None:
            self.on_progress(progress)
        if messages is not None and self.on_prompt is not None:
            self.on_prompt(messages)

    @staticmethod
    def _action_signature(name, arguments):
        # Intent is free text; leave it out so rewording it can't bypass the repeated-failure block.
        if isinstance(arguments, dict):
            arguments = {key: value for key, value in arguments.items() if key != "intent"}
        return f"{name} with arguments {json.dumps(arguments, sort_keys=True, ensure_ascii=True)}"

    @staticmethod
    def _intent(arguments):
        intent = arguments.get("intent") if isinstance(arguments, dict) else None
        return intent.strip() if isinstance(intent, str) and intent.strip() else "(not stated)"

    def _truncate(self, text):
        text = " ".join(str(text).split())
        return text if len(text) <= self.MAX_LINE_LENGTH else text[:self.MAX_LINE_LENGTH - 1] + "…"

    def _page_label(self):
        try:
            info = self.browser.page_info()
        except Exception:
            return "(unknown page)"
        title, url = info.get("title") or "", info.get("url") or ""
        if title and url:
            return f'"{title}" ({url})'
        return f'"{title}"' if title else url or "(unknown page)"

    def _snapshot_changes(self, before, after):
        if before == after:
            return "No ARIA snapshot change was detected."

        def lines(snapshot):
            seen = []
            for line in (snapshot or "").splitlines():
                line = line.strip().lstrip("- ").strip().rstrip(":")
                if line and line not in seen:
                    seen.append(line)
            return seen

        old, new = lines(before), lines(after)
        old_set, new_set = set(old), set(new)
        parts = []
        for label, items in (
            ("appeared", [line for line in new if line not in old_set]),
            ("disappeared", [line for line in old if line not in new_set]),
        ):
            if not items:
                continue
            shown = [self._truncate(item) for item in items[:self.MAX_CHANGE_LINES]]
            more = len(items) - len(shown)
            parts.append(f"{label}: " + "; ".join(shown) + (f" (+{more} more)" if more > 0 else ""))
        return "The ARIA snapshot changed. " + (" | ".join(parts) if parts else "Only ordering changed.")

    def _record_step(self, name, arguments, result, page, before, after):
        step = {
            "number": len(self.steps) + 1,
            "page": page,
            "intent": self._intent(arguments),
            "action": self._action_signature(name, arguments),
            "result": result or self._action_signature(name, arguments),
            "changes": self._snapshot_changes(before, after),
        }
        self.steps.append(step)
        log(f"Step {step['number']}: {step['intent']} -> {step['result']}. {step['changes']}")

    def _record_failure(self, page, arguments, error):
        self.failures.append(f"On {page} (intent: {self._intent(arguments)}): {error}")
        del self.failures[:-self.MAX_RECORDED_FAILURES]

    @staticmethod
    def _string_list(value):
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [" ".join(item.split()) for item in value if isinstance(item, str) and item.strip()]

    @staticmethod
    def _plan_key(text):
        # Tolerate the model copying an item with its checkbox prefix or different casing.
        text = re.sub(r"^\s*(?:\d+[.)]\s*)?\[[ xX]\]\s*", "", text)
        return " ".join(text.lower().split()).rstrip(".")

    def _update_progress(self, arguments):
        if not isinstance(arguments, dict):
            return
        self._progress_updated = True

        if "plan" in arguments:
            items = self._string_list(arguments["plan"])
            if items:
                done = {self._plan_key(item["text"]) for item in self.plan if item["done"]}
                plan, seen = [], set()
                for text in items:
                    key = self._plan_key(text)
                    if key and key not in seen:
                        seen.add(key)
                        plan.append({"text": self._truncate(re.sub(r"^\s*\[[ xX]\]\s*", "", text)),
                                     "done": key in done})
                self.plan = plan[:self.MAX_PLAN_ITEMS]

        for text in self._string_list(arguments.get("done")):
            key = self._plan_key(text)
            match = next((item for item in self.plan if self._plan_key(item["text"]) == key), None)
            if match is not None:
                match["done"] = True
            elif key and len(self.plan) < self.MAX_PLAN_ITEMS:
                self.plan.append({"text": self._truncate(text), "done": True})

        finding = arguments.get("finding")
        if isinstance(finding, str) and finding.strip():
            finding = self._truncate(finding)
            if finding not in self.findings:
                self.findings.append(finding)
                del self.findings[:-self.MAX_FINDINGS]

        log(f"Progress notes updated: {sum(item['done'] for item in self.plan)}/{len(self.plan)} plan items done, "
            f"{len(self.findings)} findings.")

    def _notes_report(self):
        parts = []
        if self.plan:
            parts.append(
                "Your plan (checklist you maintain with update_progress):\n"
                + "\n".join(f"[{'x' if item['done'] else ' '}] {item['text']}" for item in self.plan)
            )
        else:
            parts.append(
                "Your plan: none yet. Call update_progress with a plan of the sub-goals needed to reach the goal, "
                "together with your next browser action."
            )
        if self.findings:
            parts.append("Your findings so far:\n" + "\n".join(f"- {finding}" for finding in self.findings))
        return parts

    def _progress_report(self):
        parts = [f"Progress so far.\nGoal: {self.goal}"]
        parts += self._notes_report()

        if not self.steps:
            parts.append("Completed steps: none yet. This is the first action.")
        else:
            recent = self.steps[-self.MAX_RECENT_STEPS:]
            earlier = self.steps[:-self.MAX_RECENT_STEPS]
            if earlier:
                shown = earlier[-self.MAX_EARLIER_STEPS:]
                omitted = len(earlier) - len(shown)
                summary = [f"({omitted} older steps omitted)"] if omitted else []
                summary += [
                    f"Step {step['number']}: {self._truncate(step['intent'] + ' -> ' + step['result'])}"
                    for step in shown
                ]
                parts.append("Earlier steps (summarized):\n" + "\n".join(summary))
            parts.append(
                "Most recent steps:\n" + "\n".join(
                    f"Step {step['number']} on {step['page']}\n"
                    f"  intent: {step['intent']}\n"
                    f"  result: {step['result']}\n"
                    f"  page change: {step['changes']}"
                    for step in recent
                )
            )

        if self.failures:
            parts.append("Recent failed actions (on any page):\n" + "\n".join(self.failures))

        parts.append(
            "Compare the goal and your plan with the completed steps to decide what is still missing, then "
            "continue from there."
        )
        return "\n\n".join(parts)

    def _reset_blocked(self, snapshot):
        self._blocked_snapshot = snapshot
        self._blocked_actions.clear()
        self._blocked_targets.clear()

    def _blocked_elements(self):
        return self._blocked_targets

    def _available_elements(self, snapshot):
        blocked = self._blocked_elements()
        elements = []
        combobox_indent = None
        for line in snapshot.splitlines():
            indent = len(line) - len(line.lstrip())
            if combobox_indent is not None and indent <= combobox_indent:
                combobox_indent = None
            match = self._SNAPSHOT_ELEMENT_RE.match(line)
            if not match:
                continue
            if match.group(1) == "combobox":
                combobox_indent = indent
            elif combobox_indent is not None and match.group(1) == "option":
                continue  # Listed with its combobox; selected via perform_select_option.
            if match.group(1) not in self._ACTIONABLE_ROLES or "[disabled]" in match.group(3):
                continue
            element = f'{match.group(1)} "{match.group(2)}"'
            if element not in blocked and element not in elements:
                elements.append(element)
        return elements

    def _combobox_options(self, snapshot):
        comboboxes = {}
        current, current_indent = None, None
        for line in snapshot.splitlines():
            indent = len(line) - len(line.lstrip())
            if current is not None and indent <= current_indent:
                current = None
            match = self._SNAPSHOT_ELEMENT_RE.match(line)
            if not match:
                continue
            if match.group(1) == "combobox":
                if "[disabled]" in match.group(3):
                    continue
                current, current_indent = f'combobox "{match.group(2)}"', indent
                comboboxes.setdefault(current, [])
            elif current is not None and match.group(1) == "option":
                selected = " [selected]" if "[selected]" in match.group(3) else ""
                comboboxes[current].append(f'"{match.group(2)}"{selected}')
        return comboboxes

    def _page_guidance(self, snapshot, feedback=None):
        parts = []
        elements = self._available_elements(snapshot)
        if elements:
            listed = elements[:self.MAX_LISTED_ELEMENTS]
            parts.append(
                "Elements you can use now (copy one exactly as the 'element' argument):\n"
                + "\n".join(listed)
            )

        comboboxes = self._combobox_options(snapshot)
        if comboboxes:
            parts.append(
                "Dropdowns (use perform_select_option with the combobox as 'element' and an option name as "
                "'option'; custom dropdowns may not list their options until opened):\n"
                + "\n".join(
                    f"{combobox} options: {', '.join(options) if options else '(not listed)'}"
                    for combobox, options in comboboxes.items()
                )
            )

        blocked = sorted(self._blocked_elements())
        if blocked:
            parts.append("Do not use these elements; they already failed on this page:\n" + "\n".join(blocked))

        if feedback:
            parts.append(feedback)

        parts.append("Choose the next action that moves the form forward.")
        return "\n\n".join(parts)

    def explore(self):
        self.messages = [
            {"role": "system", "content": self.test_behavior},
            {"role": "user", "content": self.goal},
        ]
        self.steps = []
        self.failures = []
        self.plan = []
        self.findings = []

        no_progress_turns = 0
        feedback = None
        self._reset_blocked(None)
        while True:
            self._check_stop()
            made_progress, action_errors = self.llm_access_content(feedback)
            if made_progress:
                no_progress_turns = 0
                feedback = None
                if action_errors:
                    feedback = (
                        "One or more actions failed:\n"
                        + "\n".join(action_errors)
                        + "\nThose failed actions are blocked for this unchanged page state. Inspect the current "
                        "snapshot and choose a different visible target or a different action."
                    )
                continue

            no_progress_turns += 1
            if action_errors:
                feedback = (
                    "The previous browser action did not succeed:\n"
                    + "\n".join(action_errors)
                    + "\nDo not repeat any listed action while the ARIA snapshot is unchanged. Inspect the "
                    "current snapshot, choose a different visible target or action, and account for any page-state "
                    "change noted above."
                )
            elif self._progress_updated:
                # Notes alone don't count as progress, otherwise the agent could loop on them forever.
                feedback = (
                    "Your progress notes were saved, but no browser action was performed. update_progress does "
                    "not change the page. Now choose the next browser action from the current snapshot that "
                    "advances the next unchecked item in your plan."
                )
            else:
                feedback = (
                    "No browser action was performed. Continue the task without asking the user for guidance. "
                    "Choose a reasonable action visible in the current snapshot, or use the snapshot and prior "
                    "evidence to identify a concrete blocker. A text-only response is not proof of completion."
                )

            if no_progress_turns >= self.MAX_NO_PROGRESS_TURNS:
                self.report(
                    "No browser action succeeded for "
                    f"{no_progress_turns} consecutive turns. Assess the current page and prior evidence; "
                    "do not claim completion unless the form is visibly complete.\n"
                    + feedback
                )
                return

    def perform_click(self, instructions):
        return self.browser.click(instructions["element"])

    def perform_typing(self, instructions):
        return self.browser.type(instructions["element"], instructions["text"])

    def perform_select_option(self, instructions):
        return self.browser.use_combobox(instructions["element"], instructions["option"])

    def report(self, feedback=None):
        progress = self._progress_report()
        report_messages = self.messages + [
            {"role": "user", "content": progress},
            {
                "role": "user",
                "content": "Please provide a final report based only on observed evidence in the progress log "
                    "above. Compare the goal and your plan with the completed steps: state which parts of the goal "
                    "were completed, which were not, and identify any concrete blockers or bugs (cite the step "
                    "numbers or failures that show them). Include every finding you recorded. Do not describe "
                    "unverified actions as successful.",
            }
        ]
        if feedback:
            report_messages.append({"role": "user", "content": feedback})

        self._publish(progress, report_messages)
        response = self.llm.gen_text(report_messages, self.tools_schema)
        log(f"{response['text']}\r", clear_screen=True)