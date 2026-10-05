import json
import re

from modules.logger import log


class StopRequested(Exception):
    """Raised when the user asks Boxxy to stop exploring."""


class Mode1:
    MAX_NO_PROGRESS_TURNS = 5
    MAX_LISTED_ELEMENTS = 40
    _ACTIONABLE_ROLES = {
        "button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio", "switch",
        "tab", "menuitem", "menuitemcheckbox", "menuitemradio", "option", "spinbutton", "slider",
    }
    # Matches snapshot lines like: - button "Add form element" [disabled]
    _SNAPSHOT_ELEMENT_RE = re.compile(r'^\s*-\s+([a-z]+)\s+"((?:[^"\\]|\\.)*)"(.*)$')

    def __init__(self, test_url, local_path=None, llm=None, on_snapshot=None, stop_event=None):
        self.test_url = test_url
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
        "trying reasonable actions until the form is visibly complete or you can identify a concrete blocker."

        # Initialize notes to keep track of observations during exploration.
        self.messages = []

        from modules.browser import Browser
        self.browser = Browser(self.test_url)

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
                            }
                        },
                        "required": ["element"]
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
                            }
                        },
                        "required": ["element", "text"]
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
                            }
                        },
                        "required": ["element", "option"]
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

        messages_with_snapshot = self.messages + [
            {"role": "user", "content": "The following is the current ARIA snapshot of the page:"},
            {"role": "user", "content": content},
            {"role": "user", "content": self._page_guidance(content, feedback)},
        ]

        log("Exploring content...", clear_screen=True)
        if self.on_snapshot is not None:
            self.on_snapshot(content)
        else:
            log(content)

        response = self.llm.gen_text(messages_with_snapshot, self.tools_schema)
        self._check_stop()

        successful_action = False
        state_changed = False
        action_errors = []
        for tool_call in response["tool_calls"]:
            self._check_stop()
            log(tool_call)
            name = "unknown"
            arguments = {}
            action_failed = False
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
                    self.perform_click(arguments)
                elif name == "perform_typing":
                    self.perform_typing(arguments)
                elif name == "perform_select_option":
                    self.perform_select_option(arguments)
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
            if state_changed:
                content = updated_content
                self._reset_blocked(content)
                if self.on_snapshot is not None:
                    self.on_snapshot(content)
                else:
                    log(content)
                if action_failed:
                    action_errors.append(error + " The ARIA snapshot changed during the failed action.")
                if successful_action or action_failed:
                    break
            elif action_failed:
                self._blocked_actions.add(action_signature)
                # A bad option shouldn't hide the whole combobox; the exact action is still blocked above.
                if name != "perform_select_option" and isinstance(arguments, dict) and arguments.get("element"):
                    self._blocked_targets.add(arguments["element"].strip().lstrip("- ").strip())
                error += " The ARIA snapshot did not change."
                action_errors.append(error)

            if successful_action:
                break

        if successful_action:
            summary = (
                "Browser action succeeded: "
                f"{self._action_signature(name, arguments)}. "
                + (
                    "The ARIA snapshot changed; use the new page state to choose the next action."
                    if state_changed
                    else "No ARIA snapshot change was detected; inspect the current state before continuing."
                )
            )
            self.messages.append({"role": "model", "content": summary})
            log(summary)

        return successful_action, action_errors

    @staticmethod
    def _action_signature(name, arguments):
        return f"{name} with arguments {json.dumps(arguments, sort_keys=True, ensure_ascii=True)}"

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
            {"role": "user", "content": "Please develop a complex form that includes various input types and validation rules on the quality of pastas."},
        ]

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
        report_messages = self.messages + [
            {
                "role": "user",
                "content": "Please provide a final report based only on observed evidence. State whether the "
                    "form was completed, and identify any concrete blockers. Do not describe unverified actions "
                    "as successful.",
            }
        ]
        if feedback:
            report_messages.append({"role": "user", "content": feedback})

        response = self.llm.gen_text(report_messages, self.tools_schema)
        log(f"{response['text']}\r", clear_screen=True)