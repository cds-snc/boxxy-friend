"""The editable, bounded GC Forms smoke-test contract."""

import json
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit


class ProbeFailed(RuntimeError):
    """A required check failed; the run must not continue."""


@dataclass(frozen=True)
class Objective:
    url: str = "https://forms-staging.cdssandbox.xyz/en/form-builder"
    title: str = "Boxxy smoke test"
    element_count: int = 3
    max_actions: int = 100
    max_seconds: int = 900
    no_progress_turns: int = 5

    def __post_init__(self):
        if not isinstance(self.url, str) or len(self.url) > 2048:
            raise ValueError("Scope URL must be a string of at most 2048 characters.")
        url = urlsplit(self.url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("Scope requires an HTTP(S) URL without credentials.")
        if not (url.path == "/en/form-builder" or url.path.startswith("/en/form-builder/")):
            raise ValueError("This preset supports only English GC Forms form-builder URLs.")
        if not self.allows(self.url):
            raise ValueError("Scope URL cannot start in publishing, settings or responses.")
        if not isinstance(self.title, str) or not self.title.strip() or len(self.title) > 120:
            raise ValueError("Form title must contain 1-120 characters.")
        for name, maximum in (
            ("element_count", 10), ("max_actions", 500), ("max_seconds", 3600),
            ("no_progress_turns", 10),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer between 1 and {maximum}.")

    @property
    def fields(self):
        kinds = ("textField", "textArea", "radio")
        return [
            {"label": f"Boxxy question {index + 1}", "type": kinds[index % 3],
             "required": index == 0}
            for index in range(self.element_count)
        ]

    def allows(self, url):
        approved, candidate = urlsplit(self.url), urlsplit(url)
        return (
            candidate.scheme == approved.scheme
            and candidate.netloc == approved.netloc
            and any(
                candidate.path == prefix or candidate.path.startswith(prefix + "/")
                for prefix in ("/en/form-builder",)
            )
            and not any(
                part in {"publish", "settings", "responses"}
                for part in candidate.path.split("/")
            )
        )

    def render(self):
        types = {"textField": "short answer", "textArea": "long answer", "radio": "radio buttons"}
        fields = "\n".join(
            f"- {field['label']}: {types[field['type']]}"
            + ("; required" if field["required"] else "; optional")
            for field in self.fields
        )
        return (
            f"SCOPE\nGC Forms, signed out: {self.url}\n"
            "Stay in this origin's form builder. No sign-in, publishing, settings, responses, "
            "external navigation, or real personal data.\n\n"
            f"PROBE\nCreate one new form named and titled {self.title!r}.\n"
            f"Add exactly {self.element_count} input elements:\n{fields}\n"
            "For radio buttons add options 'Option A' and 'Option B'. Use these exact labels. "
            "Do not add other elements, pages, branches or validation rules.\n"
            "Open Test mode. Call check_required_empty before entering answers: it submits the empty "
            "form and verifies rejection. Then call check_valid_input: it fills every input with "
            "synthetic data, submits and checks for validation errors. "
            "Once both checks pass, return to Edit, open Share, and call download_form on "
            "the Download form file button.\n\n"
            "HALT\nSuccess requires both verified Test checks and a saved, validated JSON export "
            "matching the title, input types, labels, count, options and required rule. "
            "Stop on a failed required check, concrete blocker, scope violation or user cancellation.\n"
            f"Limits: {self.max_actions} actions, {self.max_seconds} seconds, "
            f"{self.no_progress_turns} consecutive no-progress turns. "
            "Application code enforces these limits between model/browser operations."
        )

    def validate_export(self, data):
        if (not isinstance(data, dict) or data.get("schemaVersion") != 1
                or not isinstance(data.get("elements"), list)):
            raise ProbeFailed("Export is not a supported GC Forms JSON object with elements.")
        if data.get("titleEn") != self.title and data.get("titleFr") != self.title:
            raise ProbeFailed("Exported form title does not match the approved objective.")
        elements = data["elements"]
        if len(elements) != self.element_count:
            raise ProbeFailed(f"Expected exactly {self.element_count} exported elements, got {len(elements)}.")
        ids = [element.get("id") for element in elements if isinstance(element, dict)]
        if (len(ids) != len(elements) or any(type(value) is not int for value in ids)
                or len(set(ids)) != len(ids) or data.get("layout") != ids):
            raise ProbeFailed("Exported layout must contain exactly the approved elements in order.")
        for expected, actual in zip(self.fields, elements):
            if not isinstance(actual, dict) or actual.get("type") != expected["type"]:
                raise ProbeFailed(f"Wrong exported input type for {expected['label']}.")
            properties = actual.get("properties", {})
            if not isinstance(properties, dict):
                raise ProbeFailed("Exported element properties must be an object.")
            if expected["label"] not in (properties.get("titleEn"), properties.get("titleFr")):
                raise ProbeFailed(f"Missing approved label {expected['label']}.")
            validation = properties.get("validation", {})
            if (not isinstance(validation, dict)
                    or validation.get("required", False) is not expected["required"]):
                raise ProbeFailed(f"Wrong required rule for {expected['label']}.")
            if expected["type"] == "radio":
                choices = properties.get("choices", [])
                if not isinstance(choices, list) or len(choices) != 2 or any(
                    not isinstance(choice, dict)
                    or label not in (choice.get("en"), choice.get("fr"))
                    for choice, label in zip(choices, ("Option A", "Option B"))
                ):
                    raise ProbeFailed(f"Wrong radio options for {expected['label']}.")


OBJECTIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "maxLength": 2048},
        "title": {"type": "string", "minLength": 1, "maxLength": 120},
        "element_count": {"type": "integer", "minimum": 1, "maximum": 10},
        "max_actions": {"type": "integer", "minimum": 1, "maximum": 500},
        "max_seconds": {"type": "integer", "minimum": 1, "maximum": 3600},
        "no_progress_turns": {"type": "integer", "minimum": 1, "maximum": 10},
    },
    "required": list(Objective.__dataclass_fields__),
    "additionalProperties": False,
}

REFINE_TOOLS = [{
    "type": "function",
    "function": {
        "name": "propose_objective",
        "description": "Propose the complete objective with only the changes requested by the user.",
        "parameters": OBJECTIVE_SCHEMA,
    },
}]


def refine_objective(llm, objective, discussion, request):
    if not isinstance(request, str) or not request.strip() or len(request) > 2000:
        raise ValueError("Discussion requests must contain 1-2000 characters.")
    messages = [
        {"role": "system", "content": (
            "Help refine a bounded GC Forms signed-out smoke test. You cannot browse or execute. "
            "Supported edits: URL, title, element count (1-10; types cycle short answer, long answer, "
            "radio buttons), action/time/no-progress limits. First input is always required. "
            "Required-empty rejection, valid-input testing and validated JSON export are mandatory. "
            "Explain unsupported requests; do not claim you changed unsupported behavior. "
            "For supported edits call propose_objective once with all fields, retaining unchanged values. "
            "Otherwise answer briefly without tools. Proposals require user approval."
        )},
        *discussion[-6:],
        {"role": "user", "content": "Current objective:\n" + json.dumps(asdict(objective))
         + "\nRequest:\n" + request},
    ]
    response = llm.gen_text(messages, REFINE_TOOLS)
    calls = response["tool_calls"]
    proposal = None
    if calls:
        if len(calls) != 1 or calls[0]["function"]["name"] != "propose_objective":
            raise ValueError("Expected exactly one propose_objective call.")
        values = calls[0]["function"]["arguments"]
        if not isinstance(values, dict) or set(values) != set(OBJECTIVE_SCHEMA["required"]):
            raise ValueError("The model proposal must contain exactly the supported objective fields.")
        proposal = Objective(**values)
    text = response["text"].strip()
    if not text and proposal is None:
        raise ValueError("The model returned neither advice nor a proposal.")
    return text or "Review the proposed objective below.", proposal
