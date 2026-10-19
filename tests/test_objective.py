import json
import unittest
from dataclasses import asdict, replace
from unittest.mock import MagicMock

from modules.objective import Objective, ProbeFailed, refine_objective


def exported_form(objective):
    elements = []
    for index, field in enumerate(objective.fields, 1):
        properties = {
            "titleEn": field["label"], "titleFr": "",
            "validation": {"required": field["required"]},
        }
        if field["type"] == "radio":
            properties["choices"] = [{"en": "Option A", "fr": ""}, {"en": "Option B", "fr": ""}]
        elements.append({"id": index, "type": field["type"], "properties": properties})
    return {"schemaVersion": 1, "titleEn": objective.title, "elements": elements,
            "layout": list(range(1, objective.element_count + 1))}


class ObjectiveTests(unittest.TestCase):
    def test_defaults_and_rendered_contract_are_bounded(self):
        objective = Objective()
        self.assertEqual((objective.element_count, objective.max_actions, objective.max_seconds,
                          objective.no_progress_turns), (3, 100, 900, 5))
        self.assertEqual([field["type"] for field in objective.fields],
                         ["textField", "textArea", "radio"])
        for heading in ("SCOPE", "PROBE", "HALT"):
            self.assertIn(heading, objective.render())
        self.assertTrue(objective.fields[0]["required"])
        self.assertFalse(objective.fields[1]["required"])

    def test_invalid_objectives_are_rejected(self):
        for values in (
            {"element_count": 0}, {"element_count": 11}, {"max_actions": True},
            {"max_seconds": 0}, {"no_progress_turns": 11}, {"title": " "},
            {"url": "file:///en/form-builder"}, {"url": "https://u:p@host/en/form-builder"},
            {"url": "https://host/en/form-builder-fake"},
            {"url": "https://host/en/form-builder/0000/publish"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Objective(**values)

    def test_scope_is_same_origin_builder_only(self):
        objective = Objective()
        for suffix in ("/0000/edit", "/0000/preview", ""):
            self.assertTrue(objective.allows(objective.url + suffix))
        for url in (
            "https://example.com/en/form-builder",
            objective.url.replace("https:", "http:"),
            objective.url + "-fake",
            objective.url + "/0000/publish",
            objective.url.replace("/form-builder", "/auth/login"),
        ):
            self.assertFalse(objective.allows(url), url)

    def test_export_requires_exact_count_types_labels_rules_and_layout(self):
        objective = Objective()
        objective.validate_export(exported_form(objective))
        for mutate in (
            lambda data: data.update(titleEn="wrong"),
            lambda data: data["elements"].pop(),
            lambda data: data["elements"][0].update(type="textArea"),
            lambda data: data["elements"][0]["properties"].update(titleEn="wrong"),
            lambda data: data["elements"][0]["properties"]["validation"].update(required=False),
            lambda data: data["elements"][2]["properties"]["choices"].pop(),
            lambda data: data.update(layout=[]),
        ):
            data = exported_form(objective)
            mutate(data)
            with self.assertRaises(ProbeFailed):
                objective.validate_export(data)
        for data in ([], {}, {"elements": []}):
            with self.assertRaises(ProbeFailed):
                objective.validate_export(data)

    def test_refinement_proposes_without_mutating_and_bounds_context(self):
        objective = Objective()
        proposed = replace(objective, element_count=2)
        llm = MagicMock()
        llm.gen_text.return_value = {
            "text": "Use two inputs.", "tool_calls": [{
                "function": {"name": "propose_objective", "arguments": asdict(proposed)}
            }],
        }
        discussion = [{"role": "user", "content": str(index)} for index in range(12)]
        text, result = refine_objective(llm, objective, discussion, "Use two")
        self.assertEqual(result, proposed)
        self.assertEqual(objective.element_count, 3)
        self.assertEqual(len(llm.gen_text.call_args.args[0]), 8)
        self.assertEqual(text, "Use two inputs.")
        self.assertIn(json.dumps(asdict(objective)), llm.gen_text.call_args.args[0][-1]["content"])

    def test_bad_proposals_and_empty_responses_are_errors(self):
        for result in (
            {"text": "", "tool_calls": []},
            {"text": "", "tool_calls": [{"function": {"name": "perform_click", "arguments": {}}}]},
            {"text": "", "tool_calls": [{"function": {"name": "propose_objective", "arguments": {}}}]},
        ):
            llm = MagicMock()
            llm.gen_text.return_value = result
            with self.assertRaises(ValueError):
                refine_objective(llm, Objective(), [], "Change it")

    def test_discussion_request_length_is_bounded_before_generation(self):
        llm = MagicMock()
        for request in ("", " ", "x" * 2001):
            with self.assertRaisesRegex(ValueError, "1-2000"):
                refine_objective(llm, Objective(), [], request)
        llm.gen_text.assert_not_called()
