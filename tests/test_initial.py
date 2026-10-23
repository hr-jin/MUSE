import unittest

from muse.initial import problem_text


class ProblemTextTests(unittest.TestCase):
    def test_plain_problem_is_unchanged(self):
        text = "  Solve the heat equation.\n\nPreserve the boundary conditions.\n"
        self.assertEqual(problem_text(text), text)

    def test_environment_sections_leave_task_requirements(self):
        text = (
            "# Task\nSolve the heat equation.\n\n"
            "## Environment\nUse the shared runtime.\n"
            "### Tools\nUse Write and Bash.\n\n"
            "## Implementation Requirements\nReturn an (N, 3) array.\n\n"
            "## Workspace protocol\nWork in the assigned directory.\n"
        )
        self.assertEqual(
            problem_text(text),
            "# Task\nSolve the heat equation.\n\n"
            "## Implementation Requirements\nReturn an (N, 3) array.",
        )

    def test_pde_guide_suffix_removed(self):
        problem = (
            "# Problem\nSolve Burgers on a sector.\n\n"
            "## Implementation Requirements\nDefine solve(case_spec)."
        )
        text = problem + (
            "\n\n---\n\n## DOLFINX 0.10.0 Guide\n"
            "# FEniCSx\nShared library guide.\n"
            "```python\nfrom dolfinx import fem\n```\n"
            "## Workspace protocol\nUse the assigned workspace.\n"
        )
        self.assertEqual(problem_text(text), problem)

    def test_kernel_submission_prefix_removed_without_losing_code(self):
        problem = (
            "Implement the following kernel.\n"
            "```python\ndef forward(x):\n    return x.sum(dim=1)\n```\n"
            "Return the same dtype as the input."
        )
        prefix = (
            "## Submission instructions\n"
            "Read the supplied task materials and write the required implementation files.\n"
            "Do not run tests. Once the files are complete, save them, submit, and finish.\n\n"
        )
        self.assertEqual(problem_text(prefix + problem), problem)
        self.assertEqual(problem_text((prefix + problem).replace("\n", "\r\n")),
                         problem.replace("\n", "\r\n"))

    def test_task_specific_submission_requirements_are_preserved(self):
        text = "## Submission instructions\nImplement forward(x) with output shape (N, 3).\n"
        self.assertEqual(problem_text(text), text)

    def test_environment_markers_inside_code_are_preserved(self):
        for fence in ("```", "~~~~"):
            code = (
                f"{fence}python\n"
                "# Environment\n"
                "## DOLFINX 0.10.0 Guide\n"
                "## Workspace protocol\n"
                f"{fence}\n"
            )
            text = "# Task\nPreserve this code.\n" + code
            self.assertEqual(problem_text(text), text)
            self.assertEqual(problem_text(text + "\n## Environment\nShared instructions."),
                             text.rstrip())

    def test_environment_only_prompt_rejected(self):
        with self.assertRaisesRegex(ValueError, "No problem text remains"):
            problem_text("## Shared environment instructions\nUse the runtime.\n")


if __name__ == "__main__":
    unittest.main()
