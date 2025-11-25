"""Collect documentation metrics from a set of git repositories and save in data files."""

import ast
import json
import os
import re
from pathlib import Path
from typing import Iterable, Callable, TypedDict
import subprocess

import git
import methodtools
from git import GitCommandError
from interrogate import coverage
import nltk
import pandas
from readability import Readability
import tqdm

import ast_node_checks
import cvar
import debug_tools
import logger

LOG = logger.get()

with open(
    cvar.resources_dir / "wiki_file_text_extensions.json", "r", encoding="utf-8"
) as f:
    WIKI_FILE_TEXT_EXTENSIONS = set(json.load(f))

nltk.download("punkt_tab", quiet=True)  # `readability` library demands.


class DocumentationMetrics(TypedDict):
    """Type for documentation metrics dictionary."""

    repo_name: str
    readme_length: int
    github_wiki_length: int
    readme_completeness: float
    docstring_coverage: float
    documentation_percentage: float
    documentation_up_to_date: float
    documentation_readability: float
    code_examples_ratio: float


documentation_metrics_dictionary: DocumentationMetrics = {
    "repo_name": "",
    "readme_length": 0,
    "github_wiki_length": 0,
    "readme_completeness": 0.0,
    "docstring_coverage": 0.0,
    "documentation_percentage": 0.0,
    "documentation_up_to_date": 0.0,
    "documentation_readability": 0.0,
    "code_examples_ratio": 0.0,
}


class DocumentationMetricsCollect:
    """Collect documentation metrics from a git repository and save in a data file."""

    def __init__(self, repo_url: str):
        if not isinstance(repo_url, str) or not repo_url:
            raise ValueError(
                f"Received repository URL was empty or invalid: {repo_url!r}"
            )

        self.repo_url = repo_url
        self.repo_name = Path(self.repo_url.rstrip("/")).stem

        self.repo_dir = (
            cvar.data_dir / "repos" / "documentation_metrics" / self.repo_name
        )
        self.repo_dir.parent.mkdir(parents=True, exist_ok=True)

        self.repo: git.Repo
        if os.path.isdir(self.repo_dir) and os.listdir(self.repo_dir):
            self.repo = git.Repo(self.repo_dir)
            LOG.debug(
                "Opened existing repository %r in %r", self.repo_name, self.repo_dir
            )
        else:
            self.repo = git.Repo.clone_from(self.repo_url, self.repo_dir)
            LOG.debug("Cloned repository %r into %r", self.repo_url, self.repo_dir)

        main_branch = self.main_branch
        self.repo.git.checkout(main_branch)
        LOG.debug(
            "Checked out main branch %r for repository %r", main_branch, self.repo_name
        )

        self.metrics = documentation_metrics_dictionary.copy()
        self.metrics["repo_name"] = self.repo_name

        with open(
            cvar.resources_dir / "readme_section_names.json", "r", encoding="utf-8"
        ) as f:
            self.readme_section_names = json.load(f)

        self.readme_path: str | None = self._find_readme()

        self.wiki_dir: Path = self.repo_dir.parent / f"{self.repo_name}.wiki"

        self.docs_dir: Path | None = self._find_docs_directory()

        self.public_source_items = (
            self._extract_public_classes_and_methods_from_source()
        )

    @property
    def main_branch(self) -> str:
        """Default, main, or master branch of the repository."""
        if not self.repo:
            raise ValueError(
                "Repository must be initialized before retrieving the main branch."
            )

        # Default branch.
        origin_info = self.repo.git.remote("show", "origin")
        matches = re.search(r"\s*HEAD branch:\s*(.*)", origin_info)
        if matches:
            default_branch = matches.group(1).strip()
            if default_branch:
                return default_branch

        candidates = ("main", "master")
        local_heads = {head.name for head in self.repo.heads}
        for candidate in candidates:
            if candidate in local_heads:
                return candidate

        error_message = (
            f"Could not find the default, main, or master branch for repository '{self.repo_name}' "
            f"from available branches {local_heads}."
        )
        raise ValueError(error_message)

    def collect_metrics(self) -> DocumentationMetrics:
        """Collect documentation metrics with per-metric fault isolation."""
        steps: Iterable[Callable[[], None]] = (
            self.get_readme_length,
            self.get_github_wiki_length,
            self.get_readme_completeness,
            self.get_docstring_coverage,
            self.get_documentation_percentage,
            self.get_documentation_up_to_date,
            self.get_documentation_readability,
            # self.get_code_examples_ratio,  # This metric feels like documentation_up_to_date but worse.
        )
        for step in steps:
            step()

        return self.metrics

    def _find_readme(self) -> str | None:
        """
        Find README file in the repository root directory.

        :return: Path to README file if found, else None.
        """
        candidates = {"readme", "read_me"}
        extensions = WIKI_FILE_TEXT_EXTENSIONS
        try:
            for entry in Path(self.repo_dir).iterdir():
                if not entry.is_file():
                    continue
                name = entry.name.lower()
                for base in candidates:
                    for ext in extensions:
                        if name == base + ext:
                            return str(entry)
        except FileNotFoundError:
            LOG.error(
                "Repository directory %r not found for repository %r when finding readme",
                self.repo_dir,
                self.repo_name,
            )
            debug_tools.debug_reraise()

        LOG.warning(
            "Could not find README file for repository %r in %r",
            self.repo_name,
            self.repo_dir,
        )
        return None

    def get_readme_length(self) -> None:
        """Calculate length of README file in characters."""
        readme_path = self.readme_path

        if readme_path:
            with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self.metrics["readme_length"] = len(content)
        else:
            self.metrics["readme_length"] = 0

    def get_github_wiki_length(self) -> None:
        """Calculate total character length of the GitHub wiki pages; 0 if no wiki."""
        wiki_url = self._github_wiki_url()
        if not wiki_url:
            self.metrics["github_wiki_length"] = 0
            return

        try:
            if os.path.isdir(self.wiki_dir) and os.listdir(self.wiki_dir):
                wiki_repo = git.Repo(self.wiki_dir)
                try:
                    wiki_repo.remotes.origin.fetch()
                except Exception as exc:
                    LOG.warning("Fetch of wiki for `%s`: %s", self.repo_name, exc)
            else:
                git.Repo.clone_from(wiki_url, self.wiki_dir)
        except GitCommandError:
            LOG.info("Wiki does not exist for %s", self.repo_name)
            self.metrics["github_wiki_length"] = 0
            return
        except Exception as exc:
            LOG.warning("Wiki not found for `%s`: %s", self.repo_name, exc)
            self.metrics["github_wiki_length"] = 0
            return

        total_characters = 0
        try:
            for root, dirs, files in os.walk(self.wiki_dir):
                if ".git" in dirs:
                    dirs.remove(".git")

                for file_name in files:
                    extension = Path(file_name).suffix.lower()
                    if extension not in WIKI_FILE_TEXT_EXTENSIONS:
                        continue
                    file_path = Path(root) / file_name
                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            total_characters += len(f.read())
                    except Exception as exc:
                        LOG.debug(
                            "Skipping unreadable wiki file %s: %s", file_path, exc
                        )
        except Exception as exc:
            LOG.debug("Failed while scanning wiki for %s: %s", self.repo_name, exc)

        self.metrics["github_wiki_length"] = total_characters

    def _github_wiki_url(self) -> str | None:
        """Build the wiki git URL if the repository is hosted on GitHub; else None."""
        url = self.repo_url.rstrip("/")

        base = url[:-4] if url.endswith(".git") else url

        # HTTPS form: https://github.com/owner/repo
        https_match = re.match(
            r"^https://github\.com/([^/]+)/([^/]+)$", base, re.IGNORECASE
        )
        if https_match:
            owner, repo = https_match.groups()
            return f"https://github.com/{owner}/{repo}.wiki.git"

        return None

    def get_readme_completeness(self) -> None:
        """
        How many of common documentation-related phrase appear in the readme, as percentage.
        Score in range [0.0, 1.0].
        """
        readme_path = self.readme_path
        if not readme_path:
            self.metrics["readme_completeness"] = 0.0
            return

        with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        hits = sum(
            1
            for s in self.readme_section_names
            if re.search(s, content, re.RegexFlag.IGNORECASE)
        )

        self.metrics["readme_completeness"] = (
            (hits / len(self.readme_section_names))
            if self.readme_section_names
            else 0.0
        )

    def get_docstring_coverage(self) -> None:
        """Calculate fraction of functions/methods with docstrings."""
        cov = coverage.InterrogateCoverage(paths=[str(self.repo_dir)])
        results = cov.get_coverage()
        self.metrics["docstring_coverage"] = results.perc_covered / 100.0

    def get_documentation_percentage(self) -> None:
        """Compute fraction of documentation lines across all Python files in our repo."""
        try:
            result = subprocess.run(
                [
                    "pygount",
                    "--suffix=py",
                    str(self.repo_dir),
                    "--format=json",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=180,
            )

            if result.returncode != 0:
                LOG.error(
                    "Pygount exited with code %s for %s: %s",
                    result.returncode,
                    self.repo_name,
                    (result.stderr or "").strip(),
                )
                raise Exception("Pygount execution failed.")

            data = json.loads(result.stdout or "{}")
            summary = data.get("summary") or {}

            total_code = summary.get("totalCodeCount")
            if not isinstance(total_code, int) or total_code == 0:
                LOG.error("Pygount found no source code for: %s", self.repo_name)
                raise Exception(f"Pygount found no source code for: {self.repo_name}")

            documentation_percentage = summary.get("totalDocumentationPercentage")

            if not isinstance(documentation_percentage, float):
                LOG.error(
                    "Invalid documentation percentage value: %s",
                    documentation_percentage,
                )
                debug_tools.debug_reraise()

            self.metrics["documentation_percentage"] = documentation_percentage / 100.0

        except FileNotFoundError:
            LOG.error("Pygount executable not found.")
            debug_tools.debug_reraise()
        except json.JSONDecodeError as exc:
            LOG.exception(
                "Failed to parse pygount JSON for %s: %s", self.repo_name, exc
            )
            debug_tools.debug_reraise()
        except Exception as exc:
            LOG.exception("Pygount analysis failed for %s: %s", self.repo_name, exc)
            debug_tools.debug_reraise()

    def _find_docs_directory(self) -> Path | None:
        """Try to find the documentation directory in the repository."""
        dir_candidates = ["docs", "doc", "documentation", "docsrc", "sphinx"]

        for candidate in dir_candidates:
            candidate_path = self.repo_dir / candidate
            if candidate_path.is_dir():
                LOG.debug(
                    "Found documentation directory for %s: %s",
                    self.repo_name,
                    candidate_path,
                )
                return candidate_path

        LOG.warning(
            "Could not find documentation directory for repository %r", self.repo_name
        )
        return None

    def _extract_public_classes_and_methods_from_source(self) -> set[str]:
        """
        Extract public classes and methods from Python source code.

        Returns set of names like `function_name`, `ClassName` and `.class_method_name`.
        """
        public_items = set()

        python_files = list(self.repo_dir.rglob("*.py"))

        for py_file in python_files:
            if (
                ".venv" in py_file.parts
                or "venv" in py_file.parts
                or "__pycache__" in py_file.parts
            ):
                continue

            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()

                tree = ast.parse(source, filename=str(py_file))

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if node.name.startswith("_") or node.name.startswith("Test"):
                            continue
                        class_name = node.name
                        public_items.add(class_name)

                        for class_attribute in node.body:
                            if isinstance(class_attribute, ast.FunctionDef):
                                if class_attribute.name.startswith("_"):
                                    continue
                                method_name = f".{class_attribute.name}"
                                public_items.add(method_name)

                    elif isinstance(node, ast.FunctionDef):
                        if node.name.startswith("_"):
                            continue
                        if not any(
                            isinstance(parent, ast.ClassDef)
                            for parent in ast.walk(tree)
                        ):
                            public_items.add(node.name)

            except (SyntaxError, UnicodeDecodeError) as exc:
                LOG.debug("Could not parse %s: %s", py_file, exc)
            except Exception as exc:
                LOG.debug("Error processing %s: %s", py_file, exc)

        return public_items

    def _extract_public_items_with_metadata(self) -> dict[str, dict]:
        """
        Extract public classes and methods with metadata (arg counts, etc).

        Returns dict mapping names to metadata:
        {
            "ClassName": {"type": "class", "name": "ClassName"},
            "function_name": {"type": "function", "name": "function_name", "arg_count": 2},
            "method_name": {"type": "method", "name": "method_name", "class": "ClassName", "arg_count": 1}
        }
        """
        public_items = {}

        python_files = list(self.repo_dir.rglob("*.py"))

        for py_file in python_files:
            if (
                ".venv" in py_file.parts
                or "venv" in py_file.parts
                or "__pycache__" in py_file.parts
                or py_file.parts[-1].startswith("test")
            ):
                continue

            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()

                tree = ast.parse(source, filename=str(py_file))

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if node.name.startswith("_"):
                            continue

                        class_name = node.name
                        public_items[class_name] = {
                            "type": "class",
                            "name": class_name,
                        }

                        for item in node.body:
                            if (
                                not isinstance(item, ast.FunctionDef)
                                or item.name.startswith("_")
                                or ast_node_checks.is_method_property(item)
                            ):
                                continue

                            method_key = f"{class_name}.{item.name}"
                            arg_count = len(
                                [arg for arg in item.args.args if arg.arg != "self"]
                            )
                            public_items[method_key] = {
                                "type": "method",
                                "name": item.name,
                                "class": class_name,
                                "arg_count": arg_count,
                            }

                    elif isinstance(node, ast.FunctionDef):
                        # Check if this is a top-level function
                        if node.name.startswith("_"):
                            continue

                        is_top_level = True
                        for parent in ast.walk(tree):
                            if isinstance(parent, ast.ClassDef):
                                if node in parent.body:
                                    is_top_level = False
                                    break

                        if is_top_level:
                            arg_count = len(node.args.args)
                            public_items[node.name] = {
                                "type": "function",
                                "name": node.name,
                                "arg_count": arg_count,
                            }

            except (SyntaxError, UnicodeDecodeError) as exc:
                LOG.debug("Could not parse %s: %s", py_file, exc)
            except Exception as exc:
                LOG.debug("Error processing %s: %s", py_file, exc)

        return public_items

    @methodtools.lru_cache()
    def _extract_documentation_text(self) -> str:
        """Extract all text from documentation sources (docs folder, README, wiki)."""
        all_text = []

        if self.readme_path:
            try:
                with open(
                    self.readme_path, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    all_text.append(f.read())
            except Exception as exc:
                LOG.error("Could not read README %s: %s", self.readme_path, exc)

        if self.docs_dir and self.docs_dir.is_dir():
            for doc_file in self.docs_dir.rglob("*"):
                if not doc_file.is_file():
                    continue

                extension = doc_file.suffix.lower()
                if extension not in WIKI_FILE_TEXT_EXTENSIONS:
                    continue

                try:
                    with open(doc_file, "r", encoding="utf-8", errors="ignore") as f:
                        all_text.append(f.read())
                except Exception as exc:
                    LOG.error("Could not read doc file %s: %s", doc_file, exc)

        if self.wiki_dir.is_dir():
            for root, dirs, files in os.walk(self.wiki_dir):
                if ".git" in dirs:
                    dirs.remove(".git")

                for file_name in files:
                    extension = Path(file_name).suffix.lower()
                    if extension in WIKI_FILE_TEXT_EXTENSIONS:
                        file_path = Path(root) / file_name
                        try:
                            with open(
                                file_path, "r", encoding="utf-8", errors="ignore"
                            ) as f:
                                all_text.append(f.read())
                        except Exception as exc:
                            LOG.error("Could not read wiki file %s: %s", file_path, exc)

        return "\n\n".join(all_text)

    def _extract_references_from_documentation(self, doc_text: str) -> set[str]:
        """
        Extract references to classes and methods from documentation text.

        For functions/methods: Checks for signatures with parentheses and argument counts.
        For classes: Checks for mentions and inheritance patterns (e.g., class X(ClassName)).
        """
        documentation_referenced_objects = set()

        public_items_metadata = self._extract_public_items_with_metadata()

        for item_key, metadata in public_items_metadata.items():
            source_item_type = metadata["type"]

            if source_item_type == "class":
                source_class_name = metadata["name"]
                source_escaped_name = re.escape(source_class_name)

                # Direct mention as word boundary
                direct_pattern = rf"\b{source_escaped_name}\b"
                if re.search(direct_pattern, doc_text):
                    documentation_referenced_objects.add(item_key)
                    continue

                # Inheritance pattern: `class X(ClassName)` or `class X(..., ClassName)`.
                inheritance_pattern = (
                    rf"class\s+\w+\s*\([^)]*\b{source_escaped_name}\b[^)]*\)"
                )
                if re.search(inheritance_pattern, doc_text):
                    documentation_referenced_objects.add(item_key)

            elif source_item_type in ("function", "method"):
                source_func_name = metadata["name"]
                source_arg_count = metadata.get("arg_count", 0)

                escaped_source_func_name = re.escape(source_func_name)

                escaped_source_func_pattern = rf"\b{escaped_source_func_name}\s*\("

                matches = list(re.finditer(escaped_source_func_pattern, doc_text))
                if not matches:
                    continue

                for match in matches:
                    start_pos = match.end()
                    # Try to find the closing parenthesis.
                    depth = 1
                    i = start_pos
                    while i < len(doc_text) and depth > 0:
                        if doc_text[i] == "(":
                            depth += 1
                        elif doc_text[i] == ")":
                            depth -= 1
                        i += 1

                    args_section = doc_text[start_pos : i - 1].strip()

                    if not args_section:
                        if source_arg_count == 0:
                            documentation_referenced_objects.add(item_key)
                            break
                    else:
                        comma_count = args_section.count(",")
                        doc_arg_count = comma_count + 1

                        # Allow off-by-one for self/cls in methods.
                        # This might only misfire if documentation
                        # references a method with same name but different signature.
                        if abs(doc_arg_count - source_arg_count) <= 1:
                            documentation_referenced_objects.add(item_key)
                            break

        return documentation_referenced_objects

    @staticmethod
    def _extract_code_examples_from_documentation(doc_text: str) -> str:
        """Extract Python code examples from documentation (markdown/rst code blocks)."""
        code_examples = []

        md_code_pattern = r"```python\s*(.*?)```"
        rst_code_pattern = r"\.\. code-block::\s*python\s*(.*?)(?=\n\S|\Z)"

        for match in re.finditer(md_code_pattern, doc_text, re.DOTALL | re.IGNORECASE):
            code_examples.append(match.group(1))

        for match in re.finditer(rst_code_pattern, doc_text, re.DOTALL | re.IGNORECASE):
            code_examples.append(match.group(1))

        return "\n\n".join(code_examples)

    def get_documentation_up_to_date(self) -> None:
        """Calculate ratio of public classes/methods documented to total public classes/methods."""
        try:
            public_items = self.public_source_items

            if not public_items:
                LOG.warning("No public items found in source for %s", self.repo_name)
                self.metrics["documentation_up_to_date"] = 0.0
                return

            doc_text = self._extract_documentation_text()

            if not doc_text.strip():
                LOG.warning("No documentation text found for %s", self.repo_name)
                self.metrics["documentation_up_to_date"] = 0.0
                return

            documented_items = self._extract_references_from_documentation(doc_text)

            ratio = len(documented_items) / len(public_items) if public_items else 0.0
            self.metrics["documentation_up_to_date"] = ratio

            LOG.debug(
                "Documentation up-to-date metric for %s: %d/%d = %.2f",
                self.repo_name,
                len(documented_items),
                len(public_items),
                ratio,
            )

        except Exception as exc:
            LOG.exception(
                "Failed to calculate documentation up-to-date for %s: %s",
                self.repo_name,
                exc,
            )
            self.metrics["documentation_up_to_date"] = 0.0

    def get_documentation_readability(self) -> None:
        """Calculate Flesch-Kincaid readability score for documentation."""
        try:
            doc_text = self._extract_documentation_text()

            if not doc_text.strip():
                LOG.warning(
                    "No documentation text for readability analysis for %s",
                    self.repo_name,
                )
                self.metrics["documentation_readability"] = 0.0
                return

            readability_analyzer = Readability(doc_text)
            flesch_kincaid_score = readability_analyzer.flesch_kincaid()

            score_value = flesch_kincaid_score.score

            self.metrics["documentation_readability"] = score_value

            LOG.debug(
                "Documentation readability for %s: %.2f", self.repo_name, score_value
            )

        except Exception as exc:
            LOG.exception(
                "Failed to calculate documentation readability for %s: %s",
                self.repo_name,
                exc,
            )
            self.metrics["documentation_readability"] = 0.0

    def get_code_examples_ratio(self) -> None:
        """Calculate ratio of public methods appearing in code examples to total public methods."""
        try:
            public_items = self.public_source_items
            public_methods = {item for item in public_items if "." in item}

            if not public_methods:
                LOG.warning("No public methods found in source for %s", self.repo_name)
                self.metrics["code_examples_ratio"] = 0.0
                return

            doc_text = self._extract_documentation_text()
            code_examples = self._extract_code_examples_from_documentation(doc_text)

            if not code_examples.strip():
                LOG.warning(
                    "No code examples found in documentation for %s", self.repo_name
                )
                self.metrics["code_examples_ratio"] = 0.0
                return

            methods_in_examples = set()
            for method in public_methods:
                escaped_method = re.escape(method)
                pattern = rf"\b{escaped_method}\b"

                if re.search(pattern, code_examples):
                    methods_in_examples.add(method)

            ratio = (
                len(methods_in_examples) / len(public_methods)
                if public_methods
                else 0.0
            )
            self.metrics["code_examples_ratio"] = ratio

            LOG.debug(
                "Code examples ratio for %s: %d/%d = %.2f",
                self.repo_name,
                len(methods_in_examples),
                len(public_methods),
                ratio,
            )

        except Exception as exc:
            LOG.exception(
                "Failed to calculate code examples ratio for %s: %s",
                self.repo_name,
                exc,
            )
            self.metrics["code_examples_ratio"] = 0.0


def schedule_documentation_metrics(
    repo_urls: Iterable[str], output_csv: Path | None = None
) -> None:
    """Schedule documentation metrics collection for a set of repositories and write to CSV."""
    output_csv = output_csv or (cvar.data_dir / "documentation_metrics.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    metrics_list: list[DocumentationMetrics] = []
    for url in tqdm.tqdm(repo_urls):
        try:
            collector = DocumentationMetricsCollect(url)
            repo_metrics = collector.collect_metrics()
            metrics_list.append(repo_metrics.copy())
            _save_metrics(metrics_list, output_csv)
        except Exception as exc:
            LOG.exception("Failed to collect metrics for %s: %s", url, exc)
            _save_metrics(metrics_list, output_csv)
            debug_tools.debug_reraise()

    _save_metrics(metrics_list, output_csv)


def _save_metrics(metrics_list: list[DocumentationMetrics], output_csv: Path | None):
    df = pandas.DataFrame(
        metrics_list, columns=tuple(documentation_metrics_dictionary.keys())
    )
    df.to_csv(output_csv, index=False)
    LOG.info("Saved documentation metrics. Wrote %d rows to %s", len(metrics_list), output_csv)


def example_usage():
    """Run on a tiny example."""
    repo_list = [
        "https://github.com/Digital-Thought/dtPyAppFramework",
        "https://github.com/qwerltaz/lucerna"
    ]
    schedule_documentation_metrics(repo_list)

def main():
    """Run on full target dataset."""
    with open(cvar.resources_dir / "security_libraries_dependents_count.json", "r", encoding="utf-8") as f:
        dependents_data = json.load(f)

    repo_urls = [item["repo_url"] for item in dependents_data if item.get("repo_url")]
    schedule_documentation_metrics(repo_urls)


if __name__ == "__main__":
    main()
