import json

import pandas as pd

import cvar


def sec_lib_dependents_count_table():
    security_libraries_path = ".." / cvar.data_dir / "licma_security_libraries.json"
    dependents_path = ".." / cvar.data_dir / "security_libraries_dependents.json"

    target_security_libraries_names = {"cryptography", "M2Crypto", "PyCrypto", "PyNaCl"}

    with open(security_libraries_path, "r", encoding="utf-8") as f:
        security_libraries: list[dict] = json.load(f)

    with open(dependents_path, "r", encoding="utf-8") as f:
        dependents: dict[str, dict] = json.load(f)

    security_libraries = [
        lib for lib in security_libraries if lib["name"] in target_security_libraries_names
    ]

    dependents_counts = dict()
    for lib in security_libraries:
        lib_name = lib["name"]
        if lib_name not in dependents:
            raise ValueError(
                f"Dependents data not found for security library: {lib_name}"
            )

        dependents_counts[lib_name] = len(dependents[lib_name]["dependents"])

    dependents_count_df = pd.DataFrame.from_dict(
        dependents_counts, orient="index", columns=["dependents_count"]
    )

    print(dependents_count_df.to_latex())


def doc_metrics_table():
    licma_doc_metrics_path = ".." / cvar.data_dir / "licma_documentation_metrics.csv"

    doc_metrics_df = pd.read_csv(licma_doc_metrics_path, index_col="repo_name")
    doc_metrics_df = doc_metrics_df.sort_index()

    doc_metrics_df = doc_metrics_df.transpose()
    doc_metrics_df = doc_metrics_df.drop(columns=["M2Crypto"])
    print(doc_metrics_df.to_latex())


def dependents_stats():
    with open(".." / cvar.data_dir / "vulnerabilities.json", "r", encoding="utf-8") as f:
        vulnerabilities_json = json.load(f)

    vulnerabilities_json.pop("M2Crypto")

    vulnerabilities_summary = pd.DataFrame(index=vulnerabilities_json.keys())
    vulnerabilities_summary["total"] = [len(dependents) for dependents in vulnerabilities_json.values()]
    vulnerabilities_summary["computed"] = [
        sum([1 for dep in dependents.values() if dep is not None]) for dependents in vulnerabilities_json.values()]
    vulnerabilities_summary["not computed"] = [
        sum([1 for dep in dependents.values() if dep is None]) for dependents in vulnerabilities_json.values()]
    vulnerabilities_summary["non-zero"] = [
        sum([1 for dep in dependents.values() if dep]) for dependents in vulnerabilities_json.values()]
    vulnerabilities_summary["non-zero percentage"] = vulnerabilities_summary["non-zero"] / vulnerabilities_summary[
        "computed"] * 100
    vulnerabilities_summary["average"] = [
        sum([dep for dep in dependents.values() if dep]) / vulnerabilities_summary.at[name, "non-zero"]
        if vulnerabilities_summary.at[name, "non-zero"] > 0 else 0
        for name, dependents in vulnerabilities_json.items()
    ]

    # sort by name
    vulnerabilities_summary = vulnerabilities_summary.sort_index()

    vulnerabilities_summary = vulnerabilities_summary.transpose()
    print(vulnerabilities_summary.to_latex(
        formatters={"name": str.upper},
        float_format="%.2f"))


if __name__ == "__main__":
    doc_metrics_table()
