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

    doc_metrics_df = pd.read_csv(licma_doc_metrics_path).transpose()

    print(doc_metrics_df.to_latex())


if __name__ == "__main__":
    doc_metrics_table()
