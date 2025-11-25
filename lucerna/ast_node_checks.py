"""Property checks for AST nodes."""

import ast


def is_method_property(node: ast.AST | ast.FunctionDef) -> bool:
    """Check if a given AST node represents a method decorated with @property."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "property":
            return True

    return False