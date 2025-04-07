import importlib.util
import datetime
import os
import numpy as np
import ast

def detect_hardcode_or_pass(func_node):
    """Detects if a function is a stub or hardcoded (e.g., pass, return constant)."""
    if not func_node.body:
        return True
    stmt = func_node.body[0]
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Return):
        return isinstance(stmt.value, (ast.Constant, ast.Str, ast.Num))
    return False

def test_student_code(solution_path):
    report_dir = os.path.join(os.path.dirname(__file__), "..", "student_workspace")
    report_path = os.path.join(report_dir, "report.txt")
    os.makedirs(report_dir, exist_ok=True)

    # AST-based function-level hardcoded detection
    with open(solution_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    bad_funcs = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for func in node.body:
                if isinstance(func, ast.FunctionDef) and detect_hardcode_or_pass(func):
                    bad_funcs.add(func.name)

    spec = importlib.util.spec_from_file_location("student_module", solution_path)
    student_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_module)
    Analyzer = student_module.OrderDataAnalyzer

    report_lines = [f"=== NumPy Order Analyzer Test Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="]

    randomized_failures = set()
    analyzer = Analyzer()

    # Independent functional checks
    try:
        if not np.array_equal(analyzer.create_order_array([10, 20]), np.array([10.0, 20.0])):
            randomized_failures.add("create_order_array")
    except Exception:
        randomized_failures.add("create_order_array")

    try:
        if analyzer.validate_order_array(np.array([])) is not False:
            randomized_failures.add("validate_order_array")
    except Exception:
        randomized_failures.add("validate_order_array")

    try:
        if analyzer.validate_order_array(np.array([1, -5])) is not False:
            randomized_failures.add("validate_order_array")
    except Exception:
        randomized_failures.add("validate_order_array")

    try:
        if analyzer.apply_discount(np.array([200]))[0] != 180.0:
            randomized_failures.add("apply_discount")
    except Exception:
        randomized_failures.add("apply_discount")

    try:
        if analyzer.format_order_amounts(np.array([100]))[0] != "$100.00":
            randomized_failures.add("format_order_amounts")
    except Exception:
        randomized_failures.add("format_order_amounts")

    try:
        expected = (300.0, 100.0, 100.0)
        if analyzer.compute_order_summary(np.array([100.0, 100.0, 100.0])) != expected:
            randomized_failures.add("compute_order_summary")
    except Exception:
        randomized_failures.add("compute_order_summary")

    try:
        out = analyzer.flag_high_value_orders(np.array([1, 2, 100]))
        if not np.array_equal(out, np.array(["Normal", "Normal", "High"])):
            randomized_failures.add("flag_high_value_orders")
    except Exception:
        randomized_failures.add("flag_high_value_orders")

    # Define Test Matrix
    test_cases = [
        ("Visible", "TC1: Create Order Array", "create_order_array", [120.5, 250.0, 75.3, 99.99], np.array([120.5, 250.0, 75.3, 99.99])),
        ("Visible", "TC2: Validate Negative", "validate_order_array", np.array([100, -50, 200]), False),
        ("Visible", "TC3: Compute Summary", "compute_order_summary", np.array([120.5, 250.0, 75.3, 99.99]), (545.79, 136.4475, 250.0)),
        ("Visible", "TC4: Apply Discount", "apply_discount", np.array([100, 200, 300]), np.array([100.0, 180.0, 270.0])),
        ("Visible", "TC5: Flag High Orders", "flag_high_value_orders", np.array([100, 200, 50]), np.array(["Normal", "High", "Normal"])),
        ("Hidden", "TC6: Format Currency", "format_order_amounts", np.array([100, 250.5]), np.array(["$100.00", "$250.50"])),
        ("Hidden", "TC7: Validate Empty", "validate_order_array", np.array([]), False),
        ("Hidden", "TC8: Discount on $200", "apply_discount", np.array([200.0]), np.array([180.0]))
    ]

    for i, (section, desc, func, arg, expected) in enumerate(test_cases, 1):
        try:
            analyzer = Analyzer()
            method = getattr(analyzer, func)
            result = method(arg)

            if func in randomized_failures:
                msg = f"❌ {section} {desc} failed | Reason: Random logic failure"
            elif func in bad_funcs:
                msg = f"❌ {section} {desc} failed | Reason: Hardcoded/stub function"
            else:
                if isinstance(expected, np.ndarray):
                    passed = np.allclose(result, expected) if result.dtype.kind == "f" else np.array_equal(result, expected)
                else:
                    passed = result == expected
                msg = f"✅ {section} {desc}" if passed else f"❌ {section} {desc} failed | Expected={expected}, Got={result}"

        except Exception as e:
            msg = f"❌ {section} {desc} crashed | Error: {str(e)}"

        print(msg)
        report_lines.append(msg)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
