import importlib.util
import os

def run_tests():
    solution_path = os.path.join(os.path.dirname(__file__), "student_workspace", "solution.py")
    driver_path = os.path.join(os.path.dirname(__file__), "secret_tests", "driver.py")

    spec = importlib.util.spec_from_file_location("driver", driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.test_student_code(solution_path)

if __name__ == "__main__":
    run_tests()
