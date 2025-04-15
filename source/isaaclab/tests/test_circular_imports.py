import unittest

class TestCircularImports(unittest.TestCase):
    def test_circular_imports(self):
        try:
            from isaaclab_tasks.manager_based.manipulation.stack.config.franka import stack_lift_joint_pos_env_cfg
        except ImportError as e:
            self.fail(f"Circular import detected: {e}")

if __name__ == '__main__':
    unittest.main()