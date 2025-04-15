import unittest

class TestImports(unittest.TestCase):
    def test_imports(self):
        try:
            import isaaclab_tasks
            import isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_lift_joint_pos_env_cfg
        except ImportError as e:
            self.fail(f"Import failed: {e}")

if __name__ == '__main__':
    unittest.main()