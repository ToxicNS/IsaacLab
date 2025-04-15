def test_circular_imports():
    try:
        import isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg
        assert False, "Import should have failed due to circular import"
    except ImportError:
        assert True  # Expected behavior

def test_imports():
    try:
        import isaaclab_tasks
        import isaaclab_tasks.manager_based.manipulation.stack.config.franka
        assert True  # Both imports should succeed
    except ImportError:
        assert False, "Import failed for isaaclab_tasks or its submodules"