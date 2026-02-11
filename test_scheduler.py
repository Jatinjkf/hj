import unittest
from unittest.mock import MagicMock, patch
import utils
import main_pro

class TestScheduler(unittest.TestCase):
    def test_get_scheduler_list(self):
        schedulers = utils.get_scheduler_list()
        self.assertIn("DPM++ 2M Karras", schedulers)
        self.assertIn("Euler", schedulers)

    def test_configure_scheduler(self):
        pipe = MagicMock()
        pipe.scheduler.config = {"num_train_timesteps": 1000}

        # Test Euler
        utils.configure_scheduler(pipe, "Euler")
        # Check if scheduler was replaced (mock behavior is minimal, but we check if it runs)
        # To be precise, we need to mock diffusers classes, but for now we just check no crash.

if __name__ == "__main__":
    unittest.main()
