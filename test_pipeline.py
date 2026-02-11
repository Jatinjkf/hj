import unittest
from unittest.mock import MagicMock, patch
import os
import utils
import main_pro

class TestTinySD(unittest.TestCase):

    def test_ensure_extension(self):
        self.assertEqual(utils.ensure_extension("test"), "test.png")
        self.assertEqual(utils.ensure_extension("test.jpg"), "test.jpg")
        self.assertEqual(utils.ensure_extension(""), "output.png")

    @patch("main_pro.StableDiffusionPipeline")
    @patch("main_pro.detect_intel_gpu")
    def test_get_pipeline_cpu(self, mock_detect, mock_pipeline):
        mock_detect.return_value = None # No Intel GPU

        # Mock pipeline instance
        mock_pipe = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_pipe

        # Test get_pipeline
        pipe = main_pro.get_pipeline("txt2img", "cpu")

        # Verify it called from_pretrained with use_safetensors=True first
        mock_pipeline.from_pretrained.assert_called()
        call_args = mock_pipeline.from_pretrained.call_args
        # Depending on how many times it was called (retry logic), check the args
        # My code tries True first.
        self.assertTrue(call_args[1].get('use_safetensors', False) or mock_pipeline.from_pretrained.call_args_list[0][1].get('use_safetensors', False))

    @patch("main_pro.StableDiffusionPipeline")
    @patch("utils.get_optimal_scheduler")
    def test_scheduler_replacement(self, mock_get_scheduler, mock_pipeline):
         mock_pipe = MagicMock()
         mock_pipeline.from_pretrained.return_value = mock_pipe
         mock_get_scheduler.return_value = "MockScheduler"

         pipe = main_pro.get_pipeline("txt2img", "cpu")

         self.assertEqual(pipe.scheduler, "MockScheduler")

if __name__ == "__main__":
    unittest.main()
