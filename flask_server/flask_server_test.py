import unittest
from unittest.mock import patch, MagicMock
from camera_server import CameraServer, ArtifactInfo  # Ensure this import matches your file structure

class TestCameraServer(unittest.TestCase):

    def setUp(self):
        # Patch rospy.init_node to prevent actual ROS node initialization
        patcher = patch('rospy.init_node')
        self.mock_init_node = patcher.start()
        self.addCleanup(patcher.stop)

        # Patch rospy.Subscriber to prevent actual subscriptions
        self.mock_subscriber = patch('rospy.Subscriber').start()
        self.addCleanup(patch.stopall)

        # Instantiate the CameraServer
        self.server = CameraServer()

        # Mock Flask app
        self.server.app = MagicMock()
        self.server.app.run = MagicMock()

    def test_query_near(self):
        self.server.at_waypoint = True
        response = self.server.query_near()
        self.assertTrue(response.json["at_waypoint"])

    def test_receive_waypoint(self):
        mock_request = MagicMock()
        mock_request.get_json.return_value = {"x": 1, "y": 2, "z": 3}
        with patch('flask.request', mock_request):
            response = self.server.receive_waypoint()
            self.assertEqual(self.server.waypoint, {"x": 1, "y": 2, "z": 3})
            self.assertEqual(response.status_code, 200)

    def test_image_callback(self):
        mock_msg = MagicMock()
        self.server.image_callback(mock_msg, 'front')
        self.assertIsNotNone(self.server.images['front'])

    def test_posearray_callback(self):
        mock_msg = MagicMock()
        mock_msg.poses = [MagicMock()]
        self.server.posearray_callback(mock_msg)
        self.assertIsNotNone(self.server.posearray_spot)

    def test_posearray_frontier_callback(self):
        mock_msg = MagicMock()
        mock_msg.poses = [MagicMock()]
        self.server.posearray_frontier_callback(mock_msg)
        self.assertIsNotNone(self.server.posearray_frontier)

    def test_projection_artifacts_callback(self):
        mock_msg = ArtifactInfo(MagicMock(), 1, {"x": 1, "y": 2, "z": 3}, "object", 0.9)
        self.server.projection_artifacts_callback(mock_msg)
        self.assertGreater(len(self.server.projection_artifacts), 0)

    def test_concatenate_images(self):
        self.server.images['front'] = MagicMock()
        self.server.images['left'] = MagicMock()
        self.server.images['right'] = MagicMock()
        concatenated_image = self.server.concatenate_images()
        self.assertIsNotNone(concatenated_image)

    def test_send_image(self):
        self.server.images['front'] = MagicMock()
        response = self.server.send_image()
        self.assertEqual(response[1], 200)

    def test_run_flask(self):
        self.server.run_flask()
        self.server.app.run.assert_called_once()

    def test_integration(self):
        # Integration test simulating the overall functionality of the CameraServer
        # This test should be expanded based on the actual application logic and requirements

if __name__ == '__main__':
    unittest.main()
