"""
Unit tests for ProcessRewardFunction

Tests all sub-reward components and composite reward function
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.process_metrics import ProcessRewardFunction, ProcessMetrics


class TestProcessRewardFunction(unittest.TestCase):
    """Test cases for ProcessRewardFunction"""

    def setUp(self):
        """Set up test fixtures"""
        self.training_traces = [
            ['Start', 'Submit', 'Review', 'Approve', 'End'],
            ['Start', 'Submit', 'Review', 'Reject', 'End'],
            ['Start', 'Submit', 'Approve', 'End'],
        ]

        self.reward_fn = ProcessRewardFunction(
            training_traces=self.training_traces,
            weights={
                'validity': 0.30,
                'fitness': 0.25,
                'conformance': 0.25,
                'diversity': 0.20
            }
        )

    # ─────────────────────────────────────────────────────────
    # VALIDITY TESTS
    # ─────────────────────────────────────────────────────────

    def test_validity_perfect_trace(self):
        """Test validity score for perfect trace"""
        trace = ['Start', 'Submit', 'Review', 'Approve', 'End']
        score = self.reward_fn.validity_score([trace])[0]
        self.assertEqual(score, 1.0, "Perfect trace should have validity = 1.0")

    def test_validity_empty_trace(self):
        """Test validity score for empty trace"""
        trace = []
        score = self.reward_fn.validity_score([trace])[0]
        self.assertEqual(score, 0.0, "Empty trace should have validity = 0.0")

    def test_validity_missing_start(self):
        """Test validity score for trace missing Start"""
        trace = ['Submit', 'Review', 'Approve', 'End']
        score = self.reward_fn.validity_score([trace])[0]
        self.assertLess(score, 1.0, "Trace without Start should have reduced validity")
        self.assertGreater(score, 0.0, "Trace should not be completely invalid")

    def test_validity_missing_end(self):
        """Test validity score for trace missing End"""
        trace = ['Start', 'Submit', 'Review', 'Approve']
        score = self.reward_fn.validity_score([trace])[0]
        self.assertLess(score, 1.0, "Trace without End should have reduced validity")

    def test_validity_duplicate_activities(self):
        """Test validity score for trace with consecutive duplicates"""
        trace = ['Start', 'Submit', 'Submit', 'End']
        score = self.reward_fn.validity_score([trace])[0]
        self.assertLess(score, 1.0, "Duplicate activities should reduce validity")

    def test_validity_end_in_middle(self):
        """Test validity score for trace with End not at last position"""
        trace = ['Start', 'End', 'Submit']
        score = self.reward_fn.validity_score([trace])[0]
        self.assertLess(score, 0.5, "'End' in middle should severely reduce validity")

    def test_validity_wrong_length(self):
        """Test validity score for trace with wrong length"""
        # Too short
        trace_short = ['Start', 'End']
        score_short = self.reward_fn.validity_score([trace_short])[0]
        self.assertLess(score_short, 1.0, "Too short trace should have reduced validity")

        # Too long (>20 activities)
        trace_long = ['Start'] + ['Activity'] * 20 + ['End']
        score_long = self.reward_fn.validity_score([trace_long])[0]
        self.assertLess(score_long, 1.0, "Too long trace should have reduced validity")

    # ─────────────────────────────────────────────────────────
    # FITNESS TESTS
    # ─────────────────────────────────────────────────────────

    def test_fitness_valid_trace(self):
        """Test fitness score for valid trace"""
        trace = ['Start', 'Submit', 'Review', 'Approve', 'End']
        score = self.reward_fn.fitness_score([trace])[0]
        self.assertGreaterEqual(score, 0.0, "Fitness should be >= 0")
        self.assertLessEqual(score, 1.0, "Fitness should be <= 1")

    def test_fitness_empty_trace(self):
        """Test fitness score for empty trace"""
        trace = []
        score = self.reward_fn.fitness_score([trace])[0]
        self.assertEqual(score, 0.0, "Empty trace should have fitness = 0.0")

    def test_fitness_with_core_activities(self):
        """Test that traces with core activities have higher fitness"""
        trace_with_core = ['Start', 'Submit', 'Review', 'Approve', 'End']
        trace_without_core = ['Start', 'Activity1', 'Activity2', 'End']

        score_with = self.reward_fn.fitness_score([trace_with_core])[0]
        score_without = self.reward_fn.fitness_score([trace_without_core])[0]

        self.assertGreater(score_with, score_without,
                          "Trace with core activities should have higher fitness")

    # ─────────────────────────────────────────────────────────
    # CONFORMANCE TESTS
    # ─────────────────────────────────────────────────────────

    def test_conformance_perfect_match(self):
        """Test conformance for trace matching expected pattern"""
        trace = ['Start', 'Submit', 'Review', 'Approve', 'End']
        score = self.reward_fn.conformance_score([trace])[0]
        self.assertGreaterEqual(score, 0.5, "Perfect match should have high conformance")

    def test_conformance_empty_trace(self):
        """Test conformance for empty trace"""
        trace = []
        score = self.reward_fn.conformance_score([trace])[0]
        self.assertEqual(score, 0.0, "Empty trace should have conformance = 0.0")

    def test_conformance_partial_match(self):
        """Test conformance for partially matching trace"""
        trace = ['Start', 'Submit', 'Approve', 'End']  # Skips Review
        score = self.reward_fn.conformance_score([trace])[0]
        self.assertGreater(score, 0.0, "Partial match should have some conformance")
        self.assertLess(score, 1.0, "Partial match should not be perfect")

    # ─────────────────────────────────────────────────────────
    # DIVERSITY TESTS
    # ─────────────────────────────────────────────────────────

    def test_diversity_identical_to_training(self):
        """Test diversity for trace identical to training"""
        trace = ['Start', 'Submit', 'Review', 'Approve', 'End']  # Same as training[0]
        score = self.reward_fn.diversity_score([trace])[0]
        self.assertEqual(score, 0.0, "Identical trace should have diversity = 0.0")

    def test_diversity_very_different(self):
        """Test diversity for very different trace"""
        trace = ['Start', 'NewActivity1', 'NewActivity2', 'NewActivity3', 'End']
        score = self.reward_fn.diversity_score([trace])[0]
        self.assertGreater(score, 0.5, "Very different trace should have high diversity")

    def test_diversity_slightly_different(self):
        """Test diversity for slightly different trace"""
        trace = ['Start', 'Submit', 'Check', 'Approve', 'End']  # One activity different
        score = self.reward_fn.diversity_score([trace])[0]
        self.assertGreater(score, 0.0, "Slightly different trace should have some diversity")
        self.assertLess(score, 0.8, "Slightly different trace should not have max diversity")

    # ─────────────────────────────────────────────────────────
    # COMPOSITE REWARD TESTS
    # ─────────────────────────────────────────────────────────

    def test_compute_reward_returns_correct_shape(self):
        """Test that compute_reward returns correct shape"""
        traces = [
            ['Start', 'Submit', 'Review', 'Approve', 'End'],
            ['Start', 'Submit', 'Approve', 'End'],
        ]
        rewards = self.reward_fn.compute_reward(traces)
        self.assertEqual(rewards.shape, (2, 1), "Reward shape should be (batch, 1)")

    def test_compute_reward_in_range(self):
        """Test that rewards are in [0,1] range"""
        traces = [
            ['Start', 'Submit', 'Review', 'Approve', 'End'],
            ['Start', 'Submit', 'Approve', 'End'],
            [],  # Edge case: empty
        ]
        rewards = self.reward_fn.compute_reward(traces)

        for reward in rewards.flatten():
            self.assertGreaterEqual(reward, 0.0, "Reward should be >= 0")
            self.assertLessEqual(reward, 1.0, "Reward should be <= 1")

    def test_compute_reward_perfect_trace(self):
        """Test reward for perfect trace"""
        trace = ['Start', 'Submit', 'Review', 'Approve', 'End']
        reward = self.reward_fn.compute_reward([trace])[0, 0]
        self.assertGreater(reward, 0.5, "Perfect trace should have reward > 0.5")

    def test_compute_reward_poor_trace(self):
        """Test reward for poor quality trace"""
        trace = []  # Empty trace
        reward = self.reward_fn.compute_reward([trace])[0, 0]
        self.assertLess(reward, 0.3, "Poor trace should have low reward")

    def test_evaluate_batch(self):
        """Test evaluate_batch returns correct metrics"""
        traces = [
            ['Start', 'Submit', 'Review', 'Approve', 'End'],
            ['Start', 'Submit', 'Approve', 'End'],
        ]
        metrics = self.reward_fn.evaluate_batch(traces)

        required_keys = [
            'validity_mean', 'validity_std',
            'fitness_mean', 'fitness_std',
            'conformance_mean', 'conformance_std',
            'diversity_mean', 'diversity_std',
            'reward_mean', 'reward_std',
            'high_quality_rate'
        ]

        for key in required_keys:
            self.assertIn(key, metrics, f"Metrics should contain '{key}'")
            self.assertIsInstance(metrics[key], (float, np.floating),
                                f"Metric '{key}' should be float")

    # ─────────────────────────────────────────────────────────
    # HELPER FUNCTION TESTS
    # ─────────────────────────────────────────────────────────

    def test_edit_distance(self):
        """Test edit distance calculation"""
        seq1 = ['A', 'B', 'C']
        seq2 = ['A', 'B', 'C']
        dist = ProcessRewardFunction.edit_distance(seq1, seq2)
        self.assertEqual(dist, 0, "Identical sequences should have distance 0")

        seq1 = ['A', 'B', 'C']
        seq2 = ['A', 'D', 'C']
        dist = ProcessRewardFunction.edit_distance(seq1, seq2)
        self.assertEqual(dist, 1, "One substitution should give distance 1")

        seq1 = ['A', 'B']
        seq2 = ['A', 'B', 'C']
        dist = ProcessRewardFunction.edit_distance(seq1, seq2)
        self.assertEqual(dist, 1, "One insertion should give distance 1")

    def test_longest_common_subsequence(self):
        """Test LCS calculation"""
        seq1 = ['A', 'B', 'C', 'D']
        seq2 = ['A', 'B', 'C', 'D']
        lcs = self.reward_fn._longest_common_subsequence(seq1, seq2)
        self.assertEqual(lcs, 4, "Identical sequences should have LCS = length")

        seq1 = ['A', 'B', 'C']
        seq2 = ['A', 'D', 'C']
        lcs = self.reward_fn._longest_common_subsequence(seq1, seq2)
        self.assertEqual(lcs, 2, "LCS should be 2 (A, C)")


class TestProcessMetrics(unittest.TestCase):
    """Test cases for ProcessMetrics helper class"""

    def test_valid_traces(self):
        """Test filtering valid traces"""
        traces = [
            ['Start', 'Submit', 'End'],
            ['Submit', 'End'],  # Missing Start
            ['Start', 'Submit', 'Review', 'End'],
            [],  # Empty
        ]
        valid = ProcessMetrics.valid_traces(traces)
        self.assertEqual(len(valid), 2, "Should return 2 valid traces")

    def test_unique_traces(self):
        """Test filtering unique traces"""
        traces = [
            ['Start', 'Submit', 'End'],
            ['Start', 'Submit', 'End'],  # Duplicate
            ['Start', 'Review', 'End'],
        ]
        unique = ProcessMetrics.unique_traces(traces)
        self.assertEqual(len(unique), 2, "Should return 2 unique traces")

    def test_novel_traces(self):
        """Test filtering novel traces"""
        training = [
            ['Start', 'Submit', 'End'],
            ['Start', 'Review', 'End'],
        ]
        traces = [
            ['Start', 'Submit', 'End'],  # Not novel
            ['Start', 'Check', 'End'],   # Novel
        ]
        novel = ProcessMetrics.novel_traces(traces, training)
        self.assertEqual(len(novel), 1, "Should return 1 novel trace")

    def test_compute_statistics(self):
        """Test statistics computation"""
        traces = [
            ['Start', 'Submit', 'End'],
            ['Start', 'Submit', 'Review', 'End'],
            ['Start', 'End'],
        ]
        stats = ProcessMetrics.compute_statistics(traces)

        self.assertEqual(stats['total_traces'], 3)
        self.assertEqual(stats['min_length'], 2)
        self.assertEqual(stats['max_length'], 4)
        self.assertAlmostEqual(stats['avg_length'], 3.0, places=1)


class TestRewardFunctionEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def setUp(self):
        """Set up test fixtures"""
        self.reward_fn = ProcessRewardFunction(
            training_traces=[['Start', 'Submit', 'End']],
            weights={'validity': 0.25, 'fitness': 0.25, 'conformance': 0.25, 'diversity': 0.25}
        )

    def test_batch_processing(self):
        """Test processing multiple traces in batch"""
        traces = [['Start', 'Submit', 'End'] for _ in range(10)]
        rewards = self.reward_fn.compute_reward(traces)
        self.assertEqual(rewards.shape, (10, 1), "Batch processing should work")

    def test_empty_training_traces(self):
        """Test reward function with no training traces"""
        reward_fn = ProcessRewardFunction(training_traces=[])
        trace = ['Start', 'Submit', 'End']
        reward = reward_fn.compute_reward([trace])
        self.assertIsInstance(reward[0, 0], (float, np.floating),
                            "Should handle empty training set")

    def test_weights_normalization(self):
        """Test that weights are normalized to sum to 1"""
        reward_fn = ProcessRewardFunction(
            weights={'validity': 1.0, 'fitness': 1.0, 'conformance': 1.0, 'diversity': 1.0}
        )
        weight_sum = sum(reward_fn.weights.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=5,
                              msg="Weights should be normalized to sum to 1")


def run_tests():
    """Run all tests"""
    print("=" * 70)
    print("Running ProcessRewardFunction Tests")
    print("=" * 70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestProcessRewardFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestRewardFunctionEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")

    print("=" * 70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
