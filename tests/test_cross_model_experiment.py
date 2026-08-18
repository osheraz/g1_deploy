from __future__ import annotations

from types import SimpleNamespace
import unittest

from experiments import run_cross_model_23dof_adaptation as campaign


class CrossModelCampaignPlanTest(unittest.TestCase):
    @staticmethod
    def _args(stage, *, models=campaign.MODELS, candidates=None, seeds=None):
        return SimpleNamespace(
            stage=stage,
            models=models,
            candidates=candidates,
            seeds=seeds,
        )

    def test_baseline_is_matched_three_model_protocol(self):
        plan = campaign._plan(self._args("baseline"))
        self.assertEqual(len(plan), 60)
        for model in campaign.MODELS:
            rows = [row for row in plan if row["model"] == model]
            self.assertEqual(len(rows), 20)
            self.assertEqual(
                sum(row["candidate"] == "native29" for row in rows), 2
            )

    def test_search_covers_all_candidates_and_five_fault_cells(self):
        plan = campaign._plan(
            self._args("search", models=("low-latency", "sonic-v1.1"))
        )
        self.assertEqual(
            len(plan),
            2 * len(campaign.SEARCH_CANDIDATES) * len(campaign.SEARCH_CONDITIONS),
        )
        self.assertTrue(all(row["embodiment"] == "g1_23dof" for row in plan))

    def test_finalists_include_native_reference_for_every_seed(self):
        plan = campaign._plan(
            self._args(
                "finalists",
                models=("low-latency",),
                candidates=("masks",),
            )
        )
        expected_per_seed = 2 + len(campaign.CONDITIONS)
        self.assertEqual(len(plan), len(campaign.DEFAULT_SEEDS) * expected_per_seed)
        for seed in campaign.DEFAULT_SEEDS:
            native = [
                row for row in plan
                if row["seed"] == seed and row["candidate"] == "native29"
            ]
            self.assertEqual({row["condition"] for row in native}, {"healthy", "lowfric"})

    def test_experiment_commands_are_mujoco_only_and_seeded(self):
        command = campaign._command(
            model="low-latency",
            embodiment="g1_23dof",
            candidate="masks",
            condition="mild",
            seed=4321,
            record=campaign.DEFAULT_OUTPUT / "test",
            steps=1800,
        )
        text = " ".join(command)
        self.assertIn("sim2sim_sonic.py", text)
        self.assertIn("--planner-seed 4321", text)
        self.assertIn("--axes=0,1,0", command)
        self.assertNotIn("deploy_eskin.py", text)
        self.assertNotIn("--net", command)


if __name__ == "__main__":
    unittest.main()
