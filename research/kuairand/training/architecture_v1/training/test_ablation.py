import unittest

from run_ablation import load_plan, materialize, selected


class AblationPlanTests(unittest.TestCase):
    def test_plan_materializes_one_factor_variants_and_unique_tracking(self):
        base, groups = load_plan()
        variants = selected(groups, 'sampling')
        self.assertEqual([name for _, name, _ in variants], ['sample_s0_none', 'sample_s1_logq_half', 'sample_s3_logq_laplace'])
        _, variant, override = variants[1]
        config = materialize(base, 'sampling', variant, override, 2026)
        self.assertEqual(config['training']['sampling_logq_alpha'], .5)
        self.assertEqual(config['wandb']['name'], '1k-ablation-sample_s1_logq_half-seed2026')
        self.assertEqual(config['ablation']['group'], 'sampling')
        self.assertEqual(base['experiment']['seed'], 42)


if __name__ == '__main__':
    unittest.main()
