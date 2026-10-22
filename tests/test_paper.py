"""Independent frozen model fixtures and reconstruction constraints."""
from pathlib import Path
import json
import unittest
import cv2
import numpy as np
from rootscope.batch import finite
from rootscope.paper.runtime import forest_predict, PaperEngine, reconstruct, artifact
from rootscope.paper.decoder import select
from rootscope.paper.batch import resolve_fold

ROOT=Path(__file__).resolve().parents[1]

class PaperTests(unittest.TestCase):
    def test_frozen_sklearn_predictions_match_numeric_runtime(self):
        fixtures=json.loads((ROOT/'tests/fixtures/paper_forest_golden.json').read_text(encoding='utf-8'))
        self.assertEqual(len(fixtures),15)
        for item in fixtures:
            with self.subTest(fold=item['fold'],kind=item['kind']):
                actual=forest_predict(item['fold'],item['kind'],item['inputs'])
                np.testing.assert_allclose(actual,item['expected'],rtol=0,atol=1e-12)

    def test_primary_decoder_preserves_axis_and_excludes_shared_edges(self):
        pool=dict(edge_lengths={'0':10.,'1':20.,'2':30.},candidates=[
            dict(edge_ids=[0]),dict(edge_ids=[1,2]),dict(edge_ids=[1]),dict(edge_ids=[2])])
        chosen,meta=select(pool,np.array([[.9,0,0],[1,0,0],[.1,1,1],[.1,1,1]]),2.,True)
        self.assertEqual(chosen,[0,1])
        self.assertTrue(meta['graph_edge_exclusive'])
        self.assertFalse(meta['reference_count_passed_to_decoder'])

    def test_empty_prediction_has_no_fabricated_roots(self):
        result=reconstruct(np.zeros((32,32,3),np.uint8),np.zeros((32,32),np.float32),.65,0)
        self.assertEqual(result['paths'],[])
        self.assertEqual(result['selected_count'],0)
        self.assertIsNone(result['geometry']['axis_curve_length_px'])

    def test_five_d3_models_are_three_channel_and_finite(self):
        gray=np.full((256,256),180,np.uint8)
        photo=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        for fold in range(5):
            with self.subTest(fold=fold):
                engine=PaperEngine(fold)
                self.assertEqual(engine.session.get_inputs()[0].shape[1],3)
                probability=engine.predict_photo(gray,photo)
                self.assertEqual(probability.shape,gray.shape)
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(((probability>=0)&(probability<=1)).all())

    def test_export_preserves_boolean_types(self):
        self.assertIs(finite({'biological_identity_validated':False})['biological_identity_validated'],False)

    def test_policy_checksums_and_unknown_image_assignment(self):
        for fold in range(5):
            self.assertIsInstance(artifact(fold,'axis_policy'),dict)
        self.assertEqual(resolve_fold(ROOT/'examples/synthetic_root.png','auto'),
                         (0,'new_image_uses_fold_0_not_external_validation'))

if __name__=='__main__':unittest.main()
