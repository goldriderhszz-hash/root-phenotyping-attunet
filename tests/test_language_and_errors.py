"""Behavioral language parity and failure-path checks for the shared pipeline."""
from pathlib import Path
import json
import string
import tempfile
import threading
import unittest
import cv2
import numpy as np
from rootscope.batch import run_batch,row_for
from rootscope.engine import InferenceEngine,read_gray
from rootscope.i18n import MESSAGES,set_language,get_language

class LanguageAndFailureTests(unittest.TestCase):
    def tearDown(self):set_language('en')

    def test_translation_placeholders_match(self):
        for key,values in MESSAGES.items():
            self.assertEqual(set(values),{'en','zh'})
            placeholders=lambda text:{field for _,field,_,_ in string.Formatter().parse(text) if field}
            self.assertEqual(placeholders(values['en']),placeholders(values['zh']),key)

    def test_language_does_not_change_masks_or_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);source=base/'中文图像.png'
            image=np.full((256,256),210,np.uint8);cv2.line(image,(125,15),(119,239),35,3)
            cv2.imencode('.png',image)[1].tofile(source)
            output=[]
            for language in ['en','zh']:
                set_language(language);r=run_batch([source],[],base/'中文结果',make_zip=False)
                self.assertFalse(r['errors']);item=r['results'][0]
                mask=read_gray(Path(r['folder'])/'images'/item['image_id']/'prediction_mask.png')
                output.append((row_for(item),mask))
            self.assertEqual(output[0][0],output[1][0]);np.testing.assert_array_equal(output[0][1],output[1][1])

    def test_language_context_is_isolated_between_threads(self):
        set_language('zh');values=[]
        thread=threading.Thread(target=lambda:values.append(get_language()));thread.start();thread.join()
        self.assertEqual(values,['en']);self.assertEqual(get_language(),'zh')

    def test_empty_inputs_threshold_missing_hash_and_write_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            with self.assertRaises(ValueError):run_batch([],[],base)
            source=base/'source.png';cv2.imwrite(str(source),np.ones((256,256),np.uint8))
            with self.assertRaises(ValueError):run_batch([source],[],base,threshold=.99)
            with self.assertRaises(ValueError):run_batch([base/'missing.png'],[],base)
            model=base/'bad.onnx';model.write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError,'integrity'):InferenceEngine(model)
            file=base/'file';file.write_text('file')
            with self.assertRaises(OSError):run_batch([source],[],file)

    def test_corrupt_image_and_reference_mismatch_are_logged(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);source=base/'source.png';reference=base/'source_mask.png';bad=base/'bad.png'
            cv2.imwrite(str(source),np.ones((256,256),np.uint8));cv2.imwrite(str(reference),np.ones((64,64),np.uint8));bad.write_bytes(b'corrupt')
            r=run_batch([source,bad],[reference],base/'output',make_zip=False)
            self.assertEqual(len(r['errors']),2);self.assertEqual(len(r['results']),0)
            self.assertIn('dimensions differ',r['errors'][0]['reason']);self.assertIn('Cannot read',r['errors'][1]['reason'])

    def test_pre_cancelled_batch_writes_a_cancelled_report(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);source=base/'source.png';cv2.imwrite(str(source),np.ones((256,256),np.uint8))
            cancellation=threading.Event();cancellation.set()
            r=run_batch([source],[],base/'output',cancel_event=cancellation,make_zip=False)
            self.assertEqual(r['report']['status'],'cancelled');self.assertEqual(r['results'],[])

if __name__=='__main__':unittest.main()
