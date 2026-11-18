"""Smoke tests: every script imports and parses its arguments, and the saved
predictions reproduce the published headline scores.

Run from the repository root:  python -m unittest discover -s code/tests
"""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path

CODE=Path(__file__).resolve().parents[1]
SCRIPTS=['apply_edit.py','make_delta.py','build_edit.py','summarise_results.py',
         'eval/eval.py','eval/fetch_images.py','eval/scorer.py','eval/no_list_eval.py','eval/no_list_scorer.py',
         'island/fetch_crops.py','island/predict.py','island/score.py',
         'recipe/fetch_practice049.py']

class ScriptsParseArguments(unittest.TestCase):
    def test_help(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                run=subprocess.run([sys.executable,str(CODE/script),'--help'],cwd=CODE/Path(script).parent,capture_output=True,text=True,timeout=300)
                self.assertEqual(run.returncode,0,run.stderr[-2000:])
                self.assertIn('usage',run.stdout.lower())
    def test_prepare_base_module(self):
        run=subprocess.run([sys.executable,'-m','recipe.prepare_base048','--help'],cwd=CODE,capture_output=True,text=True,timeout=300)
        self.assertEqual(run.returncode,0,run.stderr[-2000:])

class SavedPredictionsReproduceScores(unittest.TestCase):
    def test_headline(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'summary.json'
            run=subprocess.run([sys.executable,str(CODE/'summarise_results.py'),'--output',str(out)],cwd=CODE,capture_output=True,text=True,timeout=600)
            self.assertEqual(run.returncode,0,run.stderr[-2000:])
            s=json.loads(out.read_text())
        cam={k:v['balanced_accuracy'] for k,v in s['camera_benchmark'].items()}
        self.assertEqual(cam,{'unedited':64.09,'camera-trap-edit':81.76,'galapagos-island-edit':72.14,'camera-island-direct-edit':80.7,'camera-island-lora':79.22,'matched-siglip':82.79})
        isl=s['island_test']
        self.assertEqual(isl['galapagos-island-edit']['mockingbird_vs_brown_noddy']['balanced_accuracy'],98.33)
        self.assertEqual(isl['galapagos-island-edit']['land_iguana_vs_green_iguana']['balanced_accuracy'],81.22)
        self.assertEqual(isl['galapagos-island-edit']['marine_iguana_vs_green_iguana']['balanced_accuracy'],80.37)
        self.assertEqual(isl['galapagos-island-edit']['invasive_rat_recall']['equal_location_recall'],81.23)
        self.assertEqual([isl['camera-island-lora'][k].get('balanced_accuracy',isl['camera-island-lora'][k].get('equal_location_recall')) for k in ('land_iguana_vs_green_iguana','marine_iguana_vs_green_iguana','mockingbird_vs_brown_noddy','invasive_rat_recall')],[87.58,91.69,93.33,88.2])
        self.assertEqual(s['camera_paired_changes']['camera-island-lora minus camera-island-direct-edit']['interval_95_pp'],[-5.8,2.54])

if __name__=='__main__':unittest.main()
