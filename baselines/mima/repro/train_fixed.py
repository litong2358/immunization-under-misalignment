import os
import runpy
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
MIMA = os.path.dirname(HERE)
sys.path.insert(0, MIMA)

argv = sys.argv[1:]
mode = "name"
if "--merge_category" in argv:
    i = argv.index("--merge_category")
    mode = argv[i + 1]
    del argv[i:i + 2]
assert mode in ("name", "prompt"), mode

mp = types.ModuleType("memory_profiler")
mp.profile = lambda f: f
sys.modules["memory_profiler"] = mp

import data_pipeline

PREFIX = "An artwork of "
_getitem = data_pipeline.MIMADataset_Compose.__getitem__


def category(prompt, concept_name):
    if mode == "prompt":
        return prompt
    if prompt.startswith(PREFIX):
        return prompt[len(PREFIX):]
    return concept_name.replace("_", " ")


def __getitem__(self, index):
    examples = _getitem(self, index)
    for ex, (concept_name, _) in zip(examples, self.concept_class_pairs):
        prompt = self.instance_images_path[concept_name][0][1]
        ex["class_name"] = category(prompt, concept_name)
    return examples


data_pipeline.MIMADataset_Compose.__getitem__ = __getitem__
print(f"[train_fixed] per-concept batch keys, merge_category={mode}", flush=True)

sys.argv = [os.path.join(MIMA, "train.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")
