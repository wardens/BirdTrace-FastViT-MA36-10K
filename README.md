# BirdTrace FastViT-MA36

Bird species classification model from [BirdTrace](https://birdtrace.org), with **10,480 classes** and a **prepared 384×384 RGB image** input. This public release contains inference weights and minimal inference metadata.

## Downloads and release source

This repository publishes inference weights and necessary metadata; training code is not included. The two model binaries are distributed as GitHub Release assets, outside ordinary Git history.

Source of truth: [Hugging Face model repository](https://huggingface.co/warden123/BirdTrace-FastViT-MA36-10K), pinned to [commit `ceadaa8749f08c2938117271e5535f6bed7fdc90`](https://huggingface.co/warden123/BirdTrace-FastViT-MA36-10K/tree/ceadaa8749f08c2938117271e5535f6bed7fdc90).

Download the [complete inference ZIP](https://github.com/wardens/BirdTrace-FastViT-MA36-10K/releases/download/e15-ceadaa8749f0/BirdTrace-FastViT-MA36-10K-E15-inference.zip), or download [model.onnx](https://github.com/wardens/BirdTrace-FastViT-MA36-10K/releases/download/e15-ceadaa8749f0/model.onnx) and [model.safetensors](https://github.com/wardens/BirdTrace-FastViT-MA36-10K/releases/download/e15-ceadaa8749f0/model.safetensors) into a directory containing the metadata files below. The ZIP includes the original HF model card and its original checksum manifest. To use the inference examples below, work from the extracted ZIP's top-level directory and supply the upstream architecture and prepared image as described.

See the [release notes](https://github.com/wardens/BirdTrace-FastViT-MA36-10K/releases/tag/e15-ceadaa8749f0) for provenance and asset checksums. `SOURCE_SNAPSHOT.json` records the pinned HF file list. `HF_SHA256SUMS` is the unmodified original HF checksum manifest and applies to the original HF files, including the HF README inside the ZIP.

## Model and formats

- Architecture: [FastViT-MA36](https://github.com/apple/ml-fastvit), 55,605,164 parameters in the unfused PyTorch model; classification head: 1,216 → 10,480.
- `model.safetensors`: unfused PyTorch state dictionary, with original `backbone.` key prefixes, float32 weights and int64 tracking buffers. Remove only that prefix when loading the public architecture. This file contains no executable model implementation.
- `model.onnx`: standalone FP32 inference graph, ONNX opset 18; input `image`, float32 `[1, 3, 384, 384]`; output `visual_logits`, float32 `[1, 10480]`.
- `classes.json`: ordered records with `index`, original `species_id` string, and unchanged `species_name`. Output index `i` corresponds to record `i`. Names are common-name labels as supplied, including underscores; separate scientific names and external taxonomy identifiers are not supplied.
- `config.json` and `preprocessor_config.json`: explicit inference contracts. These are descriptive metadata, not Transformers AutoModel or AutoImageProcessor configurations.

## Training data

BirdTrace reports that the model was trained using approximately 45 million bird images across 10,480 bird classes.

The model uses the FastViT-MA36 architecture. It was not initialized from Apple's pretrained model weights, and subsequent training did not use Apple pretrained weights.

## Internal validation

The following results are BirdTrace's internal validation results for the final E15 checkpoint, covering **10,130 validated classes** out of **10,480 classification classes**. They are not independent third-party benchmarks.

| Metric | Accuracy |
| --- | --- |
| Micro Top-1 | 90.92% |
| Micro Top-3 | 96.53% |
| Macro Top-1 | 86.74% |
| Macro Top-3 | 93.18% |

## Input and output

Supply a **prepared 384×384 RGB image**. Source-photo resizing and cropping are application-dependent and are **not specified by this model release**. Photo interpolation and orientation handling are also application-dependent. No resize or crop transform is supplied here.

Convert image pixels to float32, divide by 255, and arrange channels as NCHW. For SafeTensors/PyTorch, normalize externally with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. For ONNX, supply RGB values in `[0, 1]` directly: mean/std normalization is already embedded, so do not apply it twice.

Both formats return raw logits, with no softmax, Top-K selection, or geographic adjustment inside the model. Rank logits for predictions. Optional softmax scores are relative to the supplied class vocabulary and are not calibrated confidence estimates.

## PyTorch usage

Dependencies: PyTorch, torchvision, timm, SafeTensors, NumPy, and Pillow. Verified with Python 3.11, PyTorch 2.12.0, torchvision 0.27.0, timm 1.0.27, and SafeTensors 0.8.0.

Use the public [Apple implementation at revision `8af5928238cab99c45f64fc3e4e7b1516b8224ba`](https://github.com/apple/ml-fastvit/tree/8af5928238cab99c45f64fc3e4e7b1516b8224ba). The built-in timm implementation may use different parameter names. Run this example from the downloaded release directory, with the upstream repository checked out beside it as `../ml-fastvit` and a prepared image named `prepared.png`:

```python
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, str(Path('../ml-fastvit').resolve()))
import models  # registers the public Apple FastViT factories
from timm.models import create_model

config = json.loads(Path('config.json').read_text())
classes = json.loads(Path('classes.json').read_text())
prep = json.loads(Path('preprocessor_config.json').read_text())

model = create_model('fastvit_ma36', pretrained=False,
                     num_classes=config['num_labels']).eval()
weights = load_file('model.safetensors', device='cpu')
prefix = config['state_dict_prefix']
assert all(key.startswith(prefix) for key in weights)
model.load_state_dict(
    {key[len(prefix):]: value for key, value in weights.items()}, strict=True
)

with Image.open('prepared.png') as source:
    image = source.convert('RGB')
    if image.size != (384, 384):
        raise ValueError('Provide an already prepared 384x384 RGB image')
    pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
rgb01 = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).contiguous()
mean = torch.tensor(prep['image_mean']).view(1, 3, 1, 1)
std = torch.tensor(prep['image_std']).view(1, 3, 1, 1)
with torch.inference_mode():
    logits = model((rgb01 - mean) / std)
assert logits.shape == (1, len(classes))
for index in logits.topk(3, dim=1).indices[0].tolist():
    print(classes[index]['species_id'], classes[index]['species_name'])
```

## ONNX usage

With ONNX Runtime installed (verified with 1.29.0), reuse `rgb01` and `classes` from the example above:

```python
import onnxruntime as ort

session = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])
logits = session.run(['visual_logits'], {'image': rgb01.numpy()})[0]
assert logits.shape == (1, len(classes))
for index in np.argsort(-logits[0], kind='stable')[:3]:
    print(classes[int(index)]['species_id'], classes[int(index)]['species_name'])
```

## Applications and limitations

Intended applications include noncommercial research, education, and bird-recognition experiments. Visually similar species, immature plumage, poor lighting, distant subjects, and unfamiliar image conditions can affect results. Classification always ranks the fixed vocabulary; it does not detect whether an image contains a bird or provide an unknown-species class. Human review is appropriate for consequential identifications. The reported internal validation results do not establish performance on independent datasets or in every application.

## Third-Party Adoption

[SuperPicky](https://superpicky.app/) ([GitHub](https://github.com/jamesphotography/SuperPicky)) is an independent third-party project, not an official BirdTrace partner.

Its official [v4.6.4 RC7 release notes](https://github.com/jamesphotography/SuperPicky/releases/tag/v4.6.4-rc7) report adoption of BirdTrace FastViT-MA36. The [v4.6.4 RC8 release notes](https://github.com/jamesphotography/SuperPicky/releases/tag/v4.6.4-rc8) report the transition to BirdTrace as SuperPicky's sole bird species identification model. In RC8, the developer reported **97.1% accuracy on the Victoria test batch**, compared with **96.0% previously**, after improving cropping in SuperPicky's batch-processing pipeline.

These are third-party, developer-reported, sample-specific results, not standardized independent benchmarks or a measure of general model accuracy. They reflect SuperPicky's application-specific processing and do not define this model release's source-photo resizing or cropping requirements.

## License

BirdTrace FastViT-MA36 model weights are released under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license, to the extent of BirdTrace's rights.

The model is provided for non-commercial scientific research, education, biodiversity conservation, and related non-commercial applications.

Commercial use requires separate authorization from the relevant rights holders.

Applicable third-party licenses and rights remain in effect.

For more information about the BirdTrace project, visit [https://birdtrace.org](https://birdtrace.org).

## Attribution

FastViT is credited to Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, and Anurag Ranjan (ICCV 2023); see [upstream software terms](https://github.com/apple/ml-fastvit/blob/main/LICENSE) and `THIRD_PARTY_NOTICES.txt`.

## Integrity

`SHA256SUMS` verifies the small files in this GitHub repository. Run `shasum -a 256 -c SHA256SUMS` from the repository root. `RELEASE_SHA256SUMS`, also available with the release assets, verifies both model files and the complete inference ZIP. After downloading those assets together, run `shasum -a 256 -c RELEASE_SHA256SUMS`. Inside the extracted inference ZIP, the original HF `SHA256SUMS` verifies the original HF release files.
