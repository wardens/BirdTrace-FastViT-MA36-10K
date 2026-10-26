# BirdTrace FastViT-MA36 E15

Inference weights and necessary metadata for BirdTrace's 10,480-class bird species classifier. Training code is not included.

## Pinned source

- Hugging Face repository: https://huggingface.co/warden123/BirdTrace-FastViT-MA36-10K
- Source commit: `ceadaa8749f08c2938117271e5535f6bed7fdc90`
- Exact snapshot: https://huggingface.co/warden123/BirdTrace-FastViT-MA36-10K/tree/ceadaa8749f08c2938117271e5535f6bed7fdc90
- GitHub release tag: `e15-ceadaa8749f0`

Both model files are the exact bytes downloaded from this pinned snapshot. No retraining, re-exporting, quantization, tensor renaming, or class reordering was performed. The model binaries are release assets and are not stored in ordinary Git history.

## Downloads

- [model.onnx](https://github.com/wardens/BirdTrace-FastViT-MA36-10K/releases/download/e15-ceadaa8749f0/model.onnx)
- [model.safetensors](https://github.com/wardens/BirdTrace-FastViT-MA36-10K/releases/download/e15-ceadaa8749f0/model.safetensors)
- [Complete inference ZIP](https://github.com/wardens/BirdTrace-FastViT-MA36-10K/releases/download/e15-ceadaa8749f0/BirdTrace-FastViT-MA36-10K-E15-inference.zip)
- [RELEASE_SHA256SUMS](https://github.com/wardens/BirdTrace-FastViT-MA36-10K/releases/download/e15-ceadaa8749f0/RELEASE_SHA256SUMS)

The ZIP contains the original HF `.gitattributes`, `README.md`, `SHA256SUMS`, `THIRD_PARTY_NOTICES.txt`, `classes.json`, `config.json`, `preprocessor_config.json`, and both models. It also contains `SOURCE_SNAPSHOT.json` and an additional `HF_SNAPSHOT_SHA256SUMS` covering all original files plus the source record. The original HF README and checksum manifest are unmodified.

## Verification

The original HF checksum manifest and pinned API file identities were verified against every downloaded file before packaging. Every ZIP entry was re-read and verified. Download all assets together and run:

```sh
shasum -a 256 -c RELEASE_SHA256SUMS
```

Inside the extracted inference ZIP, run:

```sh
shasum -a 256 -c SHA256SUMS
shasum -a 256 -c HF_SNAPSHOT_SHA256SUMS
```

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| BirdTrace-FastViT-MA36-10K-E15-inference.zip | 446768109 | `0ef3a2394b7024b1324d6dee6c810b03dd6fc2afccc88a56f4fd1bfbf6fac1a3` |
| model.onnx | 222385590 | `0f2c02591ac1e85146b8964d33a174f29ecfb52334bf42e3e3bcf91743a46e31` |
| model.safetensors | 222963024 | `81d698a8457a9c40d67e6d20d18ebe24f9d164b7083717b9f18a08a99f9f0ca6` |

## Scope and license

Refer to the model card for the unchanged E15 internal validation metrics, their limitations, prepared 384x384 RGB input contract, external normalization for PyTorch, embedded normalization for ONNX, inference examples, and third-party adoption. Source-photo resizing and cropping remain application-dependent and unspecified by the model release.

The CC BY-NC 4.0 terms in the source model card apply to the extent of BirdTrace's rights. Applicable third-party licenses and rights remain in effect; the original third-party notices are included.
