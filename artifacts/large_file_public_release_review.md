# Large File Public Release Review

path | size | tracked_status | decision | action | reason
--- | --- | --- | --- | --- | ---
`artifacts/checkpoints/test/stage1/best_model.pth` | >50M | local path present | archive_not_public | keep ignored, do not stage | checkpoint artifact, not public evidence
`artifacts/checkpoints/test/stage1/final_model.pth` | >50M | local path present | archive_not_public | keep ignored, do not stage | checkpoint artifact, not public evidence
`archive/model_assets/vgg16-397923af.pth` | >50M | local path present | archive_not_public | keep ignored, do not stage | model weight file should not be shipped in public repo
`data/age_groups/20-29/train_data.csv` | >50M | local path present | human_confirm | decide whether public data release is intended | raw dataset-sized file, not a small example artifact
`data/vam_data/processed_cache/train_images.npy` | >50M | local path present | archive_not_public | ignore path, do not stage | processed cache, not public evidence
`data/vam_data/processed_cache/test_images.npy` | >50M | local path present | archive_not_public | ignore path, do not stage | processed cache, not public evidence
