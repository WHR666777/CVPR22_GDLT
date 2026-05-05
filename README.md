# Likert Scoring with Grade Decoupling for Long-term Action Assessment

This is the code for CVPR2022 paper "Likert Scoring with Grade Decoupling for Long-term Action Assessment".

## Environments

- RTX2080Ti
- CUDA: 10.2
- Python: 3.9.7
- PyTorch: 1.10.1+cu102

## Features

The features and label files of Rhythmic Gymnastics dataset can be download [here](https://1drv.ms/u/s!AqXkt0Mw7p9llVaV2oV1mwmdAICG).

\[23-04-10 Update\] The features and label files of Fis-V dataset can be download [here](https://1drv.ms/u/s!AqXkt0Mw7p9llWEihc533CB87U5P?e=EadhCo).

## Running

Please fill in or select the args enclosed by {} first.

- Windows + Conda environment

```
conda env create -f environment.gdlt-win.yml
conda activate gdlt-win
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

- Fis-V (TES)

```
python main.py --dataset fisv --video-path {path of Fis-V video features} --train-label-path {path of Fis-V train labels} --test-label-path {path of Fis-V test labels} --score-key TES --score-max {max TES score} --model-name fisv_tes --lr 1e-2 --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.7 --device cuda
```

- Fis-V (PCS)

```
python main.py --dataset fisv --video-path {path of Fis-V video features} --train-label-path {path of Fis-V train labels} --test-label-path {path of Fis-V test labels} --score-key PCS --score-max {max PCS score} --model-name fisv_pcs --epoch 400 --lr 1e-2 --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.7 --device cuda
```

- Fis-V evaluation

```
python main.py --dataset fisv --video-path {path of Fis-V video features} --train-label-path {path of Fis-V train labels} --test-label-path {path of Fis-V test labels} --score-key {TES/PCS} --score-max {max score} --clip-num 124 --n_decoder 2 --n_query 4 --dropout 0.7 --device cuda --test --ckpt {checkpoint name}
```

- Joint Fis-V training (`TES + PCS`, dual-prototype AV-Likert)

```
python joint_fisv_main.py --visual-feat-dir {path of SwinTx visual features} --mm-feat-dir {path of PAMFN multimodal feature dir} --joint-train-label fis-v/train.txt --joint-test-label fis-v/test.txt --pamfn-ckpt-dir {path of PAMFN pretrained branch dir} --pamfn-dataset-name PCS --pamfn-ckpt {path of trained PAMFN PCS multimodal checkpoint} --ckpt-path ./ckpt/joint_fisv_best.pkl --lambda-rank 0.3 --lambda-gap 0.2 --warmup-epochs 40 --device cuda
```

- Joint Fis-V evaluation

```
python joint_fisv_main.py --visual-feat-dir {path of SwinTx visual features} --mm-feat-dir {path of PAMFN multimodal feature dir} --joint-train-label fis-v/train.txt --joint-test-label fis-v/test.txt --pamfn-ckpt-dir {path of PAMFN pretrained branch dir} --pamfn-dataset-name PCS --test --ckpt ./ckpt/joint_fisv_best.pkl --device cuda
```

- Windows shortcut with repo-local defaults

```
powershell -ExecutionPolicy Bypass -File .\run_joint_fisv.ps1
powershell -ExecutionPolicy Bypass -File .\run_joint_fisv.ps1 -Test
```

The PowerShell launcher assumes this workspace layout:

- visual features: `CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32`
- multimodal features: `PAMFN_Reproduce/data/features/FISV_{rgb,flow,audio}_*.npy`
- PAMFN single-modality checkpoints: `PAMFN_Reproduce/pretrained_models/feats1/PCS_{rgb,flow,audio}_*.pth`
- optional PAMFN multimodal checkpoint: `PAMFN_Reproduce/pretrained_models/feats1/PCS_multimodal.pth`

If `PCS_multimodal.pth` is absent, the launcher still runs and initializes from the three single-modality branch checkpoints only.

Key ablations are exposed in the new entrypoint:

- `--no-shared-prototype`: disable the shared base prototype bank
- `--lambda-rank 0`: remove batch-wise ranking supervision
- `--lambda-gap 0`: remove gap-aware consistency
- `--pcs-memory pooled`: replace temporal sync tokens with pooled PAMFN memory

For paper-consistent defaults, `main.py` now auto-resolves these values when they are not passed explicitly:

- `Fis-V TES`: `clip_num=124`, `epoch=320`, `dropout=0.7`, `alpha=0.5`
- `Fis-V PCS`: `clip_num=124`, `epoch=400`, `dropout=0.7`, `alpha=0.5`
- `RG`: `clip_num=68`, `dropout=0.3`, `alpha=1.0`, and `epoch={250/400/500/150}` for `{Ball/Clubs/Hoop/Ribbon}`

- Training

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --model-name {the name used to save model and log} --action-type {Ball/Clubs/Hoop/Ribbon} --lr 1e-2 --epoch {250/400/500/150} --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3
```

- Testing

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --action-type {Ball/Clubs/Hoop/Ribbon} --n_decoder 2 --n_query 4 --dropout 0.3 --test --ckpt {the name of the used checkpoint}
```

