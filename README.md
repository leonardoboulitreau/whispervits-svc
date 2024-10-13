# Setup Environment
(Tested on a Quadro RTX 5000 with NVIDIA-SMI Driver Version: 535.104.05, CUDA Version: 12.2 on a UBUNTU 22.04)

1. Install [PyTorch](https://pytorch.org/get-started/locally/).

2. Install project dependencies
    ```shell
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    ```
    **Note: whisper is already built-in, do not install it again otherwise it will cuase conflict and error**
3. Download the Timbre Encoder: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), put `best_model.pth.tar`  into `speaker_pretrain/`.

4. Download whisper model [whisper-large-v2](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt). Make sure to download `large-v2.pt`，put it into `whisper_pretrain/`.

5. Download [hubert_soft model](https://github.com/bshall/hubert/releases/tag/v0.1)，put `hubert-soft-0d54a1f4.pt` into `hubert_pretrain/`.

6. Download pitch extractor [crepe full](https://github.com/maxrmorrison/torchcrepe/tree/master/torchcrepe/assets)，put `full.pth` into `crepe/assets`.

   **Note: crepe full.pth is 84.9 MB, not 6kb**
   
7. Download trained model [lesd5_100.pretrain.pth](https://drive.google.com/file/d/1hvA3GEsufVUnX5gmGof_cSqbmQmgfV4i/view?usp=sharing), and put it into `vits_pretrain/`.
