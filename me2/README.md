# Pi5-VCM — Voice Command Model

A single trained model (no LLM, no off-the-shelf ASR) that maps raw audio to
**intent + slots** for a Raspberry Pi 4/5 voice assistant.

## Architecture

```
mic -> VAD -> openWakeWord (pretrained) -> VCM (this repo) -> parser/actions
```

`VCM` (see `model.py`):

```
log-mel (T, 80)
  -> 2D conv stack (x4 temporal downsample)
  -> BiGRU (2 layers, hidden 128)
  -> intent head : 16-way classification (15 commands + OOV)
  -> slot head   : CTC over a constrained vocab (digits, clock words,
                   contacts, reminder keywords)
```

The constrained CTC vocabulary is what replaces an LLM: greedy decode gives a
transcript, and `parse_slots()` (rule-based, in `model.py`) turns
`(intent, transcript)` into slots.

## Files

| file            | purpose                                          |
|-----------------|--------------------------------------------------|
| `config.py`     | audio settings, intent taxonomy, CTC vocab       |
| `dataset.py`    | VCMDataset + synthetic/real manifest builders    |
| `dataloader.py` | collate (padding + CTC targets) + DataLoader     |
| `model.py`      | VCM model, CTC decode, rule-based slot parser    |
| `train.py`      | train / validate / test loops                    |
| `main.py`       | CLI entry point                                  |
| `utils/`        | audio I/O + features + toy synthesis, metrics    |

## Usage

```bash
cd me2
source ../pi5venv/bin/activate

python main.py generate                      # synthetic manifest -> data/
python main.py train --epochs 10             # train (synthetic by default)
python main.py train --manifest data/real_manifest.jsonl   # real data
python main.py test                          # evaluate best.pt on test split
python main.py demo                          # run sample utterances
```

## Remote training pipeline

`pipeline.sh` validates the Python sources, packages the trainer, deploys it to
`jdrne@daniel-pc.tailfa8657.ts.net`, runs training there, and fetches the
checkpoint into `artifacts/<version>/`. Remote runs are stored in
`~/TrainingGround/<version>/` on `daniel-pc`.

The default remote platform is Windows, matching `daniel-pc`. For a Unix SSH
host, set `REMOTE_PLATFORM=unix` and choose its interpreter with
`REMOTE_PYTHON`.

### Jenkins setup

Commit this repository and push it to a Git provider before creating a Jenkins
Pipeline job. Configure the job as **Pipeline script from SCM**, select Git,
and point it at the repository and branch containing `me2/Jenkinsfile`. Set
the Jenkins Script Path to `me2/Jenkinsfile`.

Create these Jenkins **Secret file** credentials with the exact IDs below:

| Credential ID | Contents |
|---|---|
| `pi5vcm-rig-ssh-key` | Private key used by `jdrne@daniel-pc.tailfa8657.ts.net` |
| `pi5vcm-pi-ssh-key` | Private key used by `jdrnepomuceno9@192.168.68.52` |
| `pi5vcm-known-hosts` | Verified host-key lines for both hosts |

The Jenkins agent must have Bash, Python, and the project environment available.
Set `TRAIN_REMOTE` to train on the rig and `DEPLOY_PI` to deploy the fetched
checkpoint. Pi deployment pauses for manual approval.

Before using SSH operations, create `~/.ssh/pi5vcm_known_hosts` containing the
verified host keys for both `daniel-pc.tailfa8657.ts.net` and the Pi. The
pipeline refuses unknown or changed host keys. Pi deployment defaults to the
normal SSH configuration for `jdrnepomuceno9@192.168.68.52`, matching
`.ssh/command.yaml`. Set `PI_SSH_KEY` only to override that identity.

Create the file only after comparing these fingerprints with the consoles of
both machines:

```bash
ssh-keyscan -t ed25519 daniel-pc.tailfa8657.ts.net
ssh-keyscan -t ed25519 192.168.68.52
```

Save the verified, complete output lines to
`~/.ssh/pi5vcm_known_hosts`, then protect it:

```bash
chmod 600 ~/.ssh/pi5vcm_known_hosts
```

Do not use `ssh-keyscan` output as trusted evidence by itself; it must be
checked against a fingerprint obtained through an independent channel.

```bash
./pipeline.sh compile
VERSION=v0.1.0 ./pipeline.sh package
VERSION=v0.1.0 EPOCHS=30 ./pipeline.sh train-remote
VERSION=v0.1.0 ./pipeline.sh fetch
```

For a real manifest, pass `MANIFEST` to package and training. The remote host
must already have a compatible Python environment; set `REMOTE_PYTHON` to its
interpreter path when it is not `python3`:

```bash
VERSION=v0.1.0 MANIFEST=data/real_manifest.jsonl \
  REMOTE_PYTHON=python EPOCHS=30 ./pipeline.sh all
```

The fetched `best.pt` is the deployment artifact. Keep the matching
`config.py`, `model.py`, and `utils/` files with it on the Raspberry Pi, or
install the dependencies listed in `requirements-runtime.txt` and run the
existing `main.py test`/`demo` commands there.

### Deploy the trained checkpoint to the Pi

The Pi deployment copies `best.pt`, `history.json`, the runtime source, and
`requirements-runtime.txt` to `~/TrainingGround/<version>/` on
`192.168.68.52`. It uses `jdrnepomuceno9@192.168.68.52` and the normal SSH
identity by default:

```bash
VERSION=v0.1.0 \
  ./pipeline.sh deploy-pi
```

To train on the rig and deploy the resulting best checkpoint in one command:

```bash
VERSION=v0.1.0 MANIFEST=data/real_manifest.jsonl EPOCHS=30 \
  ./pipeline.sh train-deploy-pi
```

On the Pi, install the runtime dependencies and evaluate the deployed model:

```bash
cd ~/TrainingGround/v0.1.0
python3 -m pip install -r requirements-runtime.txt
python3 main.py test --ckpt best.pt
python3 main.py demo --ckpt best.pt
```

Real-data manifest format (CSV or JSONL), one row per utterance:

```
path,intent,transcript,speaker
data/wav/0001.wav,dim_lights,dim the lights to 40 percent,alice
```

## Notes

- **Synthetic data is toy speech** (formant tones from `utils/audio_utils.py`),
  not real audio. It exists so the pipeline can be developed and smoke-tested
  before recordings exist. Intent accuracy on it is meaningless; WER on it is
  a pipeline check, not a quality measure.
- CPU-only on the Pi; `--device auto` picks cuda/mps when available.
- Keep `--num-workers 0` on the Pi (2 GB RAM).
- Checkpoints land in `checkpoints/` (`best.pt`, `last.pt`, `history.json`).
