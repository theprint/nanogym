# Data

Put UTF-8 plain-text files in this folder to train on them. Every `.txt` file
here shows up in the dashboard's **New experiment** dataset list.

## Bundled datasets

| File | Size | Entries | Trained into |
|---|---|---|---|
| `writing_prompts.txt` | 62 MB | 640,000 one-line writing prompts | `runs/prompter-v0.4-Nano` |
| `occultisms_29k.txt` | 4.7 MB | 29,000 short reflective passages | `runs/occultisms-v2.4` |

The shipped configs in `configs/` point at `occultisms_29k.txt`, so you can
train straight after cloning.

## Formatting your own data

Any text works; the model learns whatever characters and patterns it sees. If
your data is a collection of separate entries (prompts, quotes, poems…), wrap
each one in `<bos>` and `<eos>`:

```text
<bos>First entry.<eos>
<bos>Second entry.<eos>
```

The model then learns where entries start and stop, and the Playground and
browser exports can generate one complete entry at a time.

Each run freezes its character vocabulary in its own `runs/<name>/vocab.json`,
so changing a dataset later never breaks an existing model. To check a file
before training, run `python scripts/prepare_data.py data/<file>.txt`.
