# Browser prompt generator

This is the app behind [ras.ms/prompts](https://ras.ms/prompts): a writing
prompt generator running the bundled `prompter-v0.4-Nano` model (0.8M
parameters) entirely in the browser. It's included as an example of a custom
front end built on a NanoGym browser export.

For a plain test page for one of your own runs, use the Playground's
**export for browser** button or `scripts/export.py` instead; see the main
README.

## What it does

Each click on **Generate** starts a fresh entry with `<bos>` and writes until
the model produces `<eos>`, then shows the prompt. **Let's write** opens a
full-screen editor, and **Save** downloads a styled HTML document with the
prompt and your writing. The light/dark preference is remembered in the browser.

## Run it

The model files (`weights.bin`, `model.json`, `vocab.json`) are included, so it
works right after cloning. From this folder, start any static web server and
open the printed address:

```bash
python -m http.server 8000
```

A server is only needed because browsers block `fetch()` for `file://` pages;
nothing runs server-side.

## Swap in a different model

To load another run into this app, overwriting the bundled model:

```bash
python export_model.py --run <run-name>
```

This reads `runs/<run-name>/ckpt_best.pt` and `vocab.json` and writes the three
model files into this folder. `model.json` records which run they came from in
its `source_run` field. To point at files outside the usual `runs/<name>/`
layout, pass `--checkpoint` and `--vocab` instead of `--run`.

The page expects a model trained on entries wrapped in `<bos>` … `<eos>`, like
the bundled writing-prompt dataset.
