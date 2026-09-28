# Exported NanoGym model

This folder is a self-contained, static web page that runs a character-level
GPT model trained with [NanoGym](https://github.com/theprint/nanogym)
entirely in the browser. No Python or server-side code runs at inference time.

| File | What it is |
|---|---|
| `index.html` | A minimal test page: click **Generate** for one complete entry |
| `model.js` | The inference engine (plain JavaScript, no dependencies) |
| `model.json` | Architecture and the layout of every tensor in `weights.bin` |
| `weights.bin` | The model weights as float32 |
| `vocab.json` | The model's character vocabulary |
| `training.json` | How the model was trained: dataset file, settings, checkpoint stats |

## Try it locally

Browsers refuse to load the model files from a `file://` page, so serve the
folder with any static web server and open the printed address:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000>.

## Put it online

Upload the whole folder to any static host (GitHub Pages, Netlify, your own
web server). There is nothing to build.

## How generation works

Each click starts a fresh entry the same way the training data marks one: with
`<bos>`. The model writes until it produces `<eos>`, the output limit, or the
end of its context window. The special tokens are hidden from the output.
`TEMPERATURE` and `MAX_NEW_CHARS` sit at the top of the script in `index.html`
if you want to change them.

The page is intentionally unstyled. Treat it as a starting point for your own
interface: `model.js` exports `BrowserGPT` and `CharTokenizer`, and
`index.html` shows the few lines needed to load and sample from them.
