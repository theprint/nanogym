const SPECIAL_TOKENS = new Set(["<pad>", "<unk>", "<bos>", "<eos>"]);

export class CharTokenizer {
  constructor(payload) {
    this.tokens = payload.tokens;
    this.stoi = new Map(this.tokens.map((token, index) => [token, index]));
    this.bosId = this.stoi.get("<bos>");
    this.eosId = this.stoi.get("<eos>");
  }

  encode(text) {
    return [...text].map((character) => this.stoi.get(character) ?? this.stoi.get("<unk>"));
  }

  decode(ids) {
    return ids
      .map((id) => this.tokens[id] ?? "<unk>")
      .filter((token) => !SPECIAL_TOKENS.has(token))
      .join("");
  }
}

function erf(value) {
  // Abramowitz and Stegun approximation; close enough to torch's exact GELU
  // for this inference path while remaining available in every browser.
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * x);
  const polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return sign * (1 - polynomial * Math.exp(-x * x));
}

function gelu(value) {
  return 0.5 * value * (1 + erf(value / Math.sqrt(2)));
}

function layerNorm(input, weight, bias) {
  let mean = 0;
  for (let i = 0; i < input.length; i += 1) mean += input[i];
  mean /= input.length;
  let variance = 0;
  for (let i = 0; i < input.length; i += 1) {
    const centered = input[i] - mean;
    variance += centered * centered;
  }
  const inverseStd = 1 / Math.sqrt(variance / input.length + 1e-5);
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i += 1) output[i] = (input[i] - mean) * inverseStd * weight[i] + bias[i];
  return output;
}

function linear(input, weight, bias, outputSize, inputSize) {
  const output = new Float32Array(outputSize);
  for (let row = 0; row < outputSize; row += 1) {
    let value = bias ? bias[row] : 0;
    const offset = row * inputSize;
    for (let column = 0; column < inputSize; column += 1) value += weight[offset + column] * input[column];
    output[row] = value;
  }
  return output;
}

function softmaxSample(logits, temperature) {
  const scaled = new Float32Array(logits.length);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    scaled[i] = logits[i] / temperature;
    if (scaled[i] > max) max = scaled[i];
  }
  let total = 0;
  for (let i = 0; i < scaled.length; i += 1) {
    scaled[i] = Math.exp(scaled[i] - max);
    total += scaled[i];
  }
  let threshold = Math.random() * total;
  for (let i = 0; i < scaled.length; i += 1) {
    threshold -= scaled[i];
    if (threshold <= 0) return i;
  }
  return scaled.length - 1;
}

export class BrowserGPT {
  constructor(metadata, weightsBuffer) {
    this.metadata = metadata;
    this.weights = new Float32Array(weightsBuffer);
    this.size = metadata.n_embd;
    this.heads = metadata.n_head;
    this.headSize = this.size / this.heads;
    this.layers = metadata.n_layer;
    this.blockSize = metadata.block_size;
    this.vocabSize = metadata.vocab_size;
    this.tensorMap = metadata.tensors;
    this.embedding = this.tensor("transformer.wte.weight");
    this.positionEmbedding = this.tensor("transformer.wpe.weight");
    this.finalNormWeight = this.tensor("transformer.ln_f.weight");
    this.finalNormBias = this.tensor("transformer.ln_f.bias");
    this.layerWeights = Array.from({ length: this.layers }, (_, index) => ({
      ln1Weight: this.tensor(`transformer.h.${index}.ln_1.weight`),
      ln1Bias: this.tensor(`transformer.h.${index}.ln_1.bias`),
      qkvWeight: this.tensor(`transformer.h.${index}.attn.c_attn.weight`),
      qkvBias: this.tensor(`transformer.h.${index}.attn.c_attn.bias`),
      projWeight: this.tensor(`transformer.h.${index}.attn.c_proj.weight`),
      projBias: this.tensor(`transformer.h.${index}.attn.c_proj.bias`),
      ln2Weight: this.tensor(`transformer.h.${index}.ln_2.weight`),
      ln2Bias: this.tensor(`transformer.h.${index}.ln_2.bias`),
      fcWeight: this.tensor(`transformer.h.${index}.mlp.c_fc.weight`),
      fcBias: this.tensor(`transformer.h.${index}.mlp.c_fc.bias`),
      mlpProjWeight: this.tensor(`transformer.h.${index}.mlp.c_proj.weight`),
      mlpProjBias: this.tensor(`transformer.h.${index}.mlp.c_proj.bias`),
    }));
  }

  tensor(name) {
    const descriptor = this.tensorMap[name];
    if (!descriptor) throw new Error(`Missing model tensor: ${name}`);
    return this.weights.subarray(descriptor.offset, descriptor.offset + descriptor.length);
  }

  resetCache() {
    this.keyCache = Array.from({ length: this.layers }, () => new Float32Array(this.blockSize * this.size));
    this.valueCache = Array.from({ length: this.layers }, () => new Float32Array(this.blockSize * this.size));
  }

  forwardToken(tokenId, position) {
    let hidden = new Float32Array(this.size);
    const embeddingOffset = tokenId * this.size;
    const positionOffset = position * this.size;
    for (let i = 0; i < this.size; i += 1) hidden[i] = this.embedding[embeddingOffset + i] + this.positionEmbedding[positionOffset + i];

    for (let layerIndex = 0; layerIndex < this.layers; layerIndex += 1) {
      const layer = this.layerWeights[layerIndex];
      const normalized = layerNorm(hidden, layer.ln1Weight, layer.ln1Bias);
      const qkv = linear(normalized, layer.qkvWeight, layer.qkvBias, 3 * this.size, this.size);
      const keys = this.keyCache[layerIndex];
      const values = this.valueCache[layerIndex];
      const cacheOffset = position * this.size;
      for (let i = 0; i < this.size; i += 1) {
        keys[cacheOffset + i] = qkv[this.size + i];
        values[cacheOffset + i] = qkv[2 * this.size + i];
      }

      const attentionOutput = new Float32Array(this.size);
      const scale = 1 / Math.sqrt(this.headSize);
      for (let head = 0; head < this.heads; head += 1) {
        const headOffset = head * this.headSize;
        const scores = new Float32Array(position + 1);
        let maxScore = -Infinity;
        for (let time = 0; time <= position; time += 1) {
          const timeOffset = time * this.size + headOffset;
          let score = 0;
          for (let i = 0; i < this.headSize; i += 1) score += qkv[headOffset + i] * keys[timeOffset + i];
          score *= scale;
          scores[time] = score;
          if (score > maxScore) maxScore = score;
        }
        let normalizer = 0;
        for (let time = 0; time <= position; time += 1) {
          scores[time] = Math.exp(scores[time] - maxScore);
          normalizer += scores[time];
        }
        for (let i = 0; i < this.headSize; i += 1) {
          let value = 0;
          for (let time = 0; time <= position; time += 1) value += (scores[time] / normalizer) * values[time * this.size + headOffset + i];
          attentionOutput[headOffset + i] = value;
        }
      }

      const projectedAttention = linear(attentionOutput, layer.projWeight, layer.projBias, this.size, this.size);
      for (let i = 0; i < this.size; i += 1) hidden[i] += projectedAttention[i];

      const normalizedMlp = layerNorm(hidden, layer.ln2Weight, layer.ln2Bias);
      const feedForward = linear(normalizedMlp, layer.fcWeight, layer.fcBias, 4 * this.size, this.size);
      for (let i = 0; i < feedForward.length; i += 1) feedForward[i] = gelu(feedForward[i]);
      const projectedMlp = linear(feedForward, layer.mlpProjWeight, layer.mlpProjBias, this.size, 4 * this.size);
      for (let i = 0; i < this.size; i += 1) hidden[i] += projectedMlp[i];
    }

    const normalizedFinal = layerNorm(hidden, this.finalNormWeight, this.finalNormBias);
    const logits = new Float32Array(this.vocabSize);
    for (let token = 0; token < this.vocabSize; token += 1) {
      const tokenOffset = token * this.size;
      let value = 0;
      for (let i = 0; i < this.size; i += 1) value += this.embedding[tokenOffset + i] * normalizedFinal[i];
      logits[token] = value;
    }
    return logits;
  }

  async generate({ inputIds, bosId, eosId, stopSequenceIds = [], maxNewTokens = 199, temperature = 0.8, onProgress } = {}) {
    this.resetCache();
    let position = 0;
    const promptIds = inputIds?.length ? inputIds : [bosId];
    let logits;
    for (const tokenId of promptIds) {
      if (position >= this.blockSize) break;
      logits = this.forwardToken(tokenId, position);
      position += 1;
    }
    const generated = [];
    const tokenLimit = Math.min(maxNewTokens, this.blockSize - position);

    for (let step = 0; step < tokenLimit; step += 1) {
      const nextToken = softmaxSample(logits, temperature);
      if (nextToken === eosId) break;
      generated.push(nextToken);
      const sequenceStart = generated.length - stopSequenceIds.length;
      const foundStopSequence = stopSequenceIds.length > 0 && sequenceStart >= 0 && stopSequenceIds.every((tokenId, index) => generated[sequenceStart + index] === tokenId);
      position += 1;
      if (onProgress) onProgress(step + 1, tokenLimit);
      if (foundStopSequence) break;
      if (position >= this.blockSize) break;
      logits = this.forwardToken(nextToken, position);
      // Yield occasionally so the button state and loading animation can update.
      if (step % 4 === 3) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return generated;
  }
}
