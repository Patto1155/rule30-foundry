# Starter Directions

Possible directions to test against Rule 30 style questions:

## 1. Center-Column Predictors

Look for compact models that predict the center column better than naive baselines.

- test shallow Markov models
- test block-frequency features
- test local neighborhood embeddings

## 2. Reversible Summaries

Try to compress observed prefixes into summaries that preserve future-relevant structure.

- test suffix automata or grammar-style compression
- compare compression ratio against prediction accuracy
- record whether the summary is stable across seed lengths

## 3. Phase / Regime Detection

Search for segments with different statistical behavior.

- compare early, middle, and late windows
- track changes in entropy, bias, and run length
- look for recurring motifs that might signal hidden structure

## 4. Search Over Representations

Treat the task as a representation search problem.

- bits, blocks, symbols, and feature transforms
- simple learned features before complex models
- keep the feature set tiny so failures are interpretable

## 5. Falsification First

For any promising idea, define the quickest way to prove it wrong.

- what pattern would invalidate it?
- what horizon would make it fail?
- what baseline must it beat?
