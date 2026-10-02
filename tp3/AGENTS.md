# TP3 implementation and analysis guidance

## Scope and authority

- Apply these instructions to all work under `tps_sia/tp3/`.
- Follow the repository-level instructions and the requirements in [Enunciado TP3.md](Enunciado%20TP3.md).
- The recommendations below come from the user's class notes. Treat them as implementation and analysis guidance; do not present them as additional requirements quoted from the official assignment.
- Write agent instructions, new infrastructure filenames, and configuration keys in English. Write academic explanations and reports in Spanish, following repository conventions.
- Implement the required exercises before optional extensions.

## Parameters and hyperparameters

- Distinguish trainable parameters (weights and biases adjusted by the optimizer) from hyperparameters selected for an experiment.
- Make the following hyperparameters configurable and record their values with every experiment:
  - Number of hidden layers and neurons per layer.
  - Learning rate, including any schedule used to change it.
  - Maximum number of epochs.
  - Training strategy: online, batch, or mini-batch, including batch size and shuffling.
  - Activation functions and any parameters they expose.
  - Weight initialization method, scale, and random seed.
  - Optimizer and its settings; gradient descent is the course baseline.
  - Error or cost function.
  - Stopping tolerance (`epsilon`), the quantity it measures, and the stopping rule.
- Pay particular attention to hyperparameter analysis in Exercise 3. Record the evidence supporting the selected configuration.

## Learning rate and learning curves

- Start learning-rate exploration with small values, including `1e-5` and `1e-4`, as suggested in the class notes. These are starting points, not universally correct values.
- Justify the learning rate using the cost function's geometry, gradient scale, data normalization, and training strategy. Explain slow convergence, oscillation, or divergence when observed.
- Always record and plot error or cost against epochs, and explain how the learning rate affects that evolution. Record the actual learning rate per epoch if it changes during training.
- During generalization studies, also examine error on held-out data over epochs. Distinguish training, validation, and final test curves.
- Use validation data from the permitted development dataset for hyperparameter selection. For Exercises 2 and 3, preserve `digits_test.csv` for final generalization evaluation as specified in the assignment; do not select learning rates, architectures, or stopping epochs from its results.

## Backpropagation and weight updates

- Explain each neuron's delta as the local sensitivity of the cost to its preactivation. For hidden neurons, explain how it combines contributions from subsequent layers with the local activation derivative.
- State the sign convention, loss reduction (sum or mean), and resulting weight and bias update rules consistently.
- Make it possible to record and plot synaptic weight evolution and weight updates while implementing the algorithm. For small examples, inspect individual weights; for larger networks, use selected weights and per-layer summaries to keep the analysis readable.
- Associate recorded updates with an epoch or update index and the experiment configuration. Make detailed recording configurable to control storage and execution cost.

## Online, batch, and mini-batch training

- Online training updates parameters after each sample; it does not average gradients across multiple samples in that update. Explain the resulting sensitivity to sample order and variability in updates.
- Batch training aggregates gradients over the entire training dataset before updating parameters. Explain that it requires computing the contributions of all samples, including their deltas, for each update.
- Mini-batch training aggregates gradients over a subset of samples. Consider it as a practical balance between computation and update variability; shuffle samples reproducibly to introduce stochasticity.
- Justify the selected strategy and batch size. Do not assume that mini-batch training always produces a better model.
- Keep gradient reduction explicit: summing versus averaging changes the update scale and the interpretation of the learning rate.

## Architecture, activations, and normalization

- Select the number of layers and neurons through documented experimentation. Compare candidate architectures and justify the choice using performance, convergence, and computational cost.
- Use activations compatible with gradient-based backpropagation in multilayer networks. Do not use the simple perceptron's step activation in a standard multilayer implementation; any exceptional use requires an explicit justification and a compatible training method.
- Explain the output range of each selected activation and its compatibility with inputs, targets, and the task. Consider saturation when relevant.
- Inspect data ranges and normalize or standardize when needed. Fit preprocessing on training data and apply the same transformation to held-out data. Save preprocessing information with the model.

## Weight initialization

- Justify initialization by activation function, layer dimensions, and scale. Explain why uncontrolled large weights can cause saturation or unstable activations and why excessively small weights can also hinder learning.
- Break symmetry between neurons; do not initialize all weights identically.
- Record the initialization method and random seed so experiments can be reproduced and compared.

## Error, cost, and stopping criteria

- Choose and justify the error or cost function for each task. Distinguish the optimized cost from evaluation metrics such as accuracy.
- Document its derivative and compatibility with the output activation, including any normalization factors used in gradients.
- If choosing a cost different from the traditional course formulation, explain the change and its implications for backpropagation.
- Automatic differentiation is an optional implementation aid, subject to assignment constraints. Justify its use and explain the computed gradients; do not introduce it merely because an alternative cost is used.
- Define what counts as a sufficiently good solution. Make `epsilon` and epoch limits explicit, and report the reason training stopped.

## Pausing and resuming training

- Support saving a checkpoint, pausing training, loading the checkpoint, and continuing from the saved state.
- Save weights, biases, architecture, activation settings, preprocessing, and training configuration.
- Preserve optimizer state when applicable, epoch and update counters, learning-rate schedule state, and random generator state so resuming does not silently reset training behavior.
- Preserve or append training history so learning curves remain interpretable across resumed runs. Distinguish resuming a run from starting a new experiment with previously trained weights.

## Experiments and optional noise analysis

- Separate experiment execution and stored measurements from plotting and analysis. Save configurations, seeds, histories, and execution times with results.
- Explain comparisons using recorded results; do not infer success solely from a decreasing training error.
- If implementing the optional robustness study, consider Gaussian noise as suggested in the notes. Describe its continuous perturbations, mean, standard deviation, data scale, and any clipping; do not assume it is always gentler than other noise models.
- Evaluate multiple noise levels on the selected model and report how performance changes. Complete required work before this optional analysis.
