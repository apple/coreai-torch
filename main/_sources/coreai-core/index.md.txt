# Core AI (coreai-core)

`coreai-core` is the Python package for building and running AI Models on Apple hardware. Use it to:

- Author an AI Model from Python (`coreai.authoring`)
- Run an existing `.aimodel` file (`coreai.runtime`)

## Get started

Install:

```bash
pip install coreai-core
```

Then work through the two tutorials in order:

```{toctree}
:maxdepth: 1
:caption: Tutorials
:hidden:

tutorials/construct-a-graph
tutorials/run-an-aimodel
```

| Tutorial | What you'll learn |
|----------|-------------------|
| [Construct a graph](tutorials/construct-a-graph) | Build an AI Model in Python and save it as a `.aimodel` file. |
| [Run an `.aimodel`](tutorials/run-an-aimodel) | Load an `.aimodel` and run inference with NumPy inputs. |

## API reference

```{toctree}
:maxdepth: 1
:caption: API Reference
:hidden:

api/coreai
```

| Module | Use it for |
|--------|------------|
| [`coreai.runtime`](api/coreai) | Loading and executing `.aimodel` files. |
| [`coreai.authoring`](api/coreai) | Building AI Models programmatically. |
