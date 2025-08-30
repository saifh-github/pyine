# PyINE Prompt Framework

@@@@@ TODO: update this readme w/ latest changes, and to document the prompt result database

### High-level overview

- The `pyine.prompts` package provides a small, test-backed framework to author, version, load,
  and render LLM prompts from YAML configuration files.
- The YAML files live under `pyine/prompts/templates` and define one or more versions of a prompt.
  Each version defines four basic blocks for each prompt, namely **role**, **context**, **examples**, and
  **question**:
  - The **role** block provides the initial instructions provided to the LLM that should describe
    its role/specialty, as a kind of "early context" that should be mostly task-independent.
  - The **context** block provides task-specific instructions that describe what is needed of the LLM.
  - The **examples** block provides in-context examples that the LLM should use as inspiration or
    guidance for answering questions; these can be structured individually and under a group template.
  - The **question** block provides the actual question that the LLM should answer.
- Python "config" modules live under `pyine/prompts/configs` and can optionally:
  - Define structured output models (Pydantic);
  - Compute dynamic partial variables to inject into the template (e.g., output format instructions); and
  - Expose convenience helpers (e.g. `get_prompt_config` and `get_prompt_template`).
- A centralized `PromptManager` loads YAML files, resolves Pydantic tags, caches prompt configs,
  lists available prompts and versions, and constructs LangChain `PromptTemplate` objects.

### Key benefits

- **Versioned prompts with defaults.** A single YAML can host many prompt versions; a `__default__`
  key selects the default version that is loaded by the manager. If no default is specified, the last
  version defined in the file is taken as the default.
- **Strongly typed examples and outputs.** YAML files can embed tagged Pydantic objects, which
  are automatically registered and instantiated when the files are loaded; your prompt unit tests
  can then directly validate round-tripping and schema correctness.
- **Render-time flexibility.** You can mix f-string and jinja2 templates to configure templates and
  make them parameter-dependent; you can also choose zero, some, or all examples to be inserted in
  prompts at render time; finally, you can pass extra variables to blocks as needed.
- **Structured output guidance.** Pair prompts with Pydantic models and LangChain’s
  `PydanticOutputParser` to consistently steer model outputs.
- **Simple discovery.** `list_prompts` and `list_prompt_versions` allow simple scripts and notebooks
  to discover all available prompts and their variants.

### Architecture and main concepts

#### Versioned YAML templates (`pyine/prompts/templates`)

Each file contains a named prompt with one or more versions. Some prompts are grouped in
subdirectories based what prompt family they belong to (e.g., the `hints/tests.yaml` prompt config is
under the same `hints` family as the `hints/docs.yaml`, as both address the task of injecting hints in
code).

**High-level YAML schema per version:**

```yaml
v1.0-universal:  # arbitrary name that identifies the VERSION of the prompt
  metadata:  # required field that provides some human-readable information about the prompt
    name: code_analysis  # 'name' of the prompt that is used by the prompt manager to answer queries
    description: Short human description of intent.
  role:  # optional, provides the initial task-independent instructions provided to the LLM
    template: "..."
    format: f-string | jinja2  # optional, defaults to f-string
    partial_variables: {...}   # optional
  context:  # optional, provides task-specific instructions to the LLM
    template: "..."
    format: f-string | jinja2  # optional, defaults to f-string
    partial_variables: {...}   # optional
  question:  # required, specifies the actual question that the LLM should answer
    template: "..."
    format: f-string | jinja2  # optional, defaults to f-string
    partial_variables: {...}   # optional
  example_template:  # optional; provides the template used to render in-context examples
    template: "..."
    format: f-string | jinja2  # optional, defaults to f-string
    partial_variables: {...}   # optional
  examples_block_template:  # optional, provides the template used to render a list of examples
    # note: a default is used in case examples are provided without a block template
    template: "..."
    format: f-string | jinja2  # optional, defaults to f-string
    partial_variables: {...}   # optional
  template_block_separator: "\n\n"  # optional; separates role/context/examples/question blocks
  examples:  # optional, specified the data used to render examples
    - input_variables: {...}  # input variables required to render the example
      output: !package.module.Model  # Pydantic model tag or plain string/JSON-like
        ... # task-specific output
      description: "..."  # optional
    - input_variables: {...}
      output: !package.module.Model
        ...
      description: "..."

v1.0-openai:  # some other version with e.g. specificities that improve performance with a particular LLM
  ...

__default__: "v1.0-universal"  # optional, prompt version to use by default when unspecified
# (if no default version is specified as above, the fallback is to use the last specified version)
```

**Special YAML keys:**

- __default__: string: default prompt version that the manager returns when none is specified.
- __defines__: mapping: used to define local anchors/aliases (group is skipped by the prompt loader).

**Example-specific conventions (enforced by config loader):**

- Examples can contain an arbitrary number of input variables (under the `input_variable` group),
  which will be filled into the `example_template` prompt that you should also specify.
- Examples must contain an `output` field that specifies the expected output value that models should
  predict. The `example_template` must also refer to this `output` variable. That output does not
  however need to be tied to a specific structure or Pydantic model.
- You may optionally use `example_idx` and `example_count` to number examples in your
  `example_template`; for example, this is a valid template even if those parameters are never
  specified in the `input_variables` of the provided examples:
  ```yaml
  ...
  # the example template specifies how examples will be individually rendered based on their parameters
  example_template:  # optional; provides the template used to render in-context examples
    template: |
      Example {example_idx}/{example_count};
        Q: {question}
        A: {output}
    # by default, the above template is assumed to be in f-string format
  ...
  examples:
    # the 'examples' group should provide a list of examples specifying template parameters
    - input_variables:
        question: "..."
      output: "..."
    - input_variables:
        question: "..."
      output: "...""
  ```

**Extra notes:**

- The YAML loader supports YAML anchors and aliases. This is useful for DRYing up template parts.
  You will see this in some of the existing templates (look for templates using `__defines__` keys).
- The YAML loader supports Pydantic tags (e.g., `!pyine.prompts.configs.code_analysis.CodeAnalysisResponse`),
  as long as the module containing the Pydantic model is registered with the framework manager. This manager
  automatically registers all Pydantic models found under the pyine package for YAML tag resolution.
  You can also register models manually with `pyine.utils.pydantic.PydanticYAMLLoader.register_models_from_module`.
- Template formatting supports both f-string and jinja2. Prefer f-string for safety; use jinja2 only
  when you control input.

#### Prompt config model and manager (`pyine/prompts/utils.py` and `pyine/prompts/manager.py`)

The `PromptConfig` class wires the four prompt blocks, manages examples, and constructs LangChain
`PromptTemplate` objects on demand. Important APIs include:

- `PromptConfig.create_prompt_template`: creates a LangChain `PromptTemplate` object from the
  role/context/examples/question blocks. Can optionally return a chat template as well.
- `PromptConfig.render_prompt`: renders a complete prompt with the provided variables.
- `PromptConfig.get_examples_as_text`: renders the examples block with the provided variables.
- `VersionedPromptConfig.from_yaml`: parses a YAML file into a version-to-prompt-config map, and
  resolves the default prompt version to be used.

The `PromptManager` is a singleton that loads YAML files, resolves Pydantic tags, caches prompt
configs, lists available prompts and versions, and constructs LangChain `PromptTemplate` objects.
Important APIs include:

- `PromptManager.get_prompt_config`: loads a prompt config from a YAML file (with in-memory cache).
  This is the main entry point for tooling and notebooks to fetch a prompt.
- `PromptManager.get_prompt_template`: instantiates a LangChain `PromptTemplate` object from a
  prompt config, with some options related to examples.
- `PromptManager.list_prompts`: lists all available prompts.
- `PromptManager.list_prompt_versions`: lists all available versions for a prompt.

### Basic usage

Listing available prompts and fetching/rendering a prompt template:

```python
import pyine.prompts.manager

available = pyine.prompts.manager.list_prompts()
# e.g. ["callable_analysis", "code_analysis", "code_execution", "hints/stubs", "hints/tests", ...]

prompt_config = pyine.prompts.manager.get_prompt_config("code_analysis")  # returns default if version unspecified
print(f"prompt metadata: {prompt_config.metadata}")
prompt_template = pyine.prompts.manager.get_prompt_template("code_analysis")
rendered = prompt_template.format(code='print("hi")')
print(f"example rendered prompt:\n\n{rendered}")
```

Rendering a prompt with examples disabled, or with specific examples:

```python
import pyine.prompts.manager

prompt_template = pyine.prompts.manager.get_prompt_template("code_analysis", include_examples=False)
# or: include_examples=True, target_examples=[0, 2] or target_examples=1 (random 1 example)
```

### Creating a new prompt

1. Add a YAML template file under `pyine/prompts/templates`:
   - Choose a file path; this path (minus `.yaml`) becomes the prompt name, for example:
     `some_group/my_prompt.yaml` should have `"some_group/my_prompt"` as its prompt name that the
     prompt manager will recognize for later queries.
   - Provide one or more version blocks (where versions are named however you want). Optionally set
     `__default__: "<some-existing-version-key>"`.
   - Define at least the `metadata` and `question` fields (see the schema description above for info).
   - Optionally add `role`, `context`, `example_template`, `examples`, `examples_block_template`, and
     `template_block_separator` fields, which make up the "system prompt".
   - If including examples, ensure `example_template` contains an "output" variable. With jinja2 you
     can write `{{output}}` (or `{{output.model_dump_json()}}` if it’s a Pydantic object). You may
     reference `{{example_idx}}` and `{{example_count}}` in the example template; the engine fills
     them in when requested.
2. (Optional but recommended) Add a config module under `pyine/prompts/configs`.
   - Create a matching Python module path mirroring your YAML file name. For example, for a template
     `some_group/my_prompt.yaml`, create a module named `pyine/prompts/configs/some_group/my_prompt.py`.
   - Define Pydantic models to structure outputs and examples, if needed; they will be auto-registered
     so that YAML files that refer to them can be loaded transparently.
   - Expose a `get_prompt_config(version: str | None = None) -> PromptConfig` function with any kind
     of custom handling your prompt config implementation might require.
   - Expose a `get_prompt_template(...) -> LangChain BasePromptTemplate` function, and optionally inject
     computed partial variables (e.g., structured output format instructions).
3. Write tests. See `tests/prompts` for patterns; they should cover discovery, prompt listing,
   defaults, and basic retrieval.

### Conventions and tips

**Template formats.** Prefer f-string format for safety. Use jinja2 only when you need logic
(e.g., conditional blocks) and control all input variables to the prompts.

**Variables.** Role/context/examples templates are fully rendered when the final `PromptTemplate`
is created. The question template is returned with its `partial_variables` preserved; you pass only
question variables at `.format()` time.

**Pydantic in YAML.** You can embed structured objects in examples via tags like
`!pyine.prompts.configs.code_analysis.CodeAnalysisResponse`. The framework's `PydanticYAMLLoader`
will instantiate them and enforce schema validity.

**Version naming.** There is no enforced version naming scheme. You can use semantic versions
(e.g. `v1.0`), provider/model-targeted versions (e.g. `openai/gpt-4o-2024-xx`), or hierarchical names
(e.g. `full_output/v1.0`). The key only has to match exactly when requested via the prompt manager.

**Contributing.** Keep YAML files self-contained and documented. Prefer adding realistic examples
that assert expected outputs for clarity and robustness. When adding Pydantic models, ensure their
fully qualified names are stable; tests should cover validation and prompt rendering. Run the
existing tests under tests/prompts and add new ones for your prompt and models. Refer to the
top-level README for more details on how to contribute.
