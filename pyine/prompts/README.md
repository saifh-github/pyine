# `pyine.prompts`

This package provides a small, test-backed toolkit to author, version, load, render, and run LLM
prompts. It is centered around YAML templates that specify the contents of the prompts in a
structured and reference-friendly (DRY) fashion.

Focus: only prompting. Everything below relates to prompt templates, their Python helpers, and an optional results log.

**What you get:**

- Versioned YAML prompts under `pyine/prompts/templates` and their associated dataclasses and
  structured output parsers under `pyine/prompts/configs`;
- Simple helpers to list prompts, load a prompt config, render a prompt, or build a Runnable chain;
- Optional SQLite-backed prompt results database for caching/analysis of generations.

______________________________________________________________________

## Quick start

- Listing available prompts and using one of them:

```python
from pyine.prompts import list_prompts, get_prompt_template

print(list_prompts())
# e.g. ["callable_analysis", "code_analysis", "code_execution", "code_summary", "code_stubbing", ...]
pt = get_prompt_template("code_analysis")  # default version unless specified
text = pt.format(code='print("Hello")')  # provide input arguments for the prompt here
print(text)
```

- Building a runnable chain for a given language model:

```python
from pyine.prompts import get_prompt_chain

# model = ...  # any `langchain_openai.chat_models.base.BaseChatOpenAI`-derived model
chain = get_prompt_chain(model, "code_analysis")
result = chain.invoke({"code": 'name = input("Enter name: ")\nprint("Hello, " + name)'})
# if the prompt module provides an output parser, the above result is a structured object
```

- Controlling examples and chat/text format:

```python
from pyine.prompts import get_prompt_template

# get a plain template containing no example
pt = get_prompt_template("code_analysis", include_examples=False)
# get a template containing one random example (instead of all, by default)
pt_w_ex = get_prompt_template("code_analysis", include_examples=True, target_examples=1)
# get a ChatPromptTemplate instead of a text template
chat_pt = get_prompt_template("code_analysis", use_chat_template=True)
```

- Loading a prompt config (includes metadata, fillable meta-templates, and examples):

```python
from pyine.prompts import get_prompt_config, list_prompt_versions

cfg = get_prompt_config("code_analysis")
print(cfg.metadata.name, cfg.metadata.version)
print(list_prompt_versions("code_analysis"))  # ["v1.0", ...]
```

### Anatomy of a YAML prompt template (per version)

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

## Prompt manager (`pyine/prompts/manager.py`)

The `PromptManager` is a singleton that loads YAML files, resolves Pydantic tags, caches prompt
configs, lists available prompts and versions, and constructs LangChain `PromptTemplate` objects.
Important APIs include:

- `PromptManager.get_prompt_config`: loads a prompt config from a YAML file (with in-memory cache).
  This is the main entry point for tooling and notebooks to fetch a prompt.
- `PromptManager.get_prompt_template`: instantiates a LangChain `PromptTemplate` object from a
  prompt config, with some options related to examples.
- `PromptManager.list_prompts`: lists all available prompts.
- `PromptManager.list_prompt_versions`: lists all available versions for a prompt.

## Prompt result database (`pyine/prompts/result_db.py`)

The `PromptResultDB` class implements a database interface that relies on SQLite to fetch previous
generations or log new ones. Example usage via fire-and-forget helper that fetches existing results
or generates new ones:

```python
import datetime
import pyine.prompts
import pyine.utils.llm_providers

records = pyine.prompts.fetch_or_generate_prompt_results(
    identifier="dataset-item-42",
    input_variables={"code": "print('hi')"},
    prompt_chain_config=pyine.prompts.PromptChainBuildConfig(
        prompt=pyine.prompts.PromptBuildConfig(
            prompt_name="code_summary",
            # ... other args if needed
        ),
        provider=pyine.utils.llm_providers.LLMProviderConfig(
            # ...
        ),
    ),
    max_result_age=datetime.timedelta(days=7),
)
for r in records:
    print(r.result)  # str (raw text) or JSON string when the chain returns a model/dict
# if you need typed outputs (e.g. pydantic models) from records, see `pyine.prompts.TypedPromptResultFetcher`
```

You can also work directly with the database:

```python
from pyine.prompts.result_db import PromptResultDB

db = PromptResultDB()  # defaults to a framework path
v = db.store(identifier="id1", prompt="...", result="...", tags=["ok"], group="g1")
records = db.get_by_identifier("id1")
```

## Creating a new prompt

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
     so that YAML files that refer to them can be loaded transparently. For these models to also be
     automatically used for structured output parsing in langchain runnables, you should also expose
     `get_output_parser(version) -> langchain_core.output_parsers.BaseOutputParser | None`.
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
[top-level README](../../README.md) for more details on how to contribute.
