## Files that needs to be removed before release

- pyine/configs/accelerate/hostfile
- pyine/configs/accelerate/\* # Possibly?

## Things that need to be cleaned up / refactored:

- The "guardrail" scorers are currently defined in different packages (i.e. evals pipeline or
  guardrails package), and we should probably have a single place for them (guardrails package?).
