# Python Project Instructions

## General philosophy

Write simple, readable Python.

Prefer the smallest solution that correctly solves the current problem.

Do not over-engineer code for hypothetical future requirements.

The existing codebase is the primary source of architectural decisions.

Before introducing a new pattern, search the repository for an existing pattern that solves a similar problem.

## Existing code first

Before creating a new:

* class
* service
* helper
* utility
* interface
* abstraction
* dependency

search the repository for an existing implementation.

Prefer extending an existing implementation when reasonable.

Do not create parallel implementations of functionality that already exists.

## Keep changes local

Change only what is necessary for the requested task.

Do not refactor unrelated code.

Do not rename or reorganize unrelated files.

Do not change public APIs unless the task requires it.

Avoid large architectural changes for small features.

## Python style

Prefer straightforward Python over clever Python.

Prefer small functions with clear responsibilities.

Use type hints where they improve clarity and are consistent with the existing project.

Avoid unnecessary classes when a function or small module is sufficient.

Do not introduce abstractions solely to satisfy a theoretical design principle.

Use existing project conventions for:

* imports
* error handling
* logging
* configuration
* dependency injection
* async code
* testing

## Dependencies

Do not add a dependency unless it provides meaningful value that cannot reasonably be provided by the existing project or Python standard library.

Before adding a dependency, explain why it is necessary.

## Error handling

Do not catch broad exceptions unless there is a concrete reason.

Do not silently swallow errors.

Follow the existing project's error-handling conventions.

## Tests

When changing behavior, add or update focused tests.

Prefer testing observable behavior rather than implementation details.

Do not write tests solely to increase coverage numbers.

## Before finishing

Review the git diff.

Remove:

* unused imports
* dead code
* unnecessary abstractions
* unrelated formatting changes

Run the relevant tests and checks.

## Anti-overengineering rule

When choosing between:

1. a simple local implementation
2. a generalized abstraction designed for possible future requirements

prefer the simple implementation unless there is a concrete current requirement for the abstraction.

Three obvious lines of code are often better than a framework created to avoid those three lines.
