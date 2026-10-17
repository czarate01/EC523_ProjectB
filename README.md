# EC523 Project B: student materials

- `handout/project_b.pdf`: student project handout.
- `handout/project_b.tex`: editable LaTeX source with equations, diagrams,
  design choices, evaluation, competition rules, and grading expectations.
- `cli/`: exact-length enhancement runner, untrained interface example,
  public aggregation, SI-SDR, synthetic demo, and tests.

The course site supplies the versioned 16 kHz data package, split manifests,
trained common baseline, compute allocation, and scoring-server instructions.
The bundled model is an untrained interface example, not a performance baseline.

Compile from `handout/` with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error project_b.tex
```

The bibliography is embedded; no BibTeX step is required.
