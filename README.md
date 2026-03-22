# Cascaded proteomics search pipeline

This package implements a **report-centric cascaded search pipeline** for DDA and DIA mass-spec data, with a focus on rare peptide groups such as non-canonical ORFs, variants, and entrapment sets.

The pipeline is built around the workflow you described:

1. concatenate selected FASTA groups into a single search FASTA while preserving group identity,
2. run a search engine,
3. optionally rescore the results,
4. estimate FDR with several strategies,
5. write all intermediate tables and plots into a transparent report directory,
6. optionally trim confident DDA MS2 spectra before the next cascade step.

The current implementation prioritizes reliability and auditability over breadth. A few deliberately omitted features are listed under **Limitations** below.

## What is implemented

### Search engines

- **Sage**
- **DIA-NN**

The code is organized through a common engine interface, so new engines can be added by implementing one class.

### Optional rescoring

- **MS2Rescore** for Sage searches
- **Oktoberfest** for Sage searches

Rescorers use a common interface and merge their outputs back into the standardized row table.

### FDR strategies

Implemented at **row/PSM level** and **peptide level**:

- `all_together`
- `per_group`
- `transferred_subgroup` (Fu & Qian style transferred subgroup FDR)
- `group_walk`

### Plots and tables

For each search step and each level (row/PSM and peptide), the pipeline writes:

- score distribution plots by group and target/decoy
- cumulative score curves
- identifications vs q-value
- per-peptide-length versions of the above
- entrapment bound plots if entrapment groups are present
- transferred-subgroup gamma-fit diagnostics
- pairwise overlap tables between FDR methods
- accepted target lists at user-defined q thresholds

### Cascaded trimming

- **DDA trimming is implemented**: the pipeline can rewrite mzML files and remove MS2 spectra corresponding to accepted target PSMs.
- **DIA trimming is not implemented** in this version. The pipeline writes a note and can either skip or fail, depending on config.

## Important design choices

### Group-aware FASTA concatenation

Each FASTA group is rewritten into a combined FASTA with stable accession tokens:

- target example: `grp=canonical|P12345`
- supplied decoy example: `rev_grp=novel|ALT_0001`

That keeps group membership visible all the way through search, rescoring, FDR, and plotting.

### Decoy handling

For a given combined search step, all selected FASTA groups must be in the same mode:

- **all groups supply decoys**, or
- **no groups supply decoys** and the search engine generates them.

Mixed mode is rejected on purpose because it becomes easy to mis-calibrate group-aware FDR.

### Entrapment support

Any FASTA group can be flagged as `is_entrapment=true`.

The pipeline then computes approximate entrapment-based FDP bounds and stores the effective search-space ratio `r_effective`, estimated from in-silico digestion of the selected FASTA groups under the configured protease.

### Peptide-level aggregation

Peptide-level analysis is intentionally simple and transparent:

- one representative row per `(label, peptide)` pair,
- selected by the best available score.

This is **not** a picked-peptide implementation. That is a deliberate simplification to keep the first version auditable and robust.

## Limitations

These are intentional in this version.

1. **No DIA XIC trimming yet.**
2. **Peptide-level aggregation is simple best-representative aggregation**, not picked-peptide competition.
3. **Transferred subgroup FDR is implemented as a practical approximation** using a fitted subgroup decoy-fraction transfer function. It is useful for experiments, but I would still treat it as a diagnostic / comparative method rather than unquestioned production truth.
4. **Group-walk is implemented directly in Python**, adapted from the reference algorithm. It should be treated as experimental but reproducible.
5. For **DIA-NN**, the row-level table is precursor/report-row level. It is still stored under the `psm` branch of the report directory for layout consistency.

## Installation

Python requirements are intentionally small:

```bash
pip install -r requirements.txt
```

External tools are **not** bundled. Install them separately and point to them in the config:

- Sage
- DIA-NN
- MS2Rescore (optional)
- Oktoberfest (optional; invoked through `python -m oktoberfest`)

## Running

With the console entry point:

```bash
cascade-ms-pipeline path/to/config.json
```

Or directly:

```bash
python -m cascade_ms_pipeline path/to/config.json
```

Validate a config without running any searches:

```bash
python -m cascade_ms_pipeline path/to/config.json --validate-only
```

Print the resolved config after validation:

```bash
python -m cascade_ms_pipeline path/to/config.json --print-config --validate-only
```

## Configuration overview

The config is JSON and has three top-level sections:

- `general`
- `fasta_groups`
- `searches`

### `general`

Example:

```json
{
  "report_dir": "report",
  "spectra": ["run1.mzML", "run2.mzML"],
  "acquisition": "dda",
  "protease": "trypsin",
  "fragmentation": "hcd",
  "keep_intermediate_spectra": false,
  "plot_format": "png",
  "dry_run": false,
  "random_seed": 1,
  "binaries": {
    "sage": "sage",
    "diann": "diann",
    "ms2rescore": "ms2rescore",
    "python": "python"
  }
}
```

### `fasta_groups`

Each FASTA entry is a peptide group in the Fu & Qian sense.

```json
[
  {
    "name": "canonical",
    "path": "db/canonical.fasta",
    "supplies_decoys": false,
    "decoy_prefix": "rev_",
    "is_entrapment": false,
    "description": "Canonical proteome"
  },
  {
    "name": "noncanonical",
    "path": "db/noncanonical.fasta",
    "supplies_decoys": false,
    "decoy_prefix": "rev_",
    "is_entrapment": false,
    "description": "Novel ORFs / alternative proteins"
  },
  {
    "name": "entrapment",
    "path": "db/archea_entrapment.fasta",
    "supplies_decoys": false,
    "decoy_prefix": "rev_",
    "is_entrapment": true,
    "description": "Entrapment set"
  }
]
```

### `searches`

A search step contains:

- engine choice
- engine params
- selected FASTA groups
- optional rescorers
- FDR config
- optional trimming

Example skeleton:

```json
{
  "name": "01_canonical_first",
  "engine": {
    "type": "sage",
    "params": {
      "extra_args": [],
      "config_overrides": {
        "database": {
          "enzyme": {
            "missed_cleavages": 2,
            "semi_enzymatic": false
          }
        }
      }
    }
  },
  "fasta_groups": ["canonical", "entrapment"],
  "rescore": [
    {"type": "ms2rescore", "params": {}}
  ],
  "fdr": {
    "methods": ["all_together", "per_group", "transferred_subgroup", "group_walk"],
    "correction": 1.0,
    "alpha_grid": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
    "score_source": "final",
    "groupwalk_k": 40,
    "groupwalk_seed": 1,
    "transferred_min_decoys": 20,
    "transferred_min_points": 8,
    "transferred_clip_min": 1e-6,
    "entrapment_strategy": "unambiguous"
  },
  "trim": {
    "enabled": true,
    "method": "per_group",
    "level": "psm",
    "alpha": 0.01,
    "unsupported_action": "skip"
  }
}
```

## Search-engine notes

### Sage

Useful engine parameters live in `engine.params`.

Supported convenience fields include:

- `config_path`
- `config_overrides`
- `extra_args`
- `write_pin`

If no template is supplied, a default Sage config is generated and then overridden.

### DIA-NN

Useful engine parameters include:

- `lib`
- `out_lib`
- `fasta_search`
- `predictor`
- `gen_spec_lib`
- `threads`
- `qvalue`
- `verbose`
- `cut`
- `min_pep_len`
- `max_pep_len`
- `extra_args`

## Rescoring notes

### MS2Rescore

Configured as:

```json
{"type": "ms2rescore", "params": {}}
```

Useful optional params:

- `spectra_path`
- `config_path`
- `processes`
- `extra_args`

### Oktoberfest

Configured as:

```json
{"type": "oktoberfest", "params": {}}
```

Useful optional params:

- `spectra_path`
- `search_results_type`
- `fdr_estimation_method`
- `kind`
- `instrument_type`
- `intensity_model`
- `irt_model`
- `prediction_server`
- `numThreads`
- `massTolerance`
- `unitMassTolerance`
- `ce_range`
- `extra_config`

## Report directory layout

A typical report tree looks like this:

```text
report/
  config.resolved.json
  input_spectra.tsv
  pipeline.log
  pipeline_summary.json
  software_versions.tsv
  steps/
    01_canonical_first/
      combined_fasta/
      engine/
      normalized/
      rescore/
      fdr/
        psm/
          all_together/
          per_group/
          transferred_subgroup/
          group_walk/
        peptide/
          ...
      trim/
```

### What is stored

The pipeline stores all intermediate *tables* it creates, for example:

- combined FASTA manifest
- normalized engine outputs
- rescoring merge reports
- q-value tables per method and level
- diagnostic tables
- accepted hit tables at each alpha
- overlap tables
- trimming manifests and trim summaries

The only large objects that may be removed are **intermediate trimmed spectra**, controlled by `general.keep_intermediate_spectra`.

## Extending the code

### Add a new search engine

Implement a new subclass of `SearchEngine` in `cascade_ms_pipeline/engines/` and register it in `engines/__init__.py`.

The engine should return a standardized row table with at least:

- `row_id`
- `source_file`
- `scan` or `spectrum_id`
- `peptide`
- `modified_peptide`
- `proteins`
- `label`
- `score_engine`
- `engine_q`

### Add a new rescorer

Implement a subclass of `Rescorer` in `cascade_ms_pipeline/rescorers/` and register it in `rescorers/__init__.py`.

### Add a new FDR method

Implement a function in `fdr.py` and add it to `run_fdr_method`.

### Add a new trimming strategy

The current DDA trimmer lives in `trimming.py`. You can add a new strategy and call it from `PipelineRunner._run_trimming`.

## Suggested experiments to add next

A few extensions that would be particularly interesting for your use case:

- open-search branch before non-canonical rescue
- explicit “best canonical alternative” margin features during rescoring
- replicate reproducibility summaries across runs
- class-conditional entrapment matched by peptide length or composition
- de novo branch on the residual spectra after canonical and modified searches
- DIA precursor/XIC suppression between cascade steps

## Example configs

Two example configs are included:

- `example_config_dda.json`
- `example_config_dia.json`

