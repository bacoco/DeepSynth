# DeepSynth Operations Cheat Sheet

The following table summarises the primary operational scripts and entry points available in the repository.

| Command | Type | Location | Summary |
|---------|------|----------|---------|
| `deepsynth-pipeline` | Python CLI | `scripts/cli/run_complete_multilingual_pipeline.py` | Run the full multilingual pipeline with incremental uploads and environment checks. |
| `deepsynth-separate-datasets` | Python CLI | `scripts/cli/run_separate_datasets.py` | Build and publish the seven language-specific dataset shards. |
| `deepsynth-parallel` | Python CLI | `scripts/cli/run_parallel_processing.py` | Launch interactive helper for the parallel dataset builder. |
| `deepsynth-benchmark` | Python CLI | `scripts/cli/run_benchmark.py` | Evaluate trained models against the bundled benchmark suites. |
| `deepsynth-check-shards` | Python CLI | `scripts/maintenance/check_shards_duplicates.py` | Validate Hub shards to detect duplicate samples. |
| `python -m deepsynth.pipelines.global_incremental_builder` | Python module | `deepsynth/pipelines/global_incremental_builder.py` | Distributed-ready incremental builder that stores progress on the Hub. |
| `run_global_pipeline.sh` | Shell | Repository root | Wrapper that orchestrates the global incremental pipeline with env checks. |
| `scripts/shell/start-dataset-generation.sh` | Shell | `scripts/shell/` | Start the CPU-only dataset generation stack via Docker Compose. |
| `scripts/shell/start-model-training.sh` | Shell | `scripts/shell/` | Launch the GPU-enabled training stack via Docker Compose. |
| `scripts/shell/start-all.sh` | Shell | `scripts/shell/` | Convenience launcher that boots both CPU and GPU services. |
| `start_docker_ui.sh` | Shell | Repository root | Start the legacy Docker UI dashboard. |

All Python entry points are also exposed through the `pyproject.toml` `console_scripts` section, enabling installation-time command creation.
