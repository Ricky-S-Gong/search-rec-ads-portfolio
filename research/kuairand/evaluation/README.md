# Week 1 validation protocol

`training/run.py` is the shared CLI. Run commands in the locked training environment.

- `prepare-week1 --source ... --archive ... --output ...`: verify Drive source hashes and official Pure archive MD5; match raw request identities; remove exact full-row duplicates; rebuild positive histories with `time_ms + play_time_ms < request_time`.
- `evaluate --manifest ... --output ... --label popularity`: train-window counts against every retained validation positive and the full eligible catalog.
- `evaluate --manifest ... --checkpoint ... --vectors ... --output ... --label basic|enhanced`: strict checkpoint/vector/config/code binding and complete candidate coverage.
- `verify-resume --manifest ... --checkpoint ... --output ...`: interrupt the enhanced configuration at step 25, restore it and compare parameters, optimizer, scaler, RNG and traversal to the full reference.
- `stage-report --results ... --resume-check ... --output ...`: compare only identical protocols; completion requires both CUDA runs and matching CUDA recovery evidence.

UTC split boundaries are explicit Unix timestamps. Raw `date` disagrees with fixed UTC and UTC+08 calendars, so it is not used for split assignment. Playback length is a conservative completion proxy, not independently logged completion. Upload date gives next-day eligibility; visibility snapshots are unavailable. Repeats stay eligible. Test data are not evaluated.

The formal enhanced model adds masked mean history to an ID-only item tower. All unversioned content metadata is excluded from the new comparison. The historical 8-category/4-numeric model remains a separate experiment.

Tie order is score descending then raw item ID ascending. Unavailable targets count as misses; conditional covered metrics are supplementary. Reported latency is per batch after warmup, not single-request service latency.

Private request-level predictions stay in Drive. Public reports contain only aggregates. The published dataset is immutable: `datasets/week1-pure-completion-v1` in the team Drive, with checksums in `training/configs/week1-data-release.json`.
