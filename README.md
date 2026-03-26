# spring-2026-real-time-sleep-stage-estimation
Team project: spring-2026-real-time-sleep-stage-estimation

## Notebook Git Hygiene

This repo uses `nbstripout` for `.ipynb` files so notebook outputs and most execution metadata do not create noisy diffs or merge conflicts.

Run this once after installing dependencies:

```bash
nbstripout --install --attributes .gitattributes
```

If you already have notebook metadata staged, re-add the notebook after installing the filter so Git stores the stripped version.
