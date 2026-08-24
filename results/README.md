# Results

Files ending in `_smoke` are deliberately short validation runs used to verify the repository end to end. They are not publication benchmarks.

To regenerate a full controlled jet-recovery experiment:

```bash
python benchmarks/jet_recovery.py --epochs 300 --seeds 3 --output results/jet_recovery.json
```
