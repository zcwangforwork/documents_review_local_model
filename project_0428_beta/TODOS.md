# TODOS

## 1. Add per-standard specialized audit prompts

- **What:** Create detailed checklist-style audit prompts for ISO 13485 (QMS), IEC 62304 (software lifecycle), and MDR compliance, matching the depth of the existing ISO 14971 risk management prompt.
- **Why:** Currently only ISO 14971 has a detailed prompt. The general audit prompt is shallow. 80% of medical device document types are not well-served.
- **Pros:** Covers the majority of document types. The multi-pass pipeline already supports any prompt per section.
- **Cons:** Each prompt needs domain expertise to write well.
- **Context:** The multi-pass pipeline (implemented in this PR) supports per-section prompts. Infrastructure is ready, prompts are missing. Start with ISO 13485 as it's the most common.
- **Depends on:** Multi-pass pipeline implementation (this PR)
