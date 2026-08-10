# Version 2.3 component contracts

All component contracts passed before Mode A. No official evaluator or API call was made, and the final DEV target was not opened.

## Inputs and cache boundary

- Raw and option-augmented inputs each contain 55 unique records.
- Exactly 41 records contain ordered multiple-choice option sets.
- The production input contains no `gold`, `answer_label`, or `correct_answer` key.
- The archived option-order/leakage audit passes.
- The cache boundary matches SHA-256 `54DA46600AFE81DAB5D8D2F10E87AC453FF5FCA206296DAF126EFFCD4D4C409D`, has 55 unique IDs, and uses the stable prediction schema.

## Tables and freeform

- The accepted cache-to-table ordering is recorded in the lineage graph.
- The verified pre-completeness parent contains 11 table answers and 45 non-empty row objects.
- The accepted freeform order is same-table completeness, source-sentence expansion, paper-selection completeness, then chart-difference completeness.
- All 17 archived counterfactuals pass. Each of the four accepted answer changes has a source trace and declares no production gold use.

## Evidence and multiple choice

- Evidence-safe cleanup changes only `evidence`, on q_049 and q_050. The excluded q_045 pruning is absent.
- The option-aware solver passes 410/410 archived permutation tests.
- Source-grounded semantic answers cover 41 MC records.
- The typed representation, slot filling, option matching, and generic activation modules are present.
- No literal query-ID production branch was found in the accepted MC module set.

## Final-parent contract

The source-grounded parent contains exactly 55 unique IDs, no missing or extra records, and the expected `answer|evidence|gold_papers|query_id` schema. The frozen final DEV target remains unopened until Mode A freezes its output.

Machine-readable evidence is in `records/COMPONENT_TEST_RESULTS.json`.
