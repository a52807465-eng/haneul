route_analysis_prompt = """You are an expert organic chemist performing a detailed multi-dimensional analysis of a proposed retrosynthetic route. The route is provided as a tree of disconnections. Each node represents a molecule being disconnected; it carries the disconnection reaction SMILES, the LLM's strategic reasoning, a detailed ops analysis, the atomic editing operations (JSON) used to generate the reactants, proposed reaction conditions, and a critic verdict.

Target molecule:
<target_molecule>
{{TARGET_MOLECULE}}
</target_molecule>

User constraint / steer query:
<user_constraint>
{{USER_CONSTRAINT}}
</user_constraint>

Route tree (retrosynthetic, root = target):
<route_tree>
{{ROUTE_TREE}}
</route_tree>

INDEX SYSTEM — read carefully before writing any output:
- Every molecule node is labelled [Mol X] where X is a letter (A, B, C, … AA, AB, …), assigned depth-first.
- Every reaction step is labelled [Step N] where N is an integer (0, 1, 2, …), assigned depth-first.
- When your output refers to a reaction, write ONLY its step_index integer — do NOT copy the reaction SMILES string.
- When your output refers to a molecule, write ONLY its mol_index letter — do NOT copy the SMILES string.
- The SMILES will be resolved and injected automatically after your response is parsed.

---

Perform the following thirteen analyses. For EACH one, output ONLY the corresponding XML tag with a JSON payload inside — no prose outside the tags.

══════════════════════════════════════════════════════
1. ROUTE SUMMARY
══════════════════════════════════════════════════════
Provide a concise 2–4 sentence overview of the overall synthetic strategy: the logic of the disconnection sequence, the key bond-forming events, and how the route relates to the user constraint.

<route_summary>
{
  "summary": "<string>"
}
</route_summary>

══════════════════════════════════════════════════════
2. KEY STEPS
══════════════════════════════════════════════════════
Identify the 1–3 most strategically important or chemically challenging reactions. Reference each by its step_index.

<key_steps>
{
  "key_steps": [
    {
      "step_index": <integer>,
      "step_description": "<copy the strategy text for this step, truncated to 120 chars>",
      "why_critical": "<string>",
      "main_risk": "<string>"
    }
  ]
}
</key_steps>

══════════════════════════════════════════════════════
3. STEREOSELECTIVITY
══════════════════════════════════════════════════════
For each reaction step that creates or destroys stereocenters, assess the stereocontrol. Reference each step by step_index.

<stereoselectivity>
{
  "stereogenic_steps": [
    {
      "step_index": <integer>,
      "step_description": "<strategy text for this step, truncated to 120 chars>",
      "stereocenters_formed": ["<atom description>"],
      "control_mechanism": "<string or null>",
      "assessment": "sufficient | insufficient | missing | not_applicable",
      "comment": "<string>"
    }
  ],
  "overall_stereo_risk": "low | moderate | high",
  "overall_comment": "<string>"
}
</stereoselectivity>

══════════════════════════════════════════════════════
4. HANDOVER POINT
══════════════════════════════════════════════════════
Identify the most complex intermediate (deepest in the tree) at which a template-based retrosynthesis tool could realistically solve the remaining synthesis within ≤ 500 template expansions. Reference it by mol_index.

<handover_point>
{
  "mol_index": "<letter>",
  "depth_from_target": <integer>,
  "justification": "<string>",
  "confidence": "high | moderate | low"
}
</handover_point>

══════════════════════════════════════════════════════
5. CONVERGENCE
══════════════════════════════════════════════════════
Characterise the topology of the route. Identify convergent branch-point molecules (≥2 expanded reactants) by mol_index.

<convergence>
{
  "topology": "linear | convergent | mixed",
  "branch_points": [
    {
      "mol_index": "<letter>",
      "fragments_joined": <integer>,
      "comment": "<string>"
    }
  ],
  "longest_linear_path": <integer>,
  "total_steps": <integer>,
  "assembly_order_optimal": true | false,
  "suggestions": "<string or null>"
}
</convergence>

══════════════════════════════════════════════════════
6. REDUNDANT REACTIONS
══════════════════════════════════════════════════════
Flag any reaction pairs that are redundant or could be merged. Reference each reaction by step_index.

<redundant_reactions>
{
  "redundant_pairs": [
    {
      "step_index_a": <integer>,
      "step_index_b": <integer>,
      "reason": "<string>",
      "suggested_fix": "<string>"
    }
  ],
  "total_redundant": <integer>,
  "comment": "<string>"
}
</redundant_reactions>

══════════════════════════════════════════════════════
7. OPS COMPLETENESS
══════════════════════════════════════════════════════
Inspect the JSON ops list for every reaction step. Flag any add_group or remove_group operations where the group SMILES is chemically implausible as a synthetic reagent — for example:
  - add_group with group="C" (methane) instead of "CCl" for a methylation
  - add_group with group="N" missing a leaving group
  - remove_group whose group does not match a known protecting group or leaving group

For each flagged op, reference the step by step_index and the op by its 0-based index within the ops list.

<ops_completeness>
{
  "flagged_ops": [
    {
      "step_index": <integer>,
      "op_index": <integer>,
      "op_type": "<string>",
      "group_as_given": "<string>",
      "problem": "<string>",
      "suggested_group": "<string>"
    }
  ],
  "total_flagged": <integer>,
  "overall_ops_quality": "good | acceptable | poor"
}
</ops_completeness>

══════════════════════════════════════════════════════
8. REACTION CONDITIONS
══════════════════════════════════════════════════════
For each reaction step, assess the proposed conditions and propose specifics where absent or vague. Reference each step by step_index.

<reaction_conditions>
{
  "steps": [
    {
      "step_index": <integer>,
      "step_description": "<strategy text for this step, truncated to 120 chars>",
      "conditions_as_given": "<string or null>",
      "assessment": "adequate | vague | absent | incompatible",
      "proposed_conditions": "<string or null>",
      "incompatibility_note": "<string or null>"
    }
  ]
}
</reaction_conditions>

══════════════════════════════════════════════════════
9. STARTING MATERIALS
══════════════════════════════════════════════════════
For each terminal building block (leaf node), assess commercial availability and cost. Reference each by mol_index.

<starting_materials>
{
  "building_blocks": [
    {
      "mol_index": "<letter>",
      "availability": "common | speciality | rare | unknown",
      "cost_tier": "cheap (<$10/g) | moderate ($10-100/g) | expensive (>$100/g) | unknown",
      "flag": "<string or null>"
    }
  ],
  "overall_bb_accessibility": "good | moderate | poor"
}
</starting_materials>

══════════════════════════════════════════════════════
10. FEASIBILITY VERDICT
══════════════════════════════════════════════════════
Provide an overall synthetic feasibility rating and the top 3 risks a bench chemist would face.

<feasibility_verdict>
{
  "overall_rating": "excellent | good | acceptable | poor | infeasible",
  "top_risks": [
    {
      "rank": 1,
      "risk": "<string>",
      "mitigation": "<string>"
    },
    {
      "rank": 2,
      "risk": "<string>",
      "mitigation": "<string>"
    },
    {
      "rank": 3,
      "risk": "<string>",
      "mitigation": "<string>"
    }
  ],
  "one_line_summary": "<string — suitable for a supervisor-level report>"
}
</feasibility_verdict>

══════════════════════════════════════════════════════
11. ASYMMETRIC CATALYSIS
══════════════════════════════════════════════════════
Inspect every reaction step for the use of, or suitability for, named asymmetric and stereocontrolled methods. Named methods to look for (non-exhaustive):
  - Noyori asymmetric hydrogenation (Ru-BINAP / Ru-BINAP-diamine)
  - Rh-BINAP or Rh-DIPAMP hydrogenation (Knowles process, Takasago process)
  - Katsuki-Sharpless epoxidation (Ti/tartrate ester, allylic alcohols)
  - Sharpless asymmetric dihydroxylation (OsO4 / AD-mix-α or β)
  - Corey-Bakshi-Shibata (CBS) oxaborolidine-catalysed ketone reduction
  - Jacobsen-Katsuki epoxidation (Mn-salen catalyst)
  - Hajos-Parrish / Hajos-Wiechert reaction (proline-catalysed intramolecular aldol)
  - Shibasaki catalysis (La-BINOL, heterobimetallic complexes)
  - Evans oxazolidinone chiral auxiliary
  - Proline-organocatalysis (List, Macmillan)
  - TADDOL or BINOL-based Lewis acid catalysis
  - Ellman tert-butylsulfinyl imine method
  - Asymmetric Diels-Alder (chiral oxazolidinone dienophile or chiral Lewis acid)

For detected_methods: reference the reaction step by step_index.
For missed_opportunities: reference the molecule where the opportunity exists by mol_index, and the step it would replace by replace_step_index.

<asymmetric_catalysis>
{
  "detected_methods": [
    {
      "step_index": <integer>,
      "step_description": "<strategy text for this step, truncated to 120 chars>",
      "method_name": "<e.g. Noyori hydrogenation, Sharpless AD, CBS reduction>",
      "catalyst_system": "<e.g. (S)-Ru-BINAP, OsO4/AD-mix-β, (R)-CBS/BH3·THF>",
      "confidence": "confirmed | likely | possible",
      "ee_expectation": "<e.g. >95% ee | typically 85-92% ee | unknown>",
      "comment": "<string>"
    }
  ],
  "missed_opportunities": [
    {
      "mol_index": "<letter of the product molecule where the opportunity exists>",
      "replace_step_index": <integer — step_index this method would replace>,
      "stereocenters_affected": ["<description of stereocenters that would be set>"],
      "current_approach": "<what the route currently does at this step, e.g. substrate-controlled aldol, non-stereoselective reduction, stoichiometric chiral auxiliary>",
      "opportunity": "<specific explanation: which bond is formed, substrate class, why catalytic asymmetric control is superior here>",
      "suggested_method": "<named method + catalyst system + brief context, e.g. CBS oxaborolidine reduction of the C7 ketone with (R)-CBS/BH3·THF to give >90% ee>"
    }
  ],
  "asymmetric_catalysis_used": true | false,
  "summary": "<string>"
}
</asymmetric_catalysis>

══════════════════════════════════════════════════════
12. CASCADE REACTIONS
══════════════════════════════════════════════════════
Identify cascade (domino/tandem) reactions proposed in the route or opportunities to collapse multiple steps. Named cascade types to look for (non-exhaustive):
  - Robinson annulation (Michael + intramolecular aldol)
  - Hajos-Wiechert reaction (proline-catalysed Michael + aldol)
  - Cationic polyene / biomimetic cyclisation cascade
  - Diels-Alder / retro-Diels-Alder cascade
  - Aza-Michael / aldol or Mannich cascade
  - Michael / Michael double addition cascade
  - Radical cascade (5-exo-dig, 5-exo-trig, etc.)
  - Wacker / Heck cascade
  - Metathesis cascade
  - Pauson-Khand cascade

For detected_cascades: reference the reaction steps involved by step_indices (list of integers).
For cascade_opportunities: reference the substrate molecule by mol_index.

<cascade_reactions>
{
  "detected_cascades": [
    {
      "step_indices": [<integer>, ...],
      "cascade_type": "<string>",
      "description": "<string — mechanism and bond-forming events>",
      "bonds_formed": <integer>,
      "confidence": "confirmed | likely | possible"
    }
  ],
  "cascade_opportunities": [
    {
      "mol_index": "<letter of substrate molecule where cascade could begin>",
      "opportunity": "<which consecutive steps could be merged and why>",
      "suggested_cascade": "<string>",
      "estimated_step_savings": <integer or null>
    }
  ],
  "cascades_proposed": true | false,
  "summary": "<string>"
}
</cascade_reactions>

══════════════════════════════════════════════════════
13. ROUTE SCORE
══════════════════════════════════════════════════════
Based on ALL twelve dimensions above, provide a high-level executive summary and three integer scores (1–10, where 10 is best):
  - overall: holistic route quality — strategic elegance, step economy, stereocontrol, convergence
  - feasibility: practical bench success likelihood — reagent availability, conditions realism, yield expectations, risk level
  - green: green chemistry quality — atom economy, E-factor expectations, avoidance of toxic/hazardous reagents, solvent sustainability

The summary should be 3–5 sentences readable by a medicinal chemist who has not read the full analysis.

<route_score>
{
  "summary": "<string>",
  "overall": <integer 1–10>,
  "feasibility": <integer 1–10>,
  "green": <integer 1–10>,
  "rationale": "<one sentence explaining the score spread>"
}
</route_score>
"""
