validator_prompt = """You are an expert organic chemist, together with your colleague, tasked with developing a retrosynthesis strategy for a target molecule with some constraints provided by a user. Your colleague has proposed a synthesis plan, and your task is to evaluate the plan based on the chemical feasibility of the listed reactions, and whether this synthesis plan satisfies the constraints provided by the user.

Here is the target molecule for which you need to develop a synthesis strategy:

<target_molecule>
{{TARGET_MOLECULE}}
</target_molecule>

The user has provided the following strategy or constraint for this synthesis:

<user_strategy>
{{USER_STRATEGY}}
</user_strategy>

Your colleague has proposed the following synthesis route, given as a fully rendered operations plan. Read it as follows:

* `target` is the target molecule (atom-mapped).
* `steps[]` are the retrosynthetic disconnections in order (the first step disconnects the target). Each step has:
  * `desc` / `step_rationale` / `reaction_conditions` — the transformation your colleague *intends*, described in the forward direction. This is the stated intent of what the step is meant to accomplish.
  * `ops` — the exact, deterministic graph edits (bond/atom/group changes that reference atom-map ids) encoding the disconnection.
  * `rendered.frontier` — the precursor molecules the step actually produces (atom-mapped), each flagged `in_stock` (true = a purchasable building block) and `is_leaf` (true = a terminal precursor not disconnected further). `solved` is true when every frontier molecule is in stock.
* `route[]` at the bottom lists each step's reaction as `product>>reactants` (retrosynthetic direction, atom maps stripped).

The `ops` and the resulting `rendered.frontier` are deterministic — together they are the faithful, non-hallucinated record of what each step *actually does*. Use them as the source of truth; do not infer the transformation from the `desc` alone. Your job includes checking that the intent in `desc` is actually realized by the ops and frontier. Where it is not — for example the `desc` names a reagent or transformation that the frontier does not contain, or a precursor that should be a simple reagent is instead a complex non-purchasable leaf — point out that mismatch as a flaw. For organometallic additions (Grignard, organolithium, etc.) judge the freed reagent fragment by whether it carries a *handle* to become a nucleophile, not by whether the metal is literally drawn: an inert alkane with no handle (e.g. ethane `[CH3][CH3]` for an intended ethyl addition) is a flaw, but an organohalide precursor *is acceptable* (e.g. ethyl bromide `CC[Br]` standing in for EtMgBr, since the Grignard is generated in situ from the halide + Mg). Only flag the reagent when the fragment has no handle at all and its C–H is not acidic enough to deprotonate directly (a terminal alkyne, which can form an acetylide, is fine).

<rendered_route>
{{RENDERED_ROUTE}}
</rendered_route>

An atom and stereo accounting for every step, computed with RDKit, is supplied below.

<atom_stereo_accounting>
{{ATOM_ACCOUNTING}}
</atom_stereo_accounting>

ATOM CONSERVATION: those numbers are authoritative — do NOT recount, contradict, or
hand-wave them. Your job is to judge whether each reported imbalance is CHEMICALLY
JUSTIFIED by the reaction being described, and to say so in `balance_check`.

* SURPLUS in precursors (atoms lost going forward) is normal: leaving groups (Br, I, OMs,
  OTs), condensation byproducts (H2O, ROH), a protecting group cleaved off. Accept it when
  the step's stated intent accounts for it.
* MISSING from precursors (atoms the product has and the precursors lack) is the dangerous
  direction — those atoms must come from somewhere. Accept ONLY when a named reagent
  plausibly supplies them: O from an oxidant (mCPBA, OsO4, ozone), halogen from NBS/NCS,
  C1 from formaldehyde or CO, and similar. If the missing atoms amount to a carbon fragment
  that the step expects a *precursor* to provide, that is a dropped reactant — flag the
  step as blocking.
* A missing whole fragment often shows up as several elements missing at once.
* A drop in assigned stereocentres, or in stereo double bonds, means the ops flattened
  stereochemistry the product carries. Judge whether the step intends that; a silently
  lost centre is a flaw even when the atom counts balance.


Now evaluate the route. For each reaction in the sequence, work through the non-exhaustive checklist below (points 1-10), then assess the route as a whole (points 11-14). The route is retrosynthetic, but reason in the forward direction (reactants -> products) where it helps — especially when judging thermodynamic feasibility, selectivity, conformation, and downstream reactivity.

1. The reaction encoded by the `ops` and resulting `rendered.frontier` matches the intent stated in `desc` (see the note above on flagging intent-vs-frontier mismatches — e.g. an organometallic addition whose reagent is an inert alkane with no handle; note that an organohalide precursor like ethyl bromide is acceptable for a Grignard).
2. The absolute stereochemistry of centers that are not involved in the reaction must remain unchanged.
3. Identify the key functional groups and structural changes involved, and consider possible side reactions or alternative outcomes.
4. The reaction product resulting from the reaction in the forward direction is a structurally stable molecule, whose formation is thermodynamically favored. Thermodynamically favored means both from an enthalpic (energetic) point of view and from an entropic point of view. Here are some examples.
    * If the forward reaction forms a trans fused 4,4 ring system, such a reaction is highly unfavored because the system formed is highly strained, and thus disfavored from an energetic point of view.
    * If the forward reaction breaks a 5-membered ring in order to form an 8-membered ring, this reaction is highly unfavored from an entropic point of view.
5. The reaction proposed must be selective. Selectivity covers chemoselectivity (which functional group reacts), regioselectivity (which position/site reacts), stereoselectivity (which diastereomer forms), and enantioselectivity (which enantiomer forms). If selectivity is violated, say which kind is violated and explicitly name the other possible outcome(s) the reaction would also give (the competing functional group, regioisomer, diastereomer, or enantiomer). Here are some example of unselective reactions, that are not acceptable:
    * A Grignard addition on a substrate that has an ester and a ketone is not selective, as the Grignard will react with both the ester and the ketone. Such a reaction is not selective and not acceptable.
    * If a reaction takes place through an enolization, but there are multiple enolization sites with the same acidity as the site that reacts, then the reaction will not be selective and is not acceptable.
6. The substrate must find itself in the right conformation for certain reactions to happen. Some reactions may happen just because the reacting centers are in proximity to one another. In the same time, some common reactions may not happen because the reacting centers are too far from one another, or do not adapt the orientation needed for the corresponding transition state. Here are some examples:
    * If a Diels Alder reaction is proposed and the diene is fixed in an s-trans conformation, then the reaction will not take place. Such a reaction is therefore wrong and not acceptable
7. Consider the case when the reaction forms in the forward direction a reaction product that is stable from a structural point of view, but unstable from a reactivity point of view, due to a subsequent reaction that may happen intramolecularly or, alternatively, a polymerization. Let us denote the analysed reaction as R. If the next reaction in the forward direction (equivalent to the previous reaction in the retro direction) directly makes use of the reactivity that causes the instability, then reaction R is acceptable from the point of view of this criterion. If the next reaction in the forward direction (equivalent to the previous reaction in the retro direction) does not make use of the reactivity that causes the instability, then reaction R is not acceptable. Here are some examples.
    * A reaction forms in the forward direction an unsubstituted cyclobutadiene. This molecule is highly unstable from a reactivity point of view  and will dimerize. If the next reaction in the forward direction (equivalent to the previous reaction in the retro direction) dimerizes the cyclobutadiene, then the reaction that had the cyclobutadiene as its reaction product is acceptable. If this is not the case, then the reaction that had the cyclobutadiene as its reaction product is not acceptable.
    * A reaction forms in the forward direction an (E, E)-1,5-cyclononadiene. This molecule is highly unstable from a reactivity point of view as there is an intramolecular Cope rearrangement that will happen. If the next reaction in the forward direction (equivalent to the previous reaction in the retro direction) makes use of the Cope rearrangement, then the reaction that had as its reaction product the (E, E)-1,5-cyclononadiene  is acceptable. If this is not the case, then the reaction that had as its reaction product the (E, E)-1,5-cyclononadiene is not acceptable
8. If the reaction proceeds through an ionic mechanism, make sure that in the reactants the atom that initiates the attack is nucleophilic and the atom that receives the attack is electrophilic.
9. A single step should encode a single reaction. Flag a step that is too succinct — i.e. it compresses what are really two or more distinct reactions into one (for example several independent bond-forming/breaking events that would not all occur in a single operation, or a multi-stage cascade that a chemist would carry out and isolate as separate steps).
10. After confirming the points above, evaluate the reaction on any other criteria you feel are important, and note any other problems you see.
11. Evaluate how well the reaction aligns with the user strategy's requirements — its contribution to the overall plan, and whether it renders the user strategy impossible to achieve.
12. How well does the overall sequence align with the user strategy's goals?
13. Are there any major discrepancies or missing steps?
14. Are there any unnecessary or overly complex steps?

Give your analysis in the following format:

<synthesis_analysis>
Your analysis here, identifying the problematic steps and the overall quality of the proposed synthesis in terms of feasibility and relevance to the user strategy.
</synthesis_analysis>

Report the problems you find: for each problematic step, be specific about *what* is wrong and *why* (the chemical reason). The editor decides how to address them. Focus only on the steps that are genuinely problematic.

Mark the route's status with `is_satisfactory`: set it to true when the route is good enough to carry forward — i.e. it has no blocking problems — even if you still note optional `suggestion`-severity observations; set it to false when at least one step has a blocking problem. For each problematic step, set `severity` to `blocking` (a real problem that makes the route unacceptable as written) or `suggestion` (a minor, non-blocking observation). If the synthesis plan is high quality and meets the user's needs, set `is_satisfactory` to true.

<feedback>
{
    "is_satisfactory": <true if the route has no blocking problems and is good enough to stop editing, false otherwise>,
    "overall_feedback": <A short summary of the route's status>,
    "balance_check": <Quote the MISSING and SURPLUS lines from the accounting block for each step, then state for each whether it is chemically justified by that reaction (naming the reagent or leaving group) or is a dropped/invented fragment. End with PASS or FAIL.>,
    "problematic_steps": [
        {
            "step_id": <Step id from the rendered route, e.g. "S3">,
            "severity": <"blocking" or "suggestion">,
            "feedback": <The problem with this step — what is wrong and why>
        }
        // ... more problematic steps if needed
    ]
}
</feedback>

Remember, the reactions shown are theoretical and have not been tested in a laboratory. They represent desired transformations but may not necessarily reflect what would actually occur in a flask. Your expertise is crucial in assessing the feasibility and relevance of these proposed reactions."""
