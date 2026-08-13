# CHECK 12/13/14: representation-level sanity checks (129 fold-0 test molecules)

- Input: ..\gradient12_descriptor_check\descriptors_all_129.csv

- gradient12 n=12, certain47 n=47, abs_error mean: g12=0.63 vs c47=0.12

- p-values uncorrected, two-sided.


## CHECK 12 - tautomer / protonation-state ambiguity

Model input has NO bond-order or formal-charge feature (element one-hot + 3D only); tautomer/charge ambiguity can only act through the geometry (H count, bond lengths) produced by RDKit ETKDG/MMFF from the SMILES.

- taut_amide_urea_lactam: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- taut_imine: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- taut_enol_OH: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- taut_1,3_dicarbonyl: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- taut_guanidine: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- taut_amidine: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- taut_aromatic_pyrrole_NH: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- taut_phenol_OH: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- taut_alpha_CH2_to_carbonyl: g12 1/12 (0.08) vs c47 8/47 (0.17) | Fisher p=0.6697
- ion_carboxylic_acid: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- ion_carboxylate: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- ion_amine_primary_secondary: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- ion_amine_tertiary: g12 1/12 (0.08) vs c47 1/47 (0.02) | Fisher p=0.3682
- ion_sulfonic_acid: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- ion_nitro: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000

**Verdict CHECK 12:** ruled out (no flags differ p=0.50) 


## CHECK 13 - unspecified stereochemistry

- undefined stereo centers: g12 mean=0.00 vs c47 mean=0.00 | Mann-Whitney p=nan | Spearman vs abs_error rho=nan (p=nan), vs signed rho=nan (p=nan)
- any undefined stereocenter: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000

**Verdict CHECK 13:** ruled out (stereo undefined p=nan)


## CHECK 14 - sanitization edge cases + featurizer trace

Featurizer trace (from element_vocab.py build_one_hot + DimeNet++ usage): atom feature = one-hot over 17 elements (H,C,N,O,F,P,S,Cl,Li,B,Na,Mg,Si,K,Ca,Br,I); edge features from 3D distances/angles only. NO formal charge, NO bond order, NO hybridization, NO aromaticity is in the model input. Pipeline: SMILES -> element gate ({H,C,N,O,F,P,S,Cl,Br,I}) -> AddHs -> ETKDGv3/MMFF (or xTB) -> atNUM+atXYZ -> one-hot+geometry.

- sanitize_warnings: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- kekulize_failed: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- radicals: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- has_charged_atoms: g12 0/12 (0.00) vs c47 1/47 (0.02) | Fisher p=1.0000
- non_model_element: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000
- non_vocab_element: g12 0/12 (0.00) vs c47 0/47 (0.00) | Fisher p=1.0000

Charged-atom encoding in input SMILES (representation loss: charge invisible to model):
- mobley_7415647 [other] N:1;O:-1  CN(C)C(=O)c1ccc(cc1)[N+](=O)[O-]
- mobley_7176290 [other] N:1;O:-1  c1cc(cc(c1)O)[N+](=O)[O-]
- mobley_2725215 [other] N:1;O:-1;N:1;O:-1  CCCN(CCC)c1c(cc(cc1[N+](=O)[O-])S(=O)(=O)C)[N+](=O)[O-]
- mobley_1922649 [other] N:1;O:-1  COP(=S)(OC)Oc1ccc(cc1)[N+](=O)[O-]
- mobley_9741965 [other] N:1;O:-1;N:1;O:-1  C[C@@H](CCO[N+](=O)[O-])O[N+](=O)[O-]
- mobley_3802803 [certain47] N:1;O:-1  CCCCCCO[N+](=O)[O-]
- mobley_2481002 [other] N:1;O:-1  C([N+](=O)[O-])(Cl)(Cl)Cl

**Verdict CHECK 14:** model input is element+geometry only; no unknown-token fallback exists (one-hot index lookup is total over the 17-element vocab). Anomaly flags: charged-atom SMILES g12=0 vs c47=1; non-model elements 0. ruled out: no group-representation difference detected
Note: every charged-atom hit above is the RDKit canonical nitro representation
[N+](=O)[O-] - a charge-separated notation for a neutral NO2 group, NOT a
real ionized species. After charge neutralization all 129 SMILES are neutral;
no protonation-state ambiguity is encoded in any input.
