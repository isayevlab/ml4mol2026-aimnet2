# Answer key — Exercise 3: Chemical reactivity

**Instructor and demonstrator copy.** All values from the student notebook unchanged, model `aimnet2-nse`, vertical (fixed-geometry) ionisation at the optimised neutral structure.

## Global descriptors

| Seat | Molecule | *I* / eV | *A* / eV | *ω* / eV | largest f⁺ |
|---|---|---|---|---|---|
| 0 | methyl vinyl ketone | 9.63 | −0.39 | 1.06 | **C4**, terminal CH₂, 0.232 |
| 1 | acrolein | 10.19 | −0.25 | 1.18 | **C0**, terminal CH₂, 0.243 |
| 2 | acrylonitrile | 10.90 | −0.46 | 1.20 | **C0**, terminal CH₂, 0.267 |
| 3 | acetone | 9.79 | −1.96 | 0.65 | **C1**, carbonyl carbon, 0.230 |

**The electrophilicity ordering among the carbonyl systems is chemically correct.** Acetone (0.65), with no conjugated double bond, is much the weakest of the four. Among the conjugated acceptors *ω* rises with the strength of the withdrawing group: methyl vinyl ketone < acrolein < acrylonitrile. Methyl vinyl ketone is the weakest of the three because of its electron-donating methyl group.

Note for the round-up: the activated arenes on seats 4 to 6 have *lower* ω than acetone (aniline 0.49, anisole 0.56, phenol 0.60). That is not a failure. They are nucleophiles, and a low electrophilicity index is the correct statement about them. Do not let the room read the ω column as a single ranking across all eight seats.

**Every conjugated acceptor places the largest f⁺ on the β-carbon**, not the carbonyl carbon: the site of conjugate (1,4) addition, recovered from partial charges alone. Acetone, having no β-carbon, correctly puts it on the carbonyl carbon.

## Aromatic substitution — mean f⁻ by ring position

| Seat | Molecule | ortho | meta | para | classical | agrees |
|---|---|---|---|---|---|---|
| 4 | phenol | 0.113 | 0.054 | 0.159 | ortho/para | **yes** |
| 5 | aniline | 0.098 | 0.036 | 0.139 | ortho/para | **yes** |
| 6 | anisole | 0.089 | 0.053 | 0.147 | ortho/para | **yes** |
| 7 | nitrobenzene | 0.087 | 0.120 | 0.143 | meta | **partly** |

For the three activated rings, ortho and para both exceed meta clearly, which is the textbook activating pattern.

**Seat 7 is the instructive failure and should be handled deliberately.** For nitrobenzene the descriptor does place meta above ortho, capturing part of the deactivating effect, but para comes out highest where the classical answer is meta.

State the reason to the room: condensed Fukui functions describe how the *ground-state* electron density redistributes when an electron is added or removed. Regioselectivity in electrophilic aromatic substitution is governed by the relative energies of the *σ*-complex intermediates, a different quantity. For strongly activated rings the two correlate; for deactivated rings they need not.

The correct conclusion is that this is a **screening** method: fast enough to rank many molecules, reliable for activated systems, with a failure mode that is known and explicable. Where the answer matters, compute the σ-complex energies explicitly using the machinery of notebook 5.

## Common difficulties

| Report | Cause |
|---|---|
| "my electron affinity is negative" | correct and expected in the gas phase; stated in the notebook |
| "`ValueError` about multiplicity" | the ions are doublets, `mult = 2`; this needs `aimnet2-nse` |
| "f⁺ is largest on oxygen" | true for some systems; step 4 restricts the comparison to carbons |
| "nitrobenzene gives para" | expected; see above, and use it |
| "my ω is lower than acetone's" | correct if you have seats 4 to 6; those molecules are nucleophiles |
