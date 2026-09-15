# Answer key — Exercise 1: Conformational analysis

**Instructor and demonstrator copy.** Values obtained by running the student notebook unchanged: three ETKDG seeds, `fmax = 0.02`, model `aimnet2`, on a CPU. The flexible systems (seats 1, 2, 3, 5, 6, 7) depend on the random seeds, so conformer counts vary by a few between runs. The **energy ordering** and the **size of the gaps** are the reproducible quantities, and are what the exercise asks for.

## Results by seat

| Seat | System | conformers | lowest gap / kcal mol⁻¹ | lowest population | key result |
|---|---|---|---|---|---|
| 0 | butane | 2 | 0.33 | 64% | anti at 180°, gauche at −64°; anti lower |
| 1 | methylcyclohexane | several | 1.48 (chair–chair) | — | equatorial below axial; compare with the A-value, 1.74. The rest of the list is twist-boats, 5 kcal/mol up |
| 2 | ethylene glycol | ~5 | 0.36 | 63% | lowest two conformers hydrogen bond at 2.30 Å; the rest sit near 3.9 Å |
| 3 | glycine | ~3 | 1.16 | 87% | contacts 2.78, 2.60, 2.34 Å: the two motifs compete and the shortest contact is not the lowest conformer |
| 4 | 1,2-dichloroethane | 2 | ~1.0 | ~85% | anti lower, by more than in butane |
| 5 | 1,4-butanediol | ~20 | 0.40 | 31% | **only the lowest conformer closes the O–H···O bond, at 2.22 Å; every other one is 4.6 Å or more** |
| 6 | ibuprofen | ~13–18 | small | ~15% | no single structure describes the molecule |
| 7 | alanine dipeptide | tens | — | ~77% | in the gas phase the extended beta / C5 basin near φ = −150° dominates |

## Points to draw out

**Seats 0 and 4 should compare directly.** Butane and 1,2-dichloroethane pose the same question with different substituents. Both prefer anti; chlorine gives the larger gap.

**Seat 5 is the best single result in this exercise.** 1,4-butanediol has five rotatable bonds, and exactly one of its twenty-odd conformers folds far enough to bring the two hydroxyls together. That conformer is the global minimum and holds about a third of the population; every other conformer leaves the oxygens more than 4.5 Å apart. The hydrogen bond is worth roughly 0.4 kcal/mol here, which is the whole reason the folded form wins. Put the seat 5 table on the screen.

**Seats 5 and 6 show the practical consequence.** Beyond about four rotatable bonds no single structure represents the molecule, and any property computed from one optimised geometry is unreliable.

**Every system forming an internal hydrogen bond overstates that preference**, because the calculation is in the gas phase and water would compete for the same donor. This is the limitation demonstrated with glycine in notebook 1.

**On what the hydrogen-bond column measures.** It reports the shortest H···(N,O) distance where the hydrogen is bonded to nitrogen or oxygen *and* is at least four bonds away from the acceptor through the molecular graph. Both conditions are needed. Without the first, any O–CH₂ group gives a geminal O···H–C contact at 2.0 Å in every conformer; without the second, a –COOH group gives a 2.3 Å O···H contact across three bonds, again in every conformer. Either would produce a number that looks like a hydrogen bond and never changes. A student who asks why the code is not simply `min(d[O][H])` has asked a good question.

## Common difficulties

| Report | Cause |
|---|---|
| "only two conformers" | correct for butane and dichloroethane; a rigid molecule has few minima |
| "nothing converged" | `fmax` tightened below 0.02 |
| "my conformer count differs" | expected for the flexible systems; compare the energy ordering instead |
| "my hydrogen-bond distance is `nan`" | the molecule has no polar hydrogen at all; check the seat number |
