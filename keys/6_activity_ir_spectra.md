# Answer key — Exercise 2: Infrared spectra

**Instructor and demonstrator copy.** Every figure below was produced by running the student notebook unchanged, model `aimnet2`, `fmax = 0.02`, on a CPU. All frequencies are **unscaled** unless stated. Intensities are relative to the strongest band of the same molecule.

## Results by seat

The "strongest band" column is what `np.argmax(intensities)` returns, which is what the student's Step 5 and Step 7 print. For six of the eight molecules it is the band a chemist would name first. The two exceptions, seats 2 and 7, are the alcohols, and the reason is worth five minutes of the room's time (see the caveat below).

| Seat | Molecule | modes | strongest band | next strongest | experiment for the strongest |
|---|---|---|---|---|---|
| 0 | water | 3 | **1597** bend | antisymmetric stretch 3942 (0.38) | bend 1595 |
| 1 | formaldehyde | 6 | **1816** C=O stretch | symmetric CH stretch 2954 (0.51) | C=O 1746 |
| 2 | methanol | 12 | **1071** C–O stretch | O–H torsion 205 (0.85), O–H stretch 3918 (0.49) | C–O 1033 |
| 3 | acetone | 24 | **1820** C=O stretch | CH₃ deformation 1388 (0.40), C–C stretch 1221 (0.32) | C=O 1731 |
| 4 | acetonitrile | 12 | **2398** C≡N stretch | CH₃ deformation 1430 (0.48, degenerate pair) | C≡N 2267 |
| 5 | chloroform | 9 | **782** C–Cl stretch (degenerate pair) | C–H bend 1223 (0.16) | C–Cl 668–774 |
| 6 | benzene | 30 | **743** out-of-plane C–H bend | C–H stretch 3214 (0.30, degenerate pair) | 673 |
| 7 | ethanol | 21 | **282** O–H torsion | C–O stretch 1083 (0.91), CH stretch 3116 (0.63) | see the caveat |

## The three results every seat should reach

**Bending and skeletal modes below about 1800 cm⁻¹ agree closely with experiment.** Water's bend is 1597 cm⁻¹ against a measured 1595; chloroform's C–Cl stretch is 782 against 774; benzene's out-of-plane bend is 743 against 673, the largest miss in this group.

**Stretches of stiff bonds are overestimated.** X–H stretches come out 3 to 5 percent high: water's 3791 and 3942 against 3657 and 3756. Multiple bonds are overestimated by a similar amount: acetone's C=O at 1820 against 1731, acetonitrile's C≡N at 2398 against 2267. This is the harmonic approximation, not the model. A real bond potential is shallower than a parabola at large displacement. The conventional 0.96 scaling brings water's stretches to 3639 and 3785, and acetone's C=O to 1747.

**Intensity follows the dipole derivative, not the amplitude.** The polar bonds win: C=O, C≡N, C–Cl and C–O all give the strongest band of their molecule, while the C–H stretches, which move the most atoms the furthest, are weak everywhere. Formaldehyde makes the point in one spectrum: its three CH₂ bending modes at 1198, 1231 and 1535 have intensities of 0.04 to 0.10 next to the C=O stretch at 1.00. Benzene makes it with symmetry: of 30 modes only seven have any intensity (743; the degenerate pairs at 1071, 1532 and 3214), which are exactly the four infrared-active fundamentals experiment sees at 673, 1038, 1486 and 3068. A vibration that does not change the dipole does not absorb.

## A caveat to have ready

Seats 2 and 7 will see a very strong band at very low frequency: 205 cm⁻¹ for methanol (0.85) and 282 cm⁻¹ for ethanol (1.00, the strongest band in the whole spectrum). These are hydroxyl torsions. They are real modes with a genuinely large dipole derivative, but they are also the modes least well described by the harmonic approximation, since the torsional potential is nowhere near quadratic. Seat 7's Step 7 answer will name the torsion as the strongest band; that is what the code returns, and the right response is "correct, and here is why that number is not a prediction of an observable band position". The chemically informative answer for ethanol is the C–O stretch at 1083 (0.91), for methanol the C–O stretch at 1071.

## Common difficulties

| Report | Cause |
|---|---|
| "I have an imaginary frequency" | geometry not converged; the notebook asserts on this |
| "a mode has zero intensity" | correct; this is the selection rule, so ask which atoms move |
| "all my stretches are too high" | correct and expected; this is anharmonicity |
| "my strongest band is at 282 cm⁻¹" | ethanol, seat 7; the hydroxyl torsion, see the caveat |
| "3N−6 does not match my count" | a linear molecule has 3N−5; the projection detects this automatically and the printed count says which was used |
