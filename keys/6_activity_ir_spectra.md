# Answer key — Exercise 2: Infrared spectra

**Instructor and demonstrator copy.** Every figure below was produced by running the student notebook unchanged, model `aimnet2`, `fmax = 0.02`, on a CPU. All frequencies are **unscaled** unless stated.

## Results by seat

The "strongest band" column is what `np.argmax(intensities)` returns, which is what the student's Step 5 and Step 7 print. **Read it before the session**: for five of the eight molecules the strongest band is *not* the functional-group band a chemist would name first, and that surprise is the most useful thing in this exercise.

| Seat | Molecule | modes | strongest band | the band they expected |
|---|---|---|---|---|
| 0 | water | 3 | **1597** bend | the bend; exp. 1595 |
| 1 | formaldehyde | 6 | **1535** CH₂ scissor | C=O, which is present but weaker |
| 2 | methanol | 12 | **3918** O–H stretch | C–O at 1071 (intensity 0.38) |
| 3 | acetone | 24 | **1388** CH₃ deformation | C=O at 1820 (intensity 0.48) |
| 4 | acetonitrile | 12 | **1411** CH₃ deformation | C≡N at 2398 (intensity 0.66) |
| 5 | chloroform | 9 | **1223** C–H bend (degenerate pair) | C–Cl at 697 and 782 (0.07, 0.05) |
| 6 | benzene | 30 | **743** out-of-plane C–H bend | correct; most other modes are inactive |
| 7 | ethanol | 21 | **3858** O–H stretch | C–O at 1083 (intensity 0.55) |

## The three results every seat should reach

**Bending and skeletal modes below about 1800 cm⁻¹ agree closely with experiment.** Water's bend is 1597 cm⁻¹ against a measured 1595.

**X–H stretches above 2800 cm⁻¹ are overestimated by 3 to 5 percent.** This is the harmonic approximation, not the model: a real X–H potential is shallower than a parabola at large displacement. The conventional 0.96 scaling brings water's stretches from 3791 and 3942 to 3639 and 3785, against measured 3657 and 3756.

**Intensity and chemical importance are different things.** A strong band is one with a large ∂μ/∂Q, not one belonging to an interesting functional group. Seats 3 and 4 are the clearest case: acetone's C=O and acetonitrile's C≡N are the bands a chemist would assign first and use for identification, and both are genuinely present at sensible positions, but a methyl deformation carries more intensity in the harmonic double-harmonic treatment used here. Draw this out explicitly rather than letting seats 3 and 4 think they made a mistake.

**Seats 1 and 6 illustrate the selection rule.** Formaldehyde has one mode of essentially zero intensity and benzene has many: a vibration that does not change the molecular dipole does not absorb, regardless of its amplitude.

## A caveat to have ready

Seats 2 and 7 will see a second strong band at very low frequency: 205 cm⁻¹ for methanol and 282 cm⁻¹ for ethanol, both around 0.8 relative intensity. These are hydroxyl torsions. They are real modes with a genuinely large dipole derivative, but they are also the modes least well described by the harmonic approximation, since the torsional potential is nowhere near quadratic. Do not let a student report them as a prediction of an observable band position.

## Common difficulties

| Report | Cause |
|---|---|
| "I have an imaginary frequency" | geometry not converged; the notebook asserts on this |
| "a mode has zero intensity" | correct; this is the selection rule, so ask which atoms move |
| "all my stretches are too high" | correct and expected; this is anharmonicity |
| "the strongest band is not the C=O" | correct; see the third point above |
| "3N−6 does not match my count" | a linear molecule has 3N−5; the projection detects this automatically and the printed count says which was used |
