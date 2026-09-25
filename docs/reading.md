# Reading list

Read in this order. After each paper, write three lines under "My notes" in your own words:
the claim, the evidence, and what it means for this project. Those notes are what the
weekly check in will quiz you on.

IDs marked (verify) were written from memory; confirm them on arXiv before citing.

## Block 1: the vision precedent (your thesis ground)
1. Tramèr and Boneh 2019, Adversarial Training and Robustness for Multiple Perturbations. arXiv 1904.13000 (verify)
   Why: the original evidence that robustness to one norm does not transfer to others.
   My notes:
2. Croce and Hein 2022, robustness against multiple and single lp threat models via quick fine tuning. arXiv 2105.12508 (verify)
   Why: cheap ways to get union robustness; the baseline your multi norm method competes with.
   My notes:

## Block 2: how LLM safety training fails
3. Wei, Haghtalab and Steinhardt 2023, Jailbroken: How Does LLM Safety Training Fail? arXiv 2307.02483 (verify)
   Why: names "mismatched generalization", the theory behind H1.
   My notes:
4. Yong, Menghini and Bach 2023, Low Resource Languages Jailbreak GPT-4. arXiv 2310.02446 (verify)
   Why: the direct precedent for the Swahili family.
   My notes:
5. Andriushchenko and Flammarion 2024, Does Refusal Training in LLMs Generalize to the Past Tense? arXiv 2407.11969
   Why: a small rewording breaks refusal training; the paraphrase family precedent.
   My notes:
6. Li et al. 2024, LLM Defenses Are Not Robust to Multi Turn Human Jailbreaks Yet. arXiv 2408.15221 (verify)
   Why: defenses that hold single turn collapse multi turn; the strongest prior evidence for H1.
   My notes:

## Block 3: training defenses and measurement
7. Mazeika et al. 2024, HarmBench. arXiv 2402.04249 (verify)
   Why: our harmful request source, judge, and the R2D2 adversarial training baseline.
   My notes:
8. Dabas et al. 2026, Adversarial Deja Vu (ASCoT), ICLR 2026. arXiv 2510.21910
   Why: the closest competitor; you must be able to say exactly how this project differs.
   My notes:
9. Rottger et al. 2024, XSTest. arXiv 2308.01263 (verify)
   Why: measures over refusal, our main cost metric (D3).
   My notes:

## Block 4: mechanism (read last)
10. Arditi et al. 2024, Refusal in Language Models Is Mediated by a Single Direction. arXiv 2406.11717 (verify)
    Why: the basis for the later mechanism chapter (H3).
    My notes:
