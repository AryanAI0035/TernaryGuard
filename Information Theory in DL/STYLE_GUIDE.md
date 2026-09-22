# Style Guide for "The Information-Theoretic Foundations of Deep Learning"

Every chapter must follow a strict three-register structure in this exact order:

## 1. Descriptive
Name the phenomenon precisely. Provide intuition, context, and a clear explanation of what is being discussed before introducing the heavy math. The tone should be authoritative yet accessible.

## 2. Mathematical
Real theorems, derivations, and proofs. **No hand-waving.** Use precise notation (as defined in `notation.tex`). Everything here must be mathematically rigorous and definitively known. State assumptions clearly.

## 3. Horizon
Every chapter **must** end with a Horizon box. Use the `\begin{horizon}` ... `\end{horizon}` environment.

- **Length**: 2-4 paragraphs.
- **Content**: State what's proven, what's conjectured, and what would need to be true to close the gap.
- **Tone**: Explicitly conjectural language. Use phrases like: "It is conjectured that...", "An open question is...", "Future work might show...". 
- **Strict Boundary**: Never let Horizon content leak into the main proofs. Never present a conjecture with the same confidence as a proved theorem.
