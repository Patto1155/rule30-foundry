# Proposed direction: eliminate period words for every onset

This is a research proposal, not a proved new result or a literature-novelty
claim. It follows the output-loop work without assuming the alternating case
will synchronize.

## The change in question

Instead of asking how far a proposed period survives in the actual center
prefix, ask which periodic **words** force an adjacent column to become
periodic under the local rule, regardless of transient length or exterior.

The existing adjacent-column obstruction for the single seed would then
exclude each certified word at every possible onset. That is a different
quantifier gain from extending a prefix scan. Constant words are the initial
positive controls; `01` is a negative control through the tested widths.

## A bounded first experiment

Enumerate primitive binary period words of lengths 1..6, keeping one rotation
representative of each. Build the periodically forced strip graph at widths
3, 5, 7, 9. Admit a word to the exclusion set only when an exhaustive recurrent-
component check proves that the left output eventually has the word's period.
This is a sufficient test; rejecting a word does not establish its possibility
on the seed. Save replayable finite graphs or equivalent checking inputs.

For survivors, the two-loop test can distinguish an actual obstruction to
arbitrary-period synchronization from failure of the same-period test alone.
No positive discovery is guaranteed: every nonconstant word may survive.

## The creative extension

Minimize the accepted certificates to discover which parts of the period word
erase boundary uncertainty. Look for a reusable block or transition rule that
lets certificates compose, rather than merely accumulating excluded words.
A proof that a whole language of periodic words forces neighbor periodicity
would exclude an infinite family of hypothetical center tails in one argument.

Any such composition law needs its own proof. A small table is not an infinite
family, repeating the same word is not a new minimal period, and a reset at one
phase does not automatically control the neighbor at every phase. These are the
conditions the first certificates should expose explicitly.
