# Contract-to-model gate

Checked 7 September 2026, before evaluating order-book bounds. This gate authorises a conditional terminal-index illustration under A. It does not establish a strict-local-martingale model of traded Bitcoin spot. B is an additional hypothetical convention and requires separate observable consistency checks.

| Item | Contract evidence and implementation |
|---|---|
| Product | BTC linear USDC European options; catalogue must say BTC base, USDC quote and settlement, linear type, multiplier one, `btc_usdc` index. |
| Gross payoff | One BTC-equivalent call pays `(I_T−K)+` USDC and put pays `(K−I_T)+` USDC. ITM exercise creates a future at strike which immediately cash settles. |
| Settlement reference | I_T is the 07:30–08:00 UTC index TWAP at expiry. It is not an instantaneous traded spot price. |
| Premium | Paid upfront in full for a standard-margin long. Quoted per BTC; multiplier one. |
| Currency basis | The exchange assumes USD/USDC parity for this index; real USDC exposure and collateral valuation can differ. |
| Fees | Displayed prices and payoffs here are gross, excluding trading and delivery fees. The study is not an executable net-profit or arbitrage calculation. |
| Futures | Corresponding future mark supplies the exchange’s option forward input. Its value is saved separately from index S0 and is never imposed as E[S_T]. |

These contract facts are documented in [Deribit’s linear USDC option specifications](https://support.deribit.com/hc/en-us/articles/31424932728093-Linear-USDC-Options), including the settlement-process change, premium timing and BTC multiplier. Raw instrument specifications additionally confirm the fields used in acquisition.

## Maintained modelling assumptions

Set S0 to the mean of the reference-index observations bracketing each expiry batch. Treat the settlement average as the terminal nonnegative variable. Maintain a common pricing measure for the retained put quotes and the restriction D E[I_T] ≤ S0. This is an assumption, not a consequence of a nonnegative distribution. A is discounted expected put-payoff pricing. B additionally stipulates a full-defect call premium.

Define X = D I_T/S0, k = DK/S0 and d = 1−E[X]. Under A, p(k)=E[(k−X)+]. Under A and B, c(k)=E[(X−k)+]+d=1−k+p(k). The put ask restriction is d≤min(1,min_i(1−k_i+p_i,ask)). Under B, call asks additionally give d≤min_i c_i,ask. Minima range over all usable selected asks; the highest strike need not minimise a noisy ask sequence. A negative put expression contradicts the model and must not be clipped.

The absence of a model of the index average and an economic funding curve prevents an unconditional spot-defect interpretation. The pilot therefore measures compatibility of an explicitly approximated terminal-index model. Neither exchange margining nor a parity check proves B. The averaged reference is not silently substituted for the theorem’s traded asset.

## Discount, carry and timing

The baseline is r=0, D=exp(−rT), with T measured on ACT/365 from the batch start. Before quotes were evaluated, conditional annual rate scenarios −2%, 0%, 5%, 10% and S0 shifts −0.1%, 0%, +0.1% were saved in configuration. These provide a bounded sensitivity exercise without claiming a known risk-free USDC curve. They are neither confidence ranges nor an estimated market discount curve. Do not choose a fitted discount rate merely because it makes a preferred defect feasible.

Every initial expiry batch targets at most 30 seconds and every utilised quote side needs positive displayed size. Books older than 30 seconds at retrieval, clocks over two seconds into the future, crossed records and unsupported contracts are excluded. Batch reference movement above 0.1% fails the batch gate. Individual ages and observed index extrema remain in the output: a passing gate does not establish perfect synchronisation. A separate ±0.1% reference exercise is much larger than the observed initial within-batch moves. Recollection checks the bound-driving instruments with new references; it does not refresh the full-chain witness.

## Gate outcomes for the delivered run

At D=1, September and October each admit checked terminal laws under A. December’s negative put restriction contradicts A plus the nonnegative-defect restriction at that input. All three expiries fail baseline AB parity; those combined numerical restrictions are not identification intervals. The positive-rate sensitivity columns are necessary bounds only: no new compatible model is claimed from the sign of a bound.

Discount-factor intersections diagnose whether any common D could reconcile pairwise parity at the specified S0. They are stored separately and never fed back into the witness fit. This separates a possible carry mismatch from an inference about the defect. The baseline put-only branch is retained for two expiries; the inconsistent branches stop at diagnosis.
