# How rewards, the policy, and guardrails fit together

_Last updated: 2026-05-31._

This document explains the system as **three cooperating parts** — the **reward**, the
**policy**, and the **guardrails** — and **why the numbers in the code are set the way they
are**. It assumes you have *not* worked with reinforcement learning or this codebase before,
so it defines each term the first time it appears. Every claim points at the code that backs
it, in case you want to dig in.

It complements three existing docs:
- [selfless_routing_analysis.md](selfless_routing_analysis.md) — *why* we chose this
  objective and traffic setup (the research story).
- [rl_pipeline_workflow_and_guards.md](rl_pipeline_workflow_and_guards.md) — how a vehicle
  flows through the code at runtime.
- [coordination_throttle.md](coordination_throttle.md) — the deep dive on the Layer A/B
  guardrail.

This doc is the **glue**: how the three parts hand information to each other, in plain terms.

---

## 0. The problem, in one paragraph

We control a fleet of cars driving on a map of New York City inside a traffic simulator
(SUMO). At certain points each car must decide **which way to go** to reach its destination.
We want to *train* a controller that makes those choices well — getting cars to their
destinations quickly, without creating traffic jams, and ideally being a little "selfless"
(occasionally taking a slower route so the whole fleet moves better). The system that learns
this has three parts, described next.

---

## 1. The three parts, in plain language

Think of training a new driver:

1. **The reward** is the *grade* the driver gets after each decision. A good outcome (made
   progress, reached the destination, avoided a jam) earns points; a bad outcome (got stuck,
   went in a circle, caused congestion) loses points. The reward exists **only during
   training** — it is how the system knows what "good" means. It is not used when the trained
   controller is later deployed.

2. **The policy** is the *driver's brain* — the thing that actually looks at the current
   situation and picks a route. It is a small neural network. We use an algorithm called
   **MAPPO** (Multi-Agent Proximal Policy Optimization) to adjust that network so its choices
   earn higher rewards over time. The policy is used **both during training and after
   deployment**.

3. **The guardrails** are the *driving instructor's hand on the wheel* — fixed, hand-written
   rules that remove obviously bad options *before* the policy chooses, and override clearly
   self-defeating choices *after* it chooses. They are not learned; they are common-sense
   safety rails. Most run in **both training and deployment**.

| Part | Where in the code | Job | Active when |
|---|---|---|---|
| **Reward** | [`compute_reward`](../core/rl_training_pipeline.py#L1942), [`compute_pending_step_reward`](../core/rl_training_pipeline.py#L2058) | Grade each decision so training knows what "good" is | Training only |
| **Policy** | [`MAPPOActor` / `MAPPOCritic`](../core/mappo.py#L150) + [`RouteCandidateGenerator`](../core/route_candidate_generator.py) | Look at the situation and pick a route | Training + deployment |
| **Guardrails** | [`SharedDecisionPolicy`](../core/shared_decision_policy.py), the "pending" state machine, [`coordination_throttle.py`](../core/coordination_throttle.py) | Strip out bad options before/after the policy | Training + deployment (one piece is deployment-only) |

**The single idea that makes them work together:**

> The guardrails decide *which routes are even allowed*, the policy picks among the allowed
> routes, and the reward teaches the policy which picks were good. All three speak the **same
> language** — they measure everything in **seconds of travel time** and **how crowded a road
> is** — so a trade-off made by one part is understood by the others without any made-up
> conversion factor.

The rest of this document is that sentence, explained.

---

## 2. The one design choice everything else depends on: the policy picks a *route*, not a *turn*

This is the most important thing to understand first.

Instead of asking the policy "turn left or right at this corner?", we do something cleverer.
A separate component, the [`RouteCandidateGenerator`](../core/route_candidate_generator.py),
proposes **up to 4 complete routes** from where the car is to its destination. The policy's
only job is to **pick one of those 4**. (`route_k = 4`, set
[here](../core/rl_training_pipeline.py#L291).)

- **Route #0 is always the normal shortest route.** Routes #1–#3 are alternative detours.
- "Took a detour" simply means the policy picked something other than #0.

Each of the 4 candidate routes comes with a small **scorecard** — a list of 11 numbers
describing it ([`ROUTE_FEATURE_DIM = 11`](../core/route_candidate_generator.py#L28), built in
[`_compute_features`](../core/route_candidate_generator.py#L467)). The numbers that matter for
this doc:

| # | what it measures | who reads it |
|---|---|---|
| 2 | how crowded this route is **on average** | policy, reward |
| **3** | how crowded its **worst** road is | policy, **guardrail** |
| **4** | how crowded its **first** road is | policy, **guardrail** |
| **7** | how many **extra seconds** this route costs vs. the shortest one (the "sacrifice") | policy, **reward** |
| **9** | how much **less crowded** this route is than the shortest one, on average (the "relief") | policy, reward, guardrail |
| **10** | the same "relief," measured on the first road only | policy, reward, guardrail |

Why this design is the foundation: because all three parts read from **the same scorecard**,
they are automatically talking about the same thing. When the reward decides a detour was
worth it, when the policy scores it, and when a guardrail checks it, they are all looking at
the **same numbers for the same route**. For example, the reward and the guardrail both
compute "relief" as `0.65 × number 9 + 0.35 × number 10` — the *identical* formula, on
purpose ([reward side](../core/rl_training_pipeline.py#L455),
[guardrail side](../core/coordination_throttle.py#L52)). That shared vocabulary is the glue.

(Technical aside, skippable: the policy network scores each of the 4 routes with the same
small sub-network and turns those scores into preferences. Routes that aren't valid right now
are given a hugely negative score so they're never picked —
[`_masked_logits`](../core/mappo.py#L88). This "score then mask" step is how the guardrails
hand their filtered list to the policy.)

(Performance aside, skippable: building the 4 candidates is the single most expensive
per-decision step — it runs several Dijkstra searches and re-scores each candidate against
live density. Two **behavior-preserving** optimizations keep it cheap, both inside
[`get_candidates`](../core/route_candidate_generator.py#L113):

- **Per-call density memo.** Within one call the live edge density is frozen — the Layer B
  reservation field is seeded only *after* the route is chosen — yet the same edge is read
  hundreds of times (by the density-aware Dijkstra cost functions on every relaxation, and
  again when route metrics and the diversity selection re-score paths). The raw density
  lookup is therefore cached for the duration of the call. Because it caches a value that is
  constant during the call, the scorecards come out **bit-for-bit identical**; it only drops
  redundant reads (~8× fewer density lookups in practice).
- **Scalar clamps.** The per-edge `np.clip(...)` / `np.max(...)` on tiny (1–40 element)
  lists are now Python `min` / `max`. For finite scalars these are identical to the NumPy
  versions but skip array-dispatch overhead. The `np.mean` and weighted `np.average`
  reductions are deliberately **left as NumPy** — its pairwise (and float32) summation is
  not reproduced by a naive Python sum, so swapping them would shift low-order bits and is
  *not* a safe no-op.

Together these cut `get_candidates` to roughly a quarter of its former wall-cost with
outputs verified bit-identical via a feature-hash check. The same scalar-clamp substitution
is applied to the corridor/guardrail density stats in
[`action_corridor_stats`](../core/shared_decision_policy.py#L452) and the per-step feature
normalizers, again only where it is provably exact.)

---

## 3. The reward: how a decision is graded, and why each number

Every reward number is set in one place
([lines 189–248](../core/rl_training_pipeline.py#L189)). The code states its own priorities
([here](../core/rl_training_pipeline.py#L212)): **(1) save travel time first, (2) avoid
adding to congestion second, (3) prefer the shorter path only as a tie-breaker.**

### 3.1 The backbone: a small "time is ticking" cost — `0.07` per step

Every moment a car is still driving, it loses `0.07` points per simulation step
([code](../core/rl_training_pipeline.py#L1981)). This is the **base unit of the whole reward
system** — the "currency." Everything else is priced relative to it. A car that takes ~300
steps to finish loses about 21 points of "time cost" along the way, which nudges the policy to
**finish quickly**. When you see other reward numbers below, read them as "how big is this
compared to the 0.07/step time cost?"

### 3.2 Congestion costs: deliberately small

- A tiny cost for being on a crowded road.
- A cost (`0.04` per unit) for being on a road that is **more crowded than the network
  average** — note this only penalizes *adding to a hotspot*, not the unavoidable background
  traffic.
- A slightly smaller cost (`0.015`) for the congestion your presence imposes on *others* (an
  "externality").

These are all **much smaller than the 0.07 time cost** on purpose: congestion-awareness
should *bend* the chosen route, never override the goal of arriving.

### 3.3 The "selflessness" dial: a shared cost measured in seconds

This is the cooperative part. Normally each car only cares about its own travel time. We can
optionally make every car also feel a **share of the whole fleet's delay**, so that helping
others actually improves its own grade.

- The knob is [`team_reward_alpha`](../core/rl_training_pipeline.py#L153), a number from 0 to 1.
  **0 = purely selfish** (only your own time matters). **1 = the fleet's delay counts as much
  as your own.** This single dial is what we slide to compare "selfish" vs. "selfless"
  behavior in experiments.
- How big is that shared cost? Controlled by
  [`team_reward_scale = 0.12`](../core/rl_training_pipeline.py#L220), chosen to be **about 1.7×
  the 0.07/step personal time cost**. The reasoning ([code comment](../core/rl_training_pipeline.py#L217)):
  when the fleet is badly jammed, this makes the shared pain roughly the same size as a car's
  own time cost — so relieving a jam is worth about as much as saving your own time. (A more
  aggressive experiment, "arm C," raises this to 0.30.)
- **A subtle but important refinement** ([code](../core/rl_training_pipeline.py#L1924)): instead
  of charging every car the *same* fleet-wide delay (which is mostly caused by random traffic
  no single car controls, and just adds noise), we charge each car its **own** slowness
  *relative to the fleet average*. A car stuck in traffic pays more; a car that found a clear
  road is rewarded. This is called a "difference reward," and it dramatically sharpens the
  learning signal.

**Why this is the keystone:** this shared cost is multiplied by elapsed time, so it is
measured in **seconds** — the very same unit as the personal time cost. That means the policy
can directly weigh "this detour costs *me* 10 extra seconds" against "it saves the *fleet* 15
seconds" with **no arbitrary conversion factor**. An earlier version of the code tried to
trade "seconds" against an abstract, unit-less "congestion relief" number, which never worked
because there was no principled exchange rate between the two (the full story is in
[selfless_routing_analysis.md §2.2](selfless_routing_analysis.md)).

### 3.4 The per-decision selfless nudge, and its clever "refund"

There is an older, more direct selflessness reward,
[`_route_candidate_balance_components`](../core/rl_training_pipeline.py#L427). It reads the same
route scorecard from §2: number 7 (the extra seconds you'd sacrifice) and the "relief" blend
(numbers 9 and 10, how much congestion you'd relieve). If a detour relieves enough congestion
to be worthwhile, it gets a bonus.

Its cleverest detail is the **refund** ([code](../core/rl_training_pipeline.py#L480)): a car
that takes a *worthwhile* selfless detour would normally still lose points for the extra time
the detour costs. So the reward **adds those points back** — it refunds the detour's time cost
— making a good selfless detour roughly break-even instead of a guaranteed loss. This only
works because the refund uses the *same* `0.07`/step in the *same* seconds unit as the time
cost it's cancelling. (The "arm C" experiment turns this older reward off so the cleaner §3.3
dial is the only driver.)

### 3.5 The big, rare rewards: reaching the goal, and disasters

These are the sparse, high-stakes outcomes:

| outcome | points | meaning |
|---|---|---|
| reached destination | **+50** | the goal; far bigger than the ~21 of time cost, so finishing always wins |
| "teleport" (SUMO's last resort when a car is hopelessly stuck) | **−40** | being stuck is nearly as bad as never arriving |
| route choice that strands the car (no way to continue) | −12, ends the trip | a fatal route mistake |
| went in a loop / U-turn / route rejected | −1.5 to −6 | the loop-guardrail's reward-side echo (see §5.1) |
| trip running very long (past 1.35× its expected time) | a growing penalty | discourages dawdling |

To keep these from swamping the math, all step-by-step rewards are **capped to the range
−20…+20**; the cap is only widened at arrival so the +50 goal reward isn't chopped down
([code](../core/rl_training_pipeline.py#L226)).

> **The cautionary example — why you can't tune one part in isolation:** the "trip running
> very long" penalty is *correct* for the goal of saving time, but it is **exactly what a
> helpful detour triggers** — a selfless car takes longer, and gets penalized for it. So when
> we want selfless behavior (arm C), we deliberately switch that penalty off. This is the
> clearest illustration that the reward terms interact, and a number that's right for one
> objective can fight another.

### 3.6 The quiet helper that makes the small signals survive

One last reward-related piece, in plumbing terms. From one episode to the next, the total
points a car earns can swing by ~25× — mostly because of random traffic, not because of the
policy. That huge swing makes part of the learning math (the "critic," explained in §4)
unstable. So we **rescale** the critic's learning target to a steady size
([`RunningMeanStd`](../core/mappo.py#L11)), while keeping the actual rewards the policy learns
from in real, honest seconds. Without this rescaling, the small but meaningful selflessness
signal from §3.3 would be drowned out by the noise.

---

## 4. The policy: the learning "brain," explained gently

The policy is a neural network trained with **MAPPO**. Two roles inside it:

- The **actor** is the part that *chooses* — it scores the 4 candidate routes and picks one.
- The **critic** is a *coach* used only during training — it estimates "how good is this
  situation, roughly?" so the system can tell whether a choice did better or worse than
  expected. (In the jargon, the critic predicts the "value"; "did better than expected" is the
  "advantage.") The critic is thrown away at deployment; only the actor drives.

A useful design fact: the critic is allowed to "peek" at a fleet-wide summary of the whole
situation during training ([18 numbers](../core/rl_training_pipeline.py#L296)), which makes its
estimates steadier, while the actor only ever sees one car's local view so it can run
independently per car after deployment. (This split is a standard recipe called
**centralized training, decentralized execution**.) Importantly, the coach makes *learning*
more stable but does **not** change *what* we're optimizing — the cooperative incentive has to
come from the reward (§3.3).

The training settings, in plain terms ([`MAPPOConfig`](../core/mappo.py#L97)):

| setting | value | what it does, plainly |
|---|---|---|
| `gamma` | **0.995** | how far ahead the policy "cares." Near 1 because trips are long (~2000 steps), so a choice must be credited for traffic it causes far in the future |
| `clip_epsilon` | 0.20 | a safety limit so each training update only nudges the policy, never lurches |
| `target_kl` | 0.015 | stop an update early if the policy is changing too much at once |
| `entropy_coef` | **0.15 → 0.05** | how much to encourage *trying new things*. High early (explore detours), low later (settle on what works) — it fades over training ([code](../core/mappo.py#L273)) |
| `actor_lr` / `critic_lr` | 0.0003 / 0.001 | learning speeds; the coach learns faster because its job is simpler |

Two finer points worth knowing:
- **Credit is kept per-car.** When the system works out which choices led to which outcomes,
  it does so along *one car's* journey at a time, and excludes the forced, mechanical
  micro-maneuvers so they don't muddy the credit for real route choices
  ([`_compute_gae`](../core/mappo.py#L451)).
- **Greedy vs. random deployment.** Once trained, the policy can be run two ways
  ([`_act_route`](../controller/MAPPOController.py#L1535)): always pick its single
  top-scored route ("greedy," repeatable), or pick randomly in proportion to its preferences
  ("stochastic," which naturally spreads the fleet across alternatives). The default is greedy;
  the trade-offs are documented in
  [selfless_routing_analysis.md §5.5](selfless_routing_analysis.md).

---

## 5. The guardrails: the instructor's hand on the wheel

Guardrails exist so the **reward doesn't have to teach basic safety**. If we tried to teach
"don't drive in circles" purely through penalties, those penalties would have to be so large
they'd drown out the subtle routing signal we actually care about. Instead, we just **take the
bad options off the table** and let the reward stay focused on travel time. The reward-side
penalties in §3.5 are a backup, not the main defense.

There are three kinds.

### 5.1 Before the policy chooses: filter out bad routes

[`policy_action_candidates`](../core/shared_decision_policy.py#L751) is a series of filters
that prune the candidate list *before* the policy scores it:
- drop moves that would create a short loop or drive into a dead end;
- drop moves that lead straight into a known jam;
- drop a much-longer "relief" detour unless it is genuinely safe and clearly helpful;
- drop a congested branch when a comparable, less-congested one exists.

Where a filter can't cleanly decide, the **reward provides a matching nudge** (small penalties
for loops, congestion pressure, etc.), so the filter and the penalty work as a pair.

### 5.2 The "pending" state machine: don't grade the car for the wrong thing

When a car is mid-maneuver, the code tracks a "pending" state (full detail in
[rl_pipeline_workflow_and_guards.md](rl_pipeline_workflow_and_guards.md#active-vs-passive-pending)).
The key idea in plain terms: it distinguishes **"healthily waiting"** (e.g. queued at a red
light — perfectly fine) from **"genuinely stuck"** (a stale decision going nowhere). This
matters because it stops the reward from punishing a car for normal waiting, while still
penalizing a real stall (−18 points). The −18 is sized to be clearly worse than ordinary
driving but not as catastrophic as a teleport (−40).

### 5.3 After the policy chooses: the saturation throttle (Layer A + Layer B)

This is the guardrail that most directly closes the loop with the policy, and it solves a
specific failure: at very high congestion, the policy's "selfless" detours can all pile onto
the *same* alternative road and make things worse. (Full design:
[coordination_throttle.md](coordination_throttle.md).)

- **Layer A — the veto** ([`detour_should_fallback`](../core/coordination_throttle.py#L74)):
  *after* the policy picks a detour, if that detour's alternative road is itself near capacity
  **and** the whole network is saturated, the choice is reverted to the normal shortest route
  (#0). The clever part: it judges the alternative by its **absolute crowdedness** (scorecard
  numbers 3 and 4), *not* by the "relief" number the policy reacted to — because when
  everything is jammed, a road can look like "relief" (slightly less jammed than the baseline)
  while still having no actual room left. Layer A catches exactly the blind spot the relief
  signal has. It runs **only at deployment**, because overriding the car's route during
  training would confuse the learning math (it would grade one route while the policy thought
  it chose another).
- **Layer B — the reservation field** ([`ReservationField`](../core/coordination_throttle.py#L121)):
  when a car commits to a route, its upcoming roads are "booked" in a quietly fading ledger.
  The candidate generator then treats a booked road as *slightly more crowded than it currently
  looks*, so the **next** car deciding in the same instant sees that the alternative is already
  filling up — and is less likely to pile on. This turns a simultaneous stampede into an
  orderly, one-at-a-time spreading. Crucially, this booking only affects what the candidates
  *look like*; it never touches the reward or the honest traffic measurements, so the learning
  signal stays truthful. It runs in **both** training and deployment.

How they reinforce each other: Layer A catches *"that road is already full,"* and Layer B
prevents *"that road is about to be full because we're all about to choose it."* The
thresholds are set conservatively so the throttle does **nothing** in normal congestion (where
the policy's detours genuinely help) and only steps in at true saturation, where the plain
shortest route is the right answer anyway.

---

## 6. How the parts hand signals to each other (the summary)

The value is in these hand-offs, not in any single part:

1. **One unit: seconds.** The time cost, the selflessness dial, the detour refund, and the
   progress bonus are all in seconds, so trade-offs need no conversion factor. *(§3.1, §3.3, §3.4)*
2. **One scorecard.** The "relief" and "crowdedness" numbers are computed once and read by the
   policy, the reward, and the guardrail — same numbers, three readers. *(§2, §3.4, §5.3)*
3. **Guardrails feed the policy.** The filtered route list *is* the menu the policy chooses
   from. *(§4, §5.1)*
4. **Reward backs up the guardrails.** Where a filter can't cleanly decide, a matching penalty
   discourages the same behavior. *(§5.1, §5.2)*
5. **Layer A sees what the reward can't.** Absolute crowdedness catches the saturated-road case
   the relief-based signal misses. *(§3.4 vs. §5.3)*
6. **Layer B keeps learning honest.** It nudges only the candidate scorecards, never the reward
   or the real measurements. *(§5.3)*
7. **The rescaler protects the small signals.** Without it, the subtle selflessness signal
   would be lost in the noise of random traffic. *(§3.6)*
8. **One dial, one objective.** "Selfish" and "selfless" are the same travel-time goal at
   different fleet weights, which is why the whole stack works unchanged as we turn the dial.
   *(§1, §3.3)*

---

## 7. Why the numbers are sized the way they are (cheat-sheet)

| number | value | sized against |
|---|---|---|
| per-step time cost | 0.07/step | the "currency"; everything else is priced relative to it |
| selflessness scale | 0.12 (aggressive: 0.30) | ≈1.7× the time cost, so a jammed fleet hurts about as much as your own delay |
| congestion (above-average) cost | 0.04 | below 0.07 → congestion is secondary to travel time |
| externality cost | 0.015 | below 0.04 → imposing on others is tertiary |
| selfless-detour thresholds | 0.004 / 0.06 | matched to the tiny crowding differences (0.002–0.02) that are real in light NYC traffic |
| reaching the goal | +50 | bigger than the ~21 of time cost, so finishing always dominates |
| teleport (stuck) | −40 | ~0.8× of missing the goal — being stuck is nearly that bad |
| stale-stall penalty | −18 | worse than normal driving, milder than a teleport |
| reward cap | ±20 (goal: +55) | keep the learning math bounded; widened so +50 isn't chopped |
| "alternative is full" threshold (Layer A) | 0.50 | absolute near-capacity; below it the veto never fires |
| "network is saturated" threshold | 0.30 | the congestion level where the plain shortest route is already best |
| how far ahead the policy cares (`gamma`) | 0.995 | trips are ~2000 steps; delay must trace back ~70+ steps |
| explore-vs-exploit (`entropy`) | 0.15 → 0.05 | try detours early, settle later |

---

## 8. Where to look in the code

| Topic | File |
|---|---|
| Reward numbers & formulas | [`core/rl_training_pipeline.py`](../core/rl_training_pipeline.py) (lines 189–248, 1942, 2058, 1902) |
| The learning brain (actor/critic, rescaler) | [`core/mappo.py`](../core/mappo.py) |
| The 4 candidate routes & their scorecards | [`core/route_candidate_generator.py`](../core/route_candidate_generator.py) |
| Pre-policy filters & the pending state | [`core/shared_decision_policy.py`](../core/shared_decision_policy.py), [`core/junction_decision_engine.py`](../core/junction_decision_engine.py) |
| The saturation throttle (Layer A/B) | [`core/coordination_throttle.py`](../core/coordination_throttle.py) |
| Deployment: choose → veto → drive → book | [`controller/MAPPOController.py`](../controller/MAPPOController.py#L1350) |
| The research "why" | [selfless_routing_analysis.md](selfless_routing_analysis.md) |
| Runtime flow & guardrails | [rl_pipeline_workflow_and_guards.md](rl_pipeline_workflow_and_guards.md) |
| Throttle deep dive | [coordination_throttle.md](coordination_throttle.md) |
</content>
</invoke>
