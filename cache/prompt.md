
# Common instruction

You are an evaluator of Digital Therapeutic Alliance (DTA) in text-based supportive dialogue.

Use ONLY the provided frozen DTA-Rubric v1.0.

The unit of evaluation is the dialogue window. Rate only observable alliance-relevant behaviour displayed by the supporter.

Bond, Goal, and Task must be evaluated independently.

Valid ratings are:

1, 2, 3, 4, 5, or "NA"

Do not use half-points.

If genuinely uncertain between two adjacent scores, select the lower score.

Do not use information outside the dialogue window.

Return valid JSON only.

---

# P1 — Zero-Shot Rubric Baseline

Using the frozen DTA-Rubric, independently rate the following dialogue window on:

* Bond
* Goal
* Task

Apply the rubric exactly as written.

Dialogue window:

{dialogue}

Return only:

{
"bond": 1-5 or "NA",
"goal": 1-5 or "NA",
"task": 1-5 or "NA"
}

---

# P2 — Structured Rubric-Guided Evaluation

Using the frozen DTA-Rubric, evaluate the following dialogue window systematically.

For each dimension:

1. Determine whether the dimension is scorable under the rubric.
2. Identify the observable supporter behaviour relevant to that dimension.
3. Match the observed behaviour to the closest rubric score anchor.
4. Apply any binding N/A rule or adjudicated edge case.
5. Assign one final rating.

Important:

* Evaluate Bond, Goal, and Task independently.
* Do not reward response length, fluency, or general pleasantness.
* For Task, apply the Task N/A decision rule before selecting a numerical score.
* If uncertain between adjacent scores, select the lower score.

Dialogue window:

{dialogue}

Return only:

{
"bond": 1-5 or "NA",
"goal": 1-5 or "NA",
"task": 1-5 or "NA"
}

---

# P3 — Few-Shot Rubric-Guided Evaluation

You are evaluating Digital Therapeutic Alliance (DTA) using the frozen DTA-Rubric v1.0.

Bond, Goal, and Task must be rated independently.

Valid ratings are:

1, 2, 3, 4, 5, or "NA"

The following adjudicated calibration examples demonstrate how the frozen rubric is applied.

## Example 1 — Low scores across dimensions

Dialogue context:

Supporter: I have found that being in isolation lowers my immunity, and if I want to be healthy, fresh air, loving friends, exercise, good healthy food is the way to do it, not isolation and loneliness. That affect me emotionally and I don't want to despair, that makes me feel terrible. It doesn't feel healthy, and we know no one that has died either. But if you are not going to go out, do you have goals to help you not feel lonely, any ideas? Any thoughts on how you can fix this so you don't feel so lonely?

Seeker: you are absolutely right. i just try to divert my mind on entertainment and cooking . my work burden increased to a greater extent.

Supporter: You might even want to call your friends and colleagues to see if any of them are meeting up. We are social creatures. We need to socialize.

Seeker: i can't afford much time.

Seeker: considering the present situation, its bad for all of us to get socialized.

Response to evaluate:

I'm sure you are busy, but realize, your emotional health is important too and we always need to make time to connect with people.

Gold rating:

{
"bond": 1,
"goal": 1,
"task": 1
}

---

## Example 2 — Dimensions can differ substantially

Dialogue context:

Therapist: Yeah. Well, I'm so glad that you reached out for help, um, and just know that you're not alone. And-and so, yeah, is that okay if I share a little bit about, um, how I tend to work with clients on that topic?

Client: Yes, please.

Therapist: Yeah. Well, um, first of all, I really am passionate about looking at the whole picture, um, physical health and emotional wellbeing and what I found in working with clients in the past is when, um, we make a focus of weight loss, sometimes, uh, physical health improves, but at the cost of emotional wellbeing.

Client: Mm-hmm.

Therapist: Um, many people find that when then- when they are restricting their intake or starting some, uh, ac-activity patterns that are really rigorous, that, um, a piece of their mental health suffers because they start to obsess, um, more than they'd like to be about those things. So, my approach with clients is just a more holistic approach. Looking at certainly some behavior changes that you could make to be more physically active, improve your energy levels.

Response to evaluate:

Um, and also look at activities that enhance your mood. We can also take a look at some dietary patterns and see if-if there are ways you could tweak how you're fueling your body throughout the day, but at the same time, sort of letting go of what happens on the scale. Letting go of, um, the final, you know, weight outcome of that, and really focusing instead on healing your relationship with food and also adopting sustainable physical activity patterns. What do you make of all that?

Gold rating:

{
"bond": 2,
"goal": 4,
"task": 4
}

---

## Example 3 — High Bond with Task not applicable

Dialogue context:

Seeker: Yea i suppose your right. Btw yes i love to soak in the tub with a bath bomb and a good book. Yea no one really seems to understand what us mothers go through on a daily basis. I'm glad it's not just me who sees it.

Supporter: Well there you go. Go run you a hot bath. Toss in a bath bomb, and grab your favorite novel or new novel and soak your troubles away!

Seeker: Your right. He won't hurt them. Not with what is at risk. I'mma enjoy myself till they come back tonight. Thank you so much!

Supporter: Of course sweetie, just remember to take care of you. There is still a woman under the mother. A lot of moms forget that and neglect themselves.

Seeker: Yea after so long you just go with I'm a mom as your identity. I'm still a woman. I'm actually starting to talk to a new guy it's kinda exciting.

Response to evaluate:

Oh that's exciting! You get to start dating again! I didn't look forward it myself mainly because im jaded and have little faith in men XD. What do you look for in a man?

Gold rating:

{
"bond": 5,
"goal": 4,
"task": "NA"
}

---

Now evaluate the NEW dialogue independently.

Important:

* Use the examples only to understand how the scoring scale is applied.
* Do not assign scores based on superficial similarity to an example.
* Apply the frozen DTA-Rubric itself.
* Evaluate Bond, Goal, and Task independently.
* Apply the Task N/A decision rule exactly as specified in the rubric.
* Do not reward response length or fluency.
* If genuinely uncertain between two adjacent scores, choose the lower score.

New dialogue:

{dialogue}

Return valid JSON only:

# P4 — Dimension-Wise Independent Evaluation

Using the frozen DTA-Rubric, evaluate the dialogue one dimension at a time.

## Bond

Consider ONLY Bond-related behaviour:

* accurate emotional understanding
* acceptance and non-judgment
* situational personalisation
* willingness to remain with distress

Do not use Goal or Task quality when assigning Bond.

## Goal

Consider ONLY Goal-related behaviour:

* goal elicitation
* understanding checks
* directional alignment
* persistence toward the seeker's expressed goal

Do not use Bond or Task quality when assigning Goal.

## Task

First apply the Task N/A decision rule.

If Task is scorable, consider ONLY:

* actionability
* rationale
* collaborative framing
* timing

Do not use Bond or Goal quality when assigning Task.

Apply all binding rubric anchors and adjudicated edge cases.

If uncertain between adjacent scores, select the lower score.

Dialogue window:

{dialogue}

Return only:

{
"bond": 1-5 or "NA",
"goal": 1-5 or "NA",
"task": 1-5 or "NA"
}
