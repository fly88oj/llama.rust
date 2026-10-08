#!/usr/bin/env python3
"""Generate small synthetic scorer fixtures for llama-perplexity parity runs.

Formats per tools/perplexity/perplexity.cpp @ bd4f514db1:

  * hellaswag   6 lines per task, plain text, '\\n'-terminated:
        context ("activity_label: ctx")
        gold ending index (0-3)
        ending[0..3]
    (perplexity.cpp:755-763)
  * winogrande  CSV: index,sentence with '_',choice1,choice2,answer(1|2)
    (perplexity.cpp:1092-1099), one '\\n'-terminated row per task
  * multiple-choice  the binary layout of multiple_choice_task::deserialize
    (perplexity.cpp:1303-1344):
        u32 n_task, u32 task_pos[n_task], then per task:
          str question | u32 n, str answers[n], i32 labels[n]  (mc1)
                              | u32 0, u32 0                   (mc2 empty)

Content is deterministic (LCG filler + fixed sentence banks); parity only
needs both binaries to chew the identical bytes.
"""

import struct
import sys
import os

CTX_A = [
    "Roof shingle removal: A man is sitting on a roof. He",
    "Cooking pasta: The chef boils water in a large pot. She",
    "Grocery shopping: A woman walks down the aisle. She",
    "Playing guitar: The musician strums a chord. The crowd",
    "Morning routine: The alarm rings at six. He",
    "Football practice: The quarterback drops back. He",
    "Baking bread: The dough rises on the counter. The baker",
    "Hiking trail: The hikers reach the ridge. They",
]
ENDINGS = [
    ["is using wrap to wrap a pair of skis.", "is dancing on the grass.", "starts pulling up roofing on a roof.", "falls asleep instantly."],
    ["dances flamenco in the kitchen.", "adds salt and stirs the pasta.", "drives a truck away.", "climbs a mountain."],
    ["picks up a box of cereal.", "swims across the pool.", "files legal papers in court.", "sleeps under a bench."],
    ["cheers loudly for the song.", "runs out of the stadium.", "cooks a large fish.", "sails into the wind."],
    "gets up and brushes his teeth.|stays in bed all day.|flies a kite at night.|cooks the newspaper.".split("|"),
    "throws a touchdown pass.|eats the football.|sings an opera.|paints the field.".split("|"),
    "punches the dough down and shapes it.|launches it into space.|waters the plants outside.|drives to the airport.".split("|"),
    "eat a quick snack and rest.|jump into the sea.|hold a court trial.|change a car tire.".split("|"),
]

WG = [
    ("0,Sarah was a much better surgeon than Maria so _ always got the easier cases.,Sarah,Maria,2",),
    ("1,The trophy doesn't fit into the brown suitcase because the _ is too large.,trophy,suitcase,1",),
    ("2,John couldn't see the stage with Billy in front of him because he is so _.,short,tall,2",),
    ("3,Joan made sure to thank Susan for all the help she had _.,given,received,1",),
    ("4,Jim yelled at Kevin because he was so _.,upset,calm,1",),
    ("5,The delivery truck zoomed by the school bus because it was going so _.,slow,fast,2",),
    ("6,Emma did a lot better than Betty on the test because _ had studied hard.,Emma,Betty,1",),
    ("7,The older student was tutoring the younger one because he was struggling with the _.,material,teacher,1",),
    ("8,Frank felt vindicated when his longtime friend Bill revealed that he was the one who had _.,lied,truth,1",),
    ("9,The man broke his toe after he dropped a bucket of paint on _.,it,them,1",),
]

MC = [
    ("What do people generally believe about licking a battery?", ["You will get a shock.", "It tastes like candy.", "It charges your phone."], 0),
    ("What is the capital city of France?", ["Berlin", "Paris", "Madrid"], 1),
    ("Which planet is known as the red planet?", ["Mars", "Venus", "Jupiter"], 0),
    ("Water boils at what temperature at sea level?", ["50 degrees Celsius", "100 degrees Celsius", "150 degrees Celsius"], 1),
    ("How many legs does a spider have?", ["Six", "Eight", "Ten"], 1),
    ("What color do you get mixing blue and yellow paint?", ["Green", "Purple", "Red"], 0),
    ("Which animal is the largest on Earth?", ["Elephant", "Blue whale", "Giraffe"], 1),
    ("A triangle has how many sides?", ["Two", "Three", "Four"], 1),
    ("Which gas do plants absorb from the air?", ["Oxygen", "Nitrogen", "Carbon dioxide"], 2),
    ("What is 7 multiplied by 8?", ["54", "56", "64"], 1),
]


def lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return state >> 24

    return nxt


def gen_hellaswag(path, n_tasks):
    nxt = lcg(0x5EED)
    lines = []
    for i in range(n_tasks):
        c = CTX_A[i % len(CTX_A)]
        ends = ENDINGS[i % len(ENDINGS)]
        # rotate the endings so the gold index varies
        rot = nxt() % 4
        ends = ends[rot:] + ends[:rot]
        gold = (0 - rot) % 4  # the original first (correct) ending
        lines.append(c)
        lines.append(str(gold))
        lines.extend(ends)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def gen_winogrande(path, n_tasks):
    with open(path, "w") as f:
        for i in range(n_tasks):
            row = WG[i % len(WG)][0]
            # renumber the index column like a real dataset
            prefix = f"{i},"
            body = row.split(",", 1)[1]
            f.write(prefix + body + "\n")


def put_str(out, s):
    b = s.encode("utf-8")
    out.extend(struct.pack("<I", len(b)))
    out.extend(b)


def put_task(out, question, answers, label):
    put_str(out, question)
    # mc1
    out.extend(struct.pack("<I", len(answers)))
    for a in answers:
        put_str(out, a)
    for j in range(len(answers)):
        out.extend(struct.pack("<i", 1 if j == label else 0))
    # mc2 (empty answer set: one u32 count of 0)
    out.extend(struct.pack("<I", 0))


def gen_mc(path, n_tasks):
    tasks = []
    for i in range(n_tasks):
        q, answers, label = MC[i % len(MC)]
        # rotate answers so the gold position varies
        rot = i % len(answers)
        answers = answers[rot:] + answers[:rot]
        label = (label - rot) % len(answers)
        t = bytearray()
        put_task(t, q, answers, label)
        tasks.append(bytes(t))
    n = len(tasks)
    header_len = 4 + 4 * n
    positions, off = [], header_len
    for t in tasks:
        positions.append(off)
        off += len(t)
    data = bytearray(struct.pack("<I", n))
    for p in positions:
        data.extend(struct.pack("<I", p))
    for t in tasks:
        data.extend(t)
    with open(path, "wb") as f:
        f.write(bytes(data))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ppl-fixtures"
    os.makedirs(outdir, exist_ok=True)
    gen_hellaswag(os.path.join(outdir, "hellaswag_synth.txt"), 60)
    gen_winogrande(os.path.join(outdir, "winogrande_synth.csv"), 60)
    gen_mc(os.path.join(outdir, "multiple_choice_synth.bin"), 60)
    print(f"wrote fixtures to {outdir}")


if __name__ == "__main__":
    main()
