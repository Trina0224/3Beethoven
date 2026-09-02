"""3Beethoven easy baseline diagnostic v0.1.

Frozen before distillation. Do not include these questions in training data.
Assumes `model` and `tokenizer` are already loaded in the notebook.
"""

import re
import pandas as pd
import torch

BENCHMARK = [
    ("composer_period", "Which composer belongs primarily to the Baroque period?", ["A. Franz Schubert", "B. Johann Sebastian Bach", "C. Gustav Mahler", "D. Claude Debussy"], "B"),
    ("composer_period", "Which composer is most closely associated with the transition from the Classical to Romantic era?", ["A. Beethoven", "B. Monteverdi", "C. Vivaldi", "D. Ravel"], "A"),
    ("composer_period", "Which composer was a major representative of French Impressionism?", ["A. Brahms", "B. Handel", "C. Debussy", "D. Haydn"], "C"),
    ("composer_period", "Which composer was born earliest?", ["A. Mozart", "B. Beethoven", "C. Bach", "D. Brahms"], "C"),
    ("composer_period", "Which composer is most strongly associated with late-Romantic symphonies of enormous scale?", ["A. Mahler", "B. Corelli", "C. Scarlatti", "D. Couperin"], "A"),
    ("form_structure", "In a traditional fugue, what is the opening presentation of the main theme called?", ["A. Subject", "B. Cadenza", "C. Trio", "D. Recitative"], "A"),
    ("form_structure", "Which section of sonata form normally brings the main themes back in the tonic key?", ["A. Development", "B. Recapitulation", "C. Coda", "D. Introduction"], "B"),
    ("form_structure", "A theme followed by a series of transformed versions of that theme is called:", ["A. Theme and variations", "B. Fugue", "C. Recitative", "D. Through-composed form"], "A"),
    ("form_structure", "Which form typically alternates a recurring principal theme with contrasting episodes?", ["A. Rondo", "B. Passacaglia", "C. Motet", "D. Canon"], "A"),
    ("form_structure", "In a concerto, the virtuosic solo passage traditionally occurring near the end of a movement is called:", ["A. Cadenza", "B. Exposition", "C. Chorale", "D. Ostinato"], "A"),
    ("instrumentation", "Which instrument is NOT normally part of the standard woodwind family?", ["A. Oboe", "B. Bassoon", "C. Clarinet", "D. French horn"], "D"),
    ("instrumentation", "Which instrument commonly doubles the cello line one octave lower in orchestral writing?", ["A. Viola", "B. Double bass", "C. Bassoon", "D. Harp"], "B"),
    ("instrumentation", "Which instrument is a double-reed woodwind?", ["A. Flute", "B. Clarinet", "C. Oboe", "D. Horn"], "C"),
    ("instrumentation", "Which orchestral instrument normally has the highest standard register?", ["A. Piccolo", "B. Bass clarinet", "C. Bassoon", "D. Tuba"], "A"),
    ("instrumentation", "Which keyboard instrument is especially associated with Baroque continuo playing?", ["A. Synthesizer", "B. Harpsichord", "C. Celesta", "D. Accordion"], "B"),
    ("history", "Who employed Joseph Haydn for much of his career?", ["A. The Medici family", "B. The Esterházy family", "C. The Habsburg emperor exclusively", "D. The French royal court"], "B"),
    ("history", "Which composer famously spent much of his career in Leipzig as Thomaskantor?", ["A. Bach", "B. Chopin", "C. Wagner", "D. Berlioz"], "A"),
    ("history", "Which composer was strongly associated with the development of the nineteenth-century symphonic poem?", ["A. Liszt", "B. Palestrina", "C. Purcell", "D. Telemann"], "A"),
    ("history", "Which composer wrote Swan Lake?", ["A. Tchaikovsky", "B. Mendelssohn", "C. Schumann", "D. Dvořák"], "A"),
    ("history", "Which composer caused a famous scandal with the 1913 premiere of The Rite of Spring?", ["A. Stravinsky", "B. Mozart", "C. Handel", "D. Brahms"], "A"),
    ("style", "Dense contrapuntal writing and elaborate fugues are especially characteristic of:", ["A. J. S. Bach", "B. Chopin", "C. Puccini", "D. Debussy"], "A"),
    ("style", "Which composer is particularly associated with nocturnes, mazurkas, and highly idiomatic piano writing?", ["A. Chopin", "B. Wagner", "C. Haydn", "D. Monteverdi"], "A"),
    ("style", "Leitmotifs used extensively across large-scale music dramas are most strongly associated with:", ["A. Wagner", "B. Vivaldi", "C. Scarlatti", "D. Clementi"], "A"),
    ("style", "Whole-tone scales, parallel chords, and unusual orchestral color are strongly associated with:", ["A. Debussy", "B. Handel", "C. Haydn", "D. Corelli"], "A"),
    ("style", "Which composer is especially known for combining Hungarian folk influences with twentieth-century modernism?", ["A. Bartók", "B. Rossini", "C. Couperin", "D. Pergolesi"], "A"),
    ("terminology", "What does 'pizzicato' instruct a string player to do?", ["A. Play very loudly", "B. Pluck the strings", "C. Use the mute", "D. Play near the bridge"], "B"),
    ("terminology", "What does 'crescendo' mean?", ["A. Gradually louder", "B. Gradually slower", "C. Suddenly soft", "D. Detached articulation"], "A"),
    ("terminology", "What is an ostinato?", ["A. A repeated musical pattern", "B. A free solo passage", "C. A change of key", "D. A type of cadence"], "A"),
    ("terminology", "What does 'rubato' generally refer to?", ["A. Flexible expressive timing", "B. Playing without vibrato", "C. Strict mechanical tempo", "D. Plucking the strings"], "A"),
    ("terminology", "A sequence in music is:", ["A. Repetition of a musical idea at successively different pitch levels", "B. A completely improvised cadenza", "C. A sudden change in instrumentation", "D. A repeated percussion rhythm only"], "A"),
]


def ask_student(category, question, choices):
    prompt = f"""Answer this classical music multiple-choice question.

Question:
{question}

{chr(10).join(choices)}

Reply with ONLY the letter A, B, C, or D.
Do not explain.
"""
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    match = re.search(r"\b([ABCD])\b", generated.upper())
    return match.group(1) if match else "INVALID"


rows = []
for idx, (category, question, choices, expected) in enumerate(BENCHMARK, 1):
    predicted = ask_student(category, question, choices)
    correct = predicted == expected
    rows.append({"id": idx, "category": category, "question": question, "expected": expected, "predicted": predicted, "correct": correct})
    print(f"{idx:02d}/{len(BENCHMARK)} {'✅' if correct else '❌'} {category} expected={expected} got={predicted}")

df = pd.DataFrame(rows)
print(f"\nOverall accuracy: {df['correct'].mean():.1%}")
print("\nAccuracy by category:")
print(df.groupby("category")["correct"].mean().sort_values())
print("\nIncorrect answers:")
display(df[~df["correct"]][["id", "category", "question", "expected", "predicted"]])
