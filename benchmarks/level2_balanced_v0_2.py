"""3Beethoven Level 2 balanced diagnostic benchmark v0.2.

Purpose:
- Establish a cleaner pre-distillation baseline.
- Reduce answer-position bias by balancing correct choices across A/B/C/D.
- Freeze this set BEFORE teacher-data generation.
- NEVER include these questions or close paraphrases in training data.

Assumes `model` and `tokenizer` are already loaded in the Kaggle notebook.
"""

import re
import pandas as pd
import torch
from collections import Counter

BENCHMARK = [
    ('harmony_counterpoint', 'In C major, which pitches form the dominant seventh chord?', ['A. C-E-G-B', 'B. F-A-C-E', 'C. G-B-D-F', 'D. D-F-A-C'], 'C'),
    ('harmony_counterpoint', 'In C major, the leading-tone diminished triad is built on:', ['A. B-D-F', 'B. A-C-E', 'C. D-F-A', 'D. E-G-B'], 'A'),
    ('harmony_counterpoint', 'A cadential 6/4 in common-practice harmony is best understood as:', ['A. A tonic prolongation with no dominant function', 'B. A predominant substitute for ii6', 'C. A modulation to the subdominant', 'D. A dominant embellishment resolving to V'], 'D'),
    ('harmony_counterpoint', 'In C major, which chord is V/ii?', ['A. E major', 'B. A major', 'C. B major', 'D. D major'], 'B'),
    ('harmony_counterpoint', 'Oblique motion occurs when:', ['A. Both voices move in the same direction by different intervals', 'B. Both voices move in opposite directions', 'C. One voice stays on the same pitch while the other moves', 'D. Both voices repeat the same interval'], 'C'),
    ('harmony_counterpoint', 'Which voice-leading motion is normally avoided in strict common-practice part writing?', ['A. Contrary sixths', 'B. Parallel perfect fifths', 'C. Oblique thirds', 'D. Stepwise contrary motion'], 'B'),
    ('harmony_counterpoint', 'A retardation differs from a suspension because the dissonance normally resolves:', ['A. By leap downward', 'B. By remaining unresolved', 'C. By chromatic descent', 'D. Upward by step'], 'D'),
    ('harmony_counterpoint', 'A descending-fifths sequence is most clearly represented by:', ['A. I-IV-vii°-iii-vi-ii-V-I', 'B. I-V-vi-IV', 'C. I-iii-IV-ii', 'D. I-ii-iii-IV'], 'A'),
    ('harmony_counterpoint', 'In a prepared 4-3 suspension, the dissonant 4th above the bass normally:', ['A. Leaps upward to a sixth', 'B. Repeats unchanged', 'C. Resolves downward by step to a third', 'D. Resolves upward by half step'], 'C'),
    ('harmony_counterpoint', 'In C major, a German augmented-sixth chord typically contains:', ['A. A-C-E-F#', 'B. Ab-C-Eb-F#', 'C. Ab-B-D-F#', 'D. F-Ab-C-D#'], 'B'),
    ('form_analysis', 'Simple binary form is most conventionally represented as:', ['A. ABA', 'B. ABACA', 'C. AAB', 'D. ||:A:||:B:||'], 'D'),
    ('form_analysis', 'Simple ternary form is most conventionally represented as:', ['A. AB', 'B. ABA', 'C. ABC', 'D. AABB'], 'B'),
    ('form_analysis', 'In sonata form, the development section typically features:', ['A. Tonal instability, fragmentation, and transformation of earlier material', 'B. A stable return of all themes in the tonic', 'C. Only a literal repeat of the exposition', 'D. A complete absence of modulation'], 'A'),
    ('form_analysis', 'Rounded binary form differs from simple binary because:', ['A. It always has three unrelated themes', 'B. The first section never returns', 'C. Material from the opening returns within the second large section', 'D. It must be in triple meter'], 'C'),
    ('form_analysis', 'The Classical minuet and trio is most accurately described as:', ['A. A through-composed form with no repetition', 'B. A compound ternary design', 'C. A strict fugue', 'D. A sonata without development'], 'B'),
    ('form_analysis', 'In a Baroque concerto, the ritornello principle centers on:', ['A. A recurring solo cadenza', 'B. A repeated ground bass only', 'C. Alternating recitative and aria', 'D. Recurring tutti material separated by contrasting solo episodes'], 'D'),
    ('form_analysis', 'A strophic song uses:', ['A. New music for every stanza', 'B. No repeated text', 'C. Essentially the same music for successive stanzas', 'D. Only instrumental music'], 'C'),
    ('form_analysis', 'Which form combines rondo returns with sonata-like tonal and developmental procedures?', ['A. Sonata-rondo', 'B. Passacaglia', 'C. Motet', 'D. Through-composed aria'], 'A'),
    ('form_analysis', 'In a fugue exposition, one normally expects:', ['A. A cadenza in each voice', 'B. Successive subject or answer entries across the participating voices', 'C. A complete silence between every entry', 'D. Only episodic sequential material'], 'B'),
    ('form_analysis', 'Theme and variations is defined most fundamentally by:', ['A. A recurring refrain alternating with episodes', 'B. A fixed bass with no melodic relation', 'C. A sequence of unrelated miniatures', 'D. Successive transformations of an initial theme'], 'D'),
    ('orchestration', 'An English horn in F sounds a written C as:', ['A. C at concert pitch', 'B. B-flat below', 'C. F a perfect fifth lower', 'D. G a perfect fourth higher'], 'C'),
    ('orchestration', 'A B-flat trumpet sounds a written C as:', ['A. B-flat, a major second lower', 'B. D, a major second higher', 'C. F, a perfect fifth lower', 'D. C at concert pitch'], 'A'),
    ('orchestration', 'The standard tuning of the viola from lowest to highest is:', ['A. G-D-A-E', 'B. C-F-Bb-Eb', 'C. E-A-D-G', 'D. C-G-D-A'], 'D'),
    ('orchestration', 'Why is tenor clef commonly used for cello in a higher register?', ['A. To indicate pizzicato', 'B. To reduce excessive ledger lines', 'C. To transpose the cello up an octave', 'D. To indicate harmonics'], 'B'),
    ('orchestration', 'Timpani are best classified as:', ['A. Untuned metal percussion', 'B. Keyboard percussion', 'C. Tuned percussion with adjustable pitch', 'D. Double-reed instruments'], 'C'),
    ('orchestration', 'In a conventional orchestral harp glissando, the available pitch collection is primarily determined by:', ["A. The harp's pedal settings", "B. The conductor's baton pattern", 'C. The bowing direction', "D. The player's embouchure"], 'A'),
    ('orchestration', 'Flutter-tonguing on a wind instrument produces sound through:', ['A. Muting the bell completely', 'B. Striking the instrument body', 'C. Using only key noise without airflow', 'D. Rapid tongue or uvular flutter in the airstream'], 'D'),
    ('orchestration', 'A natural harmonic on a bowed string instrument is commonly produced by:', ['A. Pressing the string fully to the fingerboard', 'B. Lightly touching the string at a nodal point', 'C. Playing only with the wood of the bow', 'D. Detuning the string during the note'], 'B'),
    ('orchestration', 'The contrabassoon conventionally sounds:', ['A. A major second lower than written', 'B. At written pitch', 'C. One octave lower than written', 'D. One octave higher than written'], 'C'),
    ('orchestration', 'Stopped horn technique involves:', ['A. Inserting the hand deeply into the bell to alter timbre and pitch behavior', 'B. Removing the mouthpiece', 'C. Playing only open harmonics without the hand', 'D. Muting the instrument with cloth outside the bell'], 'A'),
    ('style_comparison', 'Which composer is especially associated with piano character pieces and cycles such as Carnaval and Kreisleriana?', ['A. Verdi', 'B. Robert Schumann', 'C. Corelli', 'D. Rameau'], 'B'),
    ('style_comparison', 'Which composer is especially known for monumental symphonies, broad formal spans, and powerful brass chorales?', ['A. Couperin', 'B. Rossini', 'C. Scarlatti', 'D. Bruckner'], 'D'),
    ('style_comparison', 'Which composer is especially renowned for exceptionally refined orchestration in works such as Daphnis et Chloé?', ['A. Ravel', 'B. Palestrina', 'C. Schütz', 'D. C. P. E. Bach'], 'A'),
    ('style_comparison', 'Which composer is most closely associated with Italian verismo opera and works such as La bohème and Tosca?', ['A. Mendelssohn', 'B. Haydn', 'C. Puccini', 'D. Byrd'], 'C'),
    ('style_comparison', 'Which composer became a central figure in the English oratorio tradition with works such as Messiah?', ['A. Monteverdi', 'B. Handel', 'C. Mahler', 'D. Webern'], 'B'),
    ('style_comparison', 'Which composer is especially associated with hundreds of one-movement keyboard sonatas, often showing Iberian rhythmic and guitar-like influences?', ['A. Brahms', 'B. Franck', 'C. Elgar', 'D. Domenico Scarlatti'], 'D'),
    ('style_comparison', 'Which composer is strongly associated with Finnish nationalism and organic thematic transformation in symphonic writing?', ['A. Sibelius', 'B. Offenbach', 'C. Pergolesi', 'D. Lully'], 'A'),
    ('style_comparison', 'Which composer is especially associated with motoric rhythm, biting harmony, and a mixture of modernist and deliberately Classical gestures?', ['A. Fauré', 'B. Tallis', 'C. Prokofiev', 'D. Corelli'], 'C'),
    ('style_comparison', 'Which composer is especially associated with lush late-Romantic harmony and highly virtuosic piano writing in the twentieth century?', ['A. Boulez', 'B. Rachmaninoff', 'C. Purcell', 'D. Gluck'], 'B'),
    ('style_comparison', 'Which composer is especially associated with extremely concise, pointillistic textures within the Second Viennese School?', ['A. Dvořák', 'B. Saint-Saëns', 'C. Tchaikovsky', 'D. Webern'], 'D'),
    ('history_context', "Beethoven's Ninth Symphony premiered in 1824 in:", ['A. Vienna', 'B. Paris', 'C. London', 'D. Prague'], 'A'),
    ('history_context', 'George Frideric Handel spent the major part of his mature career in:', ['A. Rome', 'B. Leipzig', 'C. London', 'D. Madrid'], 'C'),
    ('history_context', 'Wolfgang Amadeus Mozart was born in:', ['A. Bonn', 'B. Hamburg', 'C. Vienna', 'D. Salzburg'], 'D'),
    ('history_context', 'Robert Schumann helped found and edit which influential music journal?', ['A. Allgemeine musikalische Zeitung', 'B. Neue Zeitschrift für Musik', 'C. The Musical Times', 'D. Le Ménestrel'], 'B'),
    ('history_context', 'Franz Liszt served as Kapellmeister and promoted new music in which German city?', ['A. Weimar', 'B. Dresden', 'C. Hamburg', 'D. Cologne'], 'A'),
    ('history_context', 'Which composer became symbolically associated with the Italian Risorgimento?', ['A. Gounod', 'B. Bruckner', 'C. Verdi', 'D. Vaughan Williams'], 'C'),
    ('history_context', 'Gustav Mahler served for a decade as director of the:', ['A. Leipzig Gewandhaus Orchestra', 'B. Paris Conservatoire', 'C. Royal Opera House, Covent Garden', 'D. Vienna Court Opera'], 'D'),
    ('history_context', 'Which wealthy patron supported Tchaikovsky for years through correspondence while largely avoiding personal meetings?', ['A. Alma Mahler', 'B. Nadezhda von Meck', 'C. Cosima Wagner', 'D. Pauline Viardot'], 'B'),
    ('history_context', 'Johannes Brahms spent much of his mature professional life in:', ['A. St. Petersburg', 'B. Milan', 'C. Vienna', 'D. Brussels'], 'C'),
    ('history_context', "Mendelssohn's famous 1829 revival of Bach's St Matthew Passion took place in:", ['A. Berlin', 'B. Paris', 'C. Rome', 'D. Amsterdam'], 'A'),
]

expected_distribution = Counter(item[3] for item in BENCHMARK)
print("Expected answer distribution:", dict(sorted(expected_distribution.items())))


def ask_student(item):
    category, question, choices, expected = item

    prompt = f"""Answer this classical music multiple-choice question.

Question:
{question}

{chr(10).join(choices)}

Reply with ONLY the letter A, B, C, or D.
Do not explain.
"""

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    match = re.search(r"\b([ABCD])\b", generated.upper())
    return match.group(1) if match else "INVALID"


rows = []

print("\nRunning 3Beethoven Level 2 balanced diagnostic v0.2...\n")

for idx, item in enumerate(BENCHMARK, 1):
    category, question, choices, expected = item
    predicted = ask_student(item)
    correct = predicted == expected

    rows.append(
        {
            "id": idx,
            "category": category,
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
        }
    )

    print(
        f"{idx:02d}/{len(BENCHMARK)} "
        f"{'✅' if correct else '❌'} "
        f"{category} expected={expected} got={predicted}"
    )

df = pd.DataFrame(rows)

print("\n==============================")
print("LEVEL 2 BALANCED BASELINE")
print("==============================")
print(f"Overall accuracy: {df['correct'].mean():.1%}")

print("\nAccuracy by category:")
print(df.groupby("category")["correct"].mean().sort_values())

print("\nPredicted answer distribution:")
print(df["predicted"].value_counts().sort_index())

print("\nIncorrect answers:")
display(
    df.loc[
        ~df["correct"],
        ["id", "category", "question", "expected", "predicted"],
    ]
)
