"""3Beethoven Level 2 diagnostic benchmark v0.1.

Purpose: find headroom before distillation. This set is frozen BEFORE teacher-data
creation and MUST NOT appear in training data.

Assumes `model` and `tokenizer` are already loaded in the Kaggle notebook.
"""

import re
import pandas as pd
import torch

BENCHMARK = [
    # --- harmony / counterpoint ---
    ("harmony_counterpoint", "In C major, which chord functions most conventionally as V/V?", ["A. D major", "B. D minor", "C. A minor", "D. F major"], "A"),
    ("harmony_counterpoint", "In common-practice harmony, a Neapolitan chord is best described as:", ["A. A major triad on the raised fourth degree", "B. A major triad on the lowered second degree, often in first inversion", "C. A diminished seventh built on the leading tone", "D. A minor triad on the submediant"], "B"),
    ("harmony_counterpoint", "A 4-3 suspension normally resolves by:", ["A. Leaping up a fourth", "B. Remaining stationary", "C. Moving down by step from a fourth above the bass to a third", "D. Moving up by chromatic semitone"], "C"),
    ("harmony_counterpoint", "Which progression is the conventional deceptive cadence in a major key?", ["A. IV-I", "B. ii-V", "C. V-vi", "D. I-V"], "C"),
    ("harmony_counterpoint", "What is stretto in a fugue?", ["A. A slower statement of the subject", "B. Overlapping subject entries before the previous entry has finished", "C. A cadenza inserted before the final cadence", "D. A subject played only in the bass"], "B"),
    ("harmony_counterpoint", "Invertible counterpoint at the octave means that:", ["A. The counterpoint must remain in one register", "B. The two contrapuntal voices can exchange upper and lower positions at the octave", "C. Every interval must be an octave", "D. One voice must be doubled at the octave"], "B"),
    ("harmony_counterpoint", "What is a pedal point?", ["A. A rapidly repeated ornamental turn", "B. A sustained or repeated pitch while harmonies change around it", "C. A modulation by descending fifths", "D. A silent measure before a cadence"], "B"),
    ("harmony_counterpoint", "A Picardy third is:", ["A. A minor dominant in a major key", "B. A major tonic chord ending a piece or section otherwise in a minor mode", "C. A chromatic mediant modulation", "D. A third inversion dominant seventh"], "B"),
    ("harmony_counterpoint", "In species counterpoint, contrary motion occurs when:", ["A. Both voices move in the same direction by the same interval", "B. One voice stays fixed", "C. The voices move in opposite directions", "D. Both voices repeat the same pitch"], "C"),
    ("harmony_counterpoint", "An Italian augmented-sixth chord in C major normally contains which characteristic chromatic pitches?", ["A. D-flat and B-natural", "B. A-flat and F-sharp", "C. E-flat and C-sharp", "D. B-flat and G-sharp"], "B"),

    # --- form / analysis ---
    ("form_analysis", "In a Baroque concerto, the term ritornello most commonly refers to:", ["A. A recurring orchestral passage that returns between solo episodes", "B. A solo cadenza without accompaniment", "C. A slow introduction to a fugue", "D. A repeated bass ostinato only"], "A"),
    ("form_analysis", "In a typical Classical major-key sonata exposition, the secondary theme area is most commonly established in:", ["A. The tonic minor", "B. The dominant", "C. The subdominant minor", "D. The mediant minor"], "B"),
    ("form_analysis", "What distinguishes the recapitulation from the exposition in a normative major-key sonata form?", ["A. The recapitulation omits the primary theme", "B. The recapitulation normally keeps the secondary-theme area in the tonic rather than establishing the dominant", "C. The recapitulation always changes to minor mode", "D. The recapitulation contains no cadences"], "B"),
    ("form_analysis", "Which pattern most clearly represents a five-part rondo?", ["A. ABACA", "B. AABB", "C. ABCD", "D. ABA"], "A"),
    ("form_analysis", "A Baroque da capo aria is most characteristically organized as:", ["A. Through-composed without return", "B. ABA, with the opening section returning after a contrasting middle section", "C. Sonata-allegro form", "D. Theme and double variations"], "B"),
    ("form_analysis", "A passacaglia or chaconne is commonly organized around:", ["A. A recurring bass or harmonic pattern", "B. Alternating recitative and aria", "C. A sonata exposition repeated verbatim", "D. A sequence of unrelated dances with no recurring material"], "A"),
    ("form_analysis", "In many Beethoven symphonies, the scherzo historically occupies the position previously associated with the:", ["A. Minuet", "B. Fugue", "C. Recitative", "D. Prelude"], "A"),
    ("form_analysis", "Cyclic form refers to:", ["A. Reuse or transformation of thematic material across multiple movements", "B. A movement that repeats forever without cadence", "C. Any piece in binary form", "D. A work written only for percussion"], "A"),
    ("form_analysis", "A through-composed song differs from a strophic song because it:", ["A. Uses substantially new music as the text progresses instead of repeating the same complete music for each stanza", "B. Must be unaccompanied", "C. Always uses twelve-tone technique", "D. Has no text repetition of any kind"], "A"),
    ("form_analysis", "In a fugue, an episode most commonly:", ["A. Presents a complete subject entry in the tonic every time", "B. Develops fragments or sequences between full subject entries", "C. Stops contrapuntal motion entirely", "D. Functions as the opening exposition"], "B"),

    # --- orchestration / notation ---
    ("orchestration", "For a B-flat clarinet, a written C sounds:", ["A. A major second lower, B-flat", "B. A major second higher, D", "C. A perfect fifth lower, F", "D. At concert pitch, C"], "A"),
    ("orchestration", "For a horn in F, a written C normally sounds:", ["A. A perfect fourth higher", "B. A perfect fifth lower, F", "C. A major second lower", "D. At concert pitch"], "B"),
    ("orchestration", "Which clef is most characteristic of the viola in its normal middle register?", ["A. Soprano clef", "B. Alto clef", "C. Baritone clef", "D. Treble clef only"], "B"),
    ("orchestration", "The piccolo is conventionally notated so that it sounds:", ["A. One octave lower than written", "B. A perfect fifth higher than written", "C. One octave higher than written", "D. At written pitch"], "C"),
    ("orchestration", "The orchestral double bass conventionally sounds:", ["A. One octave lower than written", "B. One octave higher than written", "C. A perfect fifth lower than written", "D. Exactly as written"], "A"),
    ("orchestration", "On bowed strings, sul ponticello means to play:", ["A. Over or near the fingerboard", "B. Near the bridge", "C. With the wood of the bow", "D. Without the bow"], "B"),
    ("orchestration", "Con sordino instructs a player to:", ["A. Use a mute", "B. Play only harmonics", "C. Double another instrument", "D. Improvise freely"], "A"),
    ("orchestration", "Which pair consists entirely of double-reed instruments?", ["A. Flute and clarinet", "B. Oboe and bassoon", "C. Clarinet and saxophone", "D. Horn and bassoon"], "B"),
    ("orchestration", "Which instrument normally provides the lowest standard member of the orchestral string section?", ["A. Viola", "B. Cello", "C. Double bass", "D. Bassoon"], "C"),
    ("orchestration", "Col legno is a string technique in which the player uses:", ["A. The wooden part of the bow to strike or contact the string", "B. Only the left hand with no bow", "C. A mute attached to the bridge", "D. The bow very near the fingerboard"], "A"),

    # --- style / composer language ---
    ("style_comparison", "Which composer is most directly associated with the idée fixe as a recurring autobiographical thematic device in Symphonie fantastique?", ["A. Berlioz", "B. Brahms", "C. Haydn", "D. Corelli"], "A"),
    ("style_comparison", "Which pairing best contrasts the nineteenth-century aesthetics of absolute music and music drama?", ["A. Brahms and Wagner", "B. Vivaldi and Corelli", "C. Palestrina and Lassus", "D. Scarlatti and Couperin"], "A"),
    ("style_comparison", "The term seconda pratica is especially associated with which composer and early-Baroque expressive practice?", ["A. Monteverdi", "B. Chopin", "C. Bruckner", "D. Rachmaninoff"], "A"),
    ("style_comparison", "Smooth vocal polyphony with carefully controlled dissonance is especially associated with the sacred style of:", ["A. Palestrina", "B. Liszt", "C. Mahler", "D. Prokofiev"], "A"),
    ("style_comparison", "Which composer is especially associated with modes of limited transposition and non-retrogradable rhythms?", ["A. Messiaen", "B. Schubert", "C. Handel", "D. Rossini"], "A"),
    ("style_comparison", "Which composer is most closely associated with codifying twelve-tone composition in the early twentieth century?", ["A. Schoenberg", "B. Debussy", "C. Elgar", "D. Grieg"], "A"),
    ("style_comparison", "Which composer is particularly associated with systematic collection and transformation of Eastern European folk music together with modernist techniques?", ["A. Bartók", "B. Puccini", "C. Saint-Saëns", "D. Pergolesi"], "A"),
    ("style_comparison", "Which work-composer pairing is a canonical example of Stravinsky's neoclassical period?", ["A. Pulcinella — Stravinsky", "B. Tristan und Isolde — Stravinsky", "C. Pelléas et Mélisande — Stravinsky", "D. Kinderszenen — Stravinsky"], "A"),
    ("style_comparison", "Which composer is especially associated with the symphonic use of very large orchestras, song-derived material, and movements that juxtapose irony, folk-like elements, and metaphysical ambition?", ["A. Mahler", "B. Telemann", "C. Clementi", "D. Lully"], "A"),
    ("style_comparison", "Which description most strongly points toward Debussy rather than Brahms?", ["A. Frequent use of whole-tone and modal sonorities with timbre functioning as a major structural element", "B. Dense motivic development within Austro-German Classical forms", "C. Strong emphasis on developing variation in the Beethoven tradition", "D. Regular use of Lutheran chorale as a Baroque cantata framework"], "A"),

    # --- history / context ---
    ("history_context", "Beethoven originally associated the Eroica Symphony with which political figure before famously withdrawing the dedication?", ["A. Napoleon Bonaparte", "B. Metternich", "C. Louis XIV", "D. Bismarck"], "A"),
    ("history_context", "The premiere of Stravinsky's The Rite of Spring took place in 1913 in:", ["A. Paris", "B. Vienna", "C. Leipzig", "D. Venice"], "A"),
    ("history_context", "Which institution employed J. S. Bach as Thomaskantor?", ["A. St. Thomas Church and its associated school in Leipzig", "B. The Paris Opéra", "C. The Esterházy court", "D. La Scala"], "A"),
    ("history_context", "Haydn's long service to the Esterházy family most directly illustrates which eighteenth-century system of musical employment?", ["A. Aristocratic patronage", "B. Modern recording contracts", "C. State conservatory tenure", "D. Broadway production"], "A"),
    ("history_context", "Which nineteenth-century development most helped shift virtuoso performers such as Liszt toward a modern public-concert culture?", ["A. Expansion of urban middle-class audiences and public concert institutions", "B. Elimination of the piano from concert life", "C. Closure of European opera houses", "D. Replacement of notation by oral transmission"], "A"),
    ("history_context", "Wagner's Bayreuth Festival Theatre was designed primarily to present:", ["A. Wagner's own music dramas under specialized theatrical conditions", "B. Italian comic opera exclusively", "C. Baroque sacred cantatas", "D. Solo piano recitals"], "A"),
    ("history_context", "Which composer spent much of his mature career in Paris but was born in Poland and became a central figure in Romantic piano music?", ["A. Chopin", "B. Sibelius", "C. Verdi", "D. Purcell"], "A"),
    ("history_context", "Dvořák's New World Symphony was composed during his time in:", ["A. The United States", "B. Russia", "C. Spain", "D. Norway"], "A"),
    ("history_context", "Which composer served as director of the Leipzig Gewandhaus Orchestra and played a major role in reviving public interest in J. S. Bach's music?", ["A. Mendelssohn", "B. Verdi", "C. Rameau", "D. Scriabin"], "A"),
    ("history_context", "The nineteenth-century 'War of the Romantics' is commonly associated with disagreement between supporters of:", ["A. Brahms/Schumann-oriented absolute-music traditions and the Liszt/Wagner New German School", "B. Bach and Handel over opera reform", "C. Mozart and Haydn over serialism", "D. Debussy and Ravel over Baroque continuo"], "A"),
]


def ask_student(item):
    category, question, choices, expected = item
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
        output = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    match = re.search(r"\b([ABCD])\b", generated.upper())
    return match.group(1) if match else "INVALID"


rows = []
print("Running 3Beethoven Level 2 diagnostic...\n")
for idx, item in enumerate(BENCHMARK, 1):
    category, question, choices, expected = item
    predicted = ask_student(item)
    correct = predicted == expected
    rows.append({"id": idx, "category": category, "question": question, "expected": expected, "predicted": predicted, "correct": correct})
    print(f"{idx:02d}/{len(BENCHMARK)} {'✅' if correct else '❌'} {category} expected={expected} got={predicted}")

df = pd.DataFrame(rows)
print("\n==============================")
print("LEVEL 2 BASELINE RESULTS")
print("==============================")
print(f"Overall accuracy: {df['correct'].mean():.1%}")
print("\nAccuracy by category:")
print(df.groupby("category")["correct"].mean().sort_values())
print("\nIncorrect answers:")
display(df[~df["correct"]][["id", "category", "question", "expected", "predicted"]])
