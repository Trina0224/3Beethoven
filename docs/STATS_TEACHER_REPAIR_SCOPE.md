# Teacher data correction scope

User-authorized after viewing V56 results: directly correct teacher mistakes and audit curriculum instead of attributing failures to training randomness. This is an amendment after results, not a preregistered prediction.

Perform one offline pass over existing V56 training data. Fix target-dependent teacher intermediate schemas, normalize exact decimal expressions to rational syntax, remove unsupported mean working from variance questions, and explicitly disambiguate legacy second-moment training wording. Revalidate corrected targets against an independent domain oracle; keep original raw responses in the existing archive. Mark edited records as corrected supervision. Do not manufacture responses for uncollected questions. No new API calls, training, weights, or test-score changes.

Expected: valid usable teacher responses are no longer rejected solely for exact decimal notation; variance prompts no longer request an unknowable mean; ambiguous training wording is removed. This does not establish improved student generalization. Stop after one audited correction pass; unresolved expressions are excluded. Existing evaluation questions and identities remain unchanged. A future curriculum expansion must cover language, context and representation contrasts and use a fresh independent holdout.
