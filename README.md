# Era-Prediction-for-Literature

CSCI3349.01 Final Project

Ray's Update Version 1.0.0

WHAT I CHANGED

- src/book_select.py: N_PER_ERA is 100 instead of 50 (still one author per era, up to 100 per era). Run python src/book_select.py to refresh Dataset/sample_by_era.csv and sample_by_decade.csv. I didn't change anything about the decade yet

- src/config.py and src/book_select.py: RANDOM_STATE is 42 instead of 612 for sampling and training.

- src/book_clean.py: after normal Gutenberg header/footer removal, books that are too short (under 300 words) or have too high a fraction of lines that look like MIDI paths, copyright lines, or Producer lines are not written to era_sample_clean or decade_sample_clean; an old clean file with the same name is deleted if the book fails the check. Run python src/book_clean.py to apply this to your local clean folders.