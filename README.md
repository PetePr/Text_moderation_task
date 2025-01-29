> My approach to text moderation (with mBERT) <

The code is split between the two files: preprocess.py and tokenization_training_finetuning.py. 
The first one is a preprocessing pipeline that imports original file and outputs another .csv file that contains the following columns:
1. about_me -- cleaned text
2-3.  label and contact details -- labels(unchanged)
4. non_alpha_density -- proportion of symbols that are neither numbers nor regular letters, expressed as a percentage of the whole text
5. high_density_clusters -- chunks that contain more than half of non-alphanumeric symbols(local desity>0.5)
6. language -- sample's detected language
7. intermixed words -- uninterrupted strings that contain both letters of sample's "native" language and any other symbols
The process:
After being uploaded as a dataframe, the data is first whitespace normalized, since excessive spacing is almost never carries any meaning and can only used to obfuscate data.
Before any further transformation, extra features 4, 5 are extracted, along with another unused metric: sub_super_count, which accounts for sub- and superscript symbols in the sample.
This feature is ommited in the present version since this dataset specifically only contains one sample for which it makes sense; althogh provided a wider dataset is used
this feature can become meaningful and very useful indeed. Next step is repeated character normalization, after which the language is detected for language-specific treatise.
(LangDetect is used with a character-level extension to handle samples with little and\or misspeled words, as well as mixed scripts). Another transformative function after that
is meant to convert emojiis and special fonts into regular text to unmask hidden messages. This normalized text is split by detected language for further cleaning: numerals normalization,
contractions handling and spellcheck(not implemented yet). Finally, it is merged back again and exported to a .csv file(see examples with underscores).
Room for improvement: more robust language detection and extended language-specific treatment, such as clean text and actually implement Spellcheck(which is quite demanding of input), add
contraction handling to more languages and treat slang words, determine optimal threshholds for high-density clusters(now length=10, density>0.5), engineer some additional features.

This preprocessed is then tokenized with multilingualBERT tokenizer with custom tokens and fed into mBert to produce over 85% accuracy after 3 epochs.
Room for improvement: determine optimal training time(results do improve up to some epoch beyond 3rd), more fine-tuning, and obviously 
generate more training data(even after addition of 125 samples results improved).
