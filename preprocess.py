# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:42:59 2025

@author: Admin
"""

import pandas as pd
import contractions
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from collections import Counter
import re
import emoji
import unicodedata
from spellchecker import SpellChecker

file_path = r"C:\Users\Admin\Documents\Study\Test_tasks\ftdata.csv"

# Load the CSV file
data = pd.read_csv(file_path)
text_column = 'about_me'

# Whitespace normalization function
def normalize_whitespace(text):
    if isinstance(text, str):
        # Remove leading/trailing spaces and normalize multiple spaces
        return ' '.join(text.split())
    return text

# Apply whitespace normalization
data[text_column] = data[text_column].apply(normalize_whitespace)

def character_level_analysis(data, text_column, cluster_length=10, density_threshold=0.5):
    """
    Analyze character-level features of the specified text column in a DataFrame.
    Modifies the DataFrame in place by adding new feature columns.
    :param data: Input DataFrame.
    :param text_column: Name of the text column to process.
    :param cluster_length: Length of text segments for high-density cluster analysis.
    :param density_threshold: Density threshold to define high-density clusters.
    """
    def analyze_text(text):
        if not isinstance(text, str):
            return {"non_alpha_density": 0, "sub_super_count": 0, "high_density_clusters": 0}
        
        # Calculate non-alphanumeric density
        non_alpha_density = sum(1 for char in text if not char.isalnum() and not char.isspace()) / max(len(text), 1)

        # Detect subscript/superscript symbols
        sub_super_match = re.findall(r"[\u2070-\u209F]+", text)

        # Unicode ranges for combining characters and decorative marks
        combining_marks = r"\u0300-\u036F\uFE20-\uFE2F\u20D0-\u20FF"
    
        # Regex pattern to identify non-alphanumeric characters excluding emojis
        non_alnum_pattern = rf"[^\w\s{combining_marks}]"

        # Remove emojis from the text for analysis
        text_without_emojis = emoji.replace_emoji(text, replace='')

        cluster_count = 0
        clusters = []
        i = 0

        # Iterate through the text to count clusters of high-density non-alphanumeric characters
        while i <= len(text_without_emojis) - cluster_length:
            segment = text_without_emojis[i:i + cluster_length]
            non_alnum_count = len(re.findall(non_alnum_pattern, segment))
            if non_alnum_count / cluster_length >= density_threshold:
                cluster_count += 1
                clusters.append(segment)
                # Skip the length of the cluster to avoid double-counting overlapping segments
                i += cluster_length
            else:
                i += 1
                
        return {
            "non_alpha_density": non_alpha_density,
            "sub_super_count": len(sub_super_match),
            "high_density_clusters": clusters, # Excludes emojis but includes combining characters and decorative marks
            "high_density_cluster_count": cluster_count
        }

    # Apply analysis to the specified text column
    analysis_results = data[text_column].apply(analyze_text)

    # Extract and add new columns to the DataFrame
    data["non_alpha_density"] = analysis_results.apply(lambda x: x["non_alpha_density"])
    #data["sub_super_count"] = analysis_results.apply(lambda x: x["sub_super_count"])
    data["high_density_clusters"] = analysis_results.apply(lambda x: x["high_density_clusters"])

# Run the function on the 'about_me' column
character_level_analysis(data, text_column="about_me", cluster_length=10, density_threshold=0.5)


# Function to normalize repeated characters
def normalize_repeated_chars_exclude_numbers(text, threshold=2):
    """
    Normalizes repeated characters but excludes sequences of digits.
    :param text: Input string to process.
    :param threshold: Max allowed repetitions before truncation.
    :return: Normalized string.
    """
    if isinstance(text, str):
        # Regex to match non-digit repeated characters
        pattern = re.compile(r"([^0-9])\1{" + str(threshold) + ",}")
        # Replace matches, keeping threshold repetitions
        normalized_text = pattern.sub(r"\1" * threshold, text)
        return normalized_text
    return text

data[text_column] = data[text_column].apply(
    lambda x: normalize_repeated_chars_exclude_numbers(x, threshold=2)
)

# Define the popular languages whitelist
POPULAR_LANGUAGES = {"en", "es", "fr", "de", "ru", "pt", "uk", "it", "pl"}

# Ensure consistent results with langdetect
DetectorFactory.seed = 0
# Language detection and DataFrame separation
def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang in POPULAR_LANGUAGES else "unknown"
    except LangDetectException:
        return "unknown"

# Character-based fallback for unknown languages
def analyze_characters(text):
    if re.search(r'[\u0400-\u04FF]', text):  # Cyrillic characters
        if "і" in text.lower():
            return "uk"  # Likely Ukrainian
        return "ru"  # Likely Russian
    elif re.search(r'[àâäéèêëîïôùûüç]', text):  # French accents
        return "fr"
    elif re.search(r'[ñáéíóúü]', text):  # Spanish accents
        return "es"
    elif re.search(r'[äöüß]', text):  # German characters
        return "de"
    return "unknown"

# Detect language with fallback
def robust_language_detection(text):
    lang = detect_language(text)
    if lang == "unknown":
        lang = analyze_characters(text)
    return lang

# Apply robust language detection
data["language"] = data["about_me"].apply(robust_language_detection)

# Function to detect words with letters intermixed with other characters (e.g., numbers, symbols, etc., excluding punctuation)
def detect_intermixed_words(text):
    # Define the regex pattern: words containing both letters and non-letters intermixed (excluding punctuation-only words)
    # Also detect words with mixed alphabet letters (e.g., Latin + Cyrillic)
    pattern = r'\b(?=[^\s]*[a-zA-Z])(?=[^\s]*[\u0400-\u04FF])[^\s]+\b|\b(?=[^\s]*[a-zA-Z\u0400-\u04FF])(?=[^\s]*[\d@#$%^&*_])[^\s]+\b'
    intermixed = re.findall(pattern, text)
    return intermixed

# Apply the detection function to the 'about_me' column
data['intermixed_words'] = data['about_me'].apply(detect_intermixed_words)

# Function to replace styled symbols and emojis with plain letters
def replace_symbols_and_emojis(text):
    # Fullwidth, bold, italic, circled, etc. character mappings
    unicode_mappings = {}
    
    # Add Fullwidth A-Z, a-z
    for code in range(0xFF21, 0xFF3B):  # Fullwidth A-Z
        unicode_mappings[chr(code)] = chr(code - 0xFF21 + ord('A'))
    for code in range(0xFF41, 0xFF5B):  # Fullwidth a-z
        unicode_mappings[chr(code)] = chr(code - 0xFF41 + ord('a'))
    # Add Mathematical Sans-Serif Uppercase Letters (A-Z)
    for code in range(0x1D5A0, 0x1D5BA):
        unicode_mappings[chr(code)] = chr(code - 0x1D5A0 + ord('A'))
    # Add Mathematical Sans-Serif Lowercase Letters (a-z)
    for code in range(0x1D5BA, 0x1D5D4):
        unicode_mappings[chr(code)] = chr(code - 0x1D5BA + ord('a'))
    # Add Enclosed Alphanumeric Button Letters (🅰 - 🆊)
    for code in range(0x1F170, 0x1F17A):  # A-J
        unicode_mappings[chr(code)] = chr(code - 0x1F170 + ord('A'))
    for code in range(0x1F17A, 0x1F183):  # K-T
        unicode_mappings[chr(code)] = chr(code - 0x1F17A + ord('K'))
    for code in range(0x1F183, 0x1F189):  # U-Z
        unicode_mappings[chr(code)] = chr(code - 0x1F183 + ord('T'))
    # Add Mathematical Bold, Italic, Script, Fraktur mappings (A-Z, a-z)
    for offset, start in enumerate([0x1D400, 0x1D41A, 0x1D434, 0x1D44E, 0x1D504, 0x1D51E]):
        for code in range(start, start + 26):
            unicode_mappings[chr(code)] = chr(ord('A') + code - start)
        for code in range(start + 26, start + 52):
            unicode_mappings[chr(code)] = chr(ord('a') + code - start - 26)
    # Add Circled and Parenthesized A-Z, a-z
    for code in range(0x24B6, 0x24D0):  # Circled A-Z
        unicode_mappings[chr(code)] = chr(code - 0x24B6 + ord('A'))
    for code in range(0x24D0, 0x24EA):  # Circled a-z
        unicode_mappings[chr(code)] = chr(code - 0x24D0 + ord('a'))
    # Add Squared Latin Letters (🄰 - 🅉)
    for code in range(0x1F130, 0x1F14A): # Squared uppercase A-Z
        unicode_mappings[chr(code)] = chr(code - 0x1F130 + ord('A'))
    # Add Negative Squared Latin Letters (🅰 - 🅉)
    for code in range(0x1F150, 0x1F16A):  # Negative squared uppercase A-Z
        unicode_mappings[chr(code)] = chr(code - 0x1F150 + ord('A'))
       
    # Add Regional Indicator Symbols (A-Z)
    for code in range(0x1F1E6, 0x1F200):
        unicode_mappings[chr(code)] = chr(code - 0x1F1E6 + ord('A'))
    
    # Add Blood Type Emojis
    emoji_mappings = {
        "🅰": "A", "🅱": "B", "🅾": "O", "🆎": "AB",
        "ℹ️": "i", "Ⓜ️": "M"
    }
    unicode_mappings.update(emoji_mappings)
    
    # Replace styled symbols and emojis with plain equivalents
    text = ''.join(unicode_mappings.get(char, char) for char in text)
    
    # Demojize remaining emojis
    text = emoji.demojize(text, delimiters=(" :", ": "))
    
    return text

# Function to process the "about_me" column in the DataFrame
def replace_in_dataframe(data):
    # Check if the "about_me" column exists
    if "about_me" not in data.columns:
        raise ValueError('The DataFrame does not have a column named "about_me".')

    # Apply the replacement function to the "about_me" column
    data["about_me"] = data["about_me"].apply(lambda x: replace_symbols_and_emojis(str(x)))

replace_in_dataframe(data)

# Add mappings for superscript and subscript digits
sub_super_mappings = {}

# Superscript digits (¹²³⁴⁵⁶⁷⁸⁹⁰)
superscript_digits = "¹²³⁴⁵⁶⁷⁸⁹⁰"
for i, char in enumerate(superscript_digits):
    sub_super_mappings[char] = str(i)

# Subscript digits (₀₁₂₃₄₅₆₇₈₉)
subscript_digits = "₀₁₂₃₄₅₆₇₈₉"
for i, char in enumerate(subscript_digits):
    sub_super_mappings[char] = str(i)

# Function to replace subscript and superscript digits
def normalize_sub_super_digits(text):
    return ''.join(sub_super_mappings.get(char, char) for char in text)

data[text_column] = data[text_column].apply(normalize_sub_super_digits)

# Separate DataFrames by detected language
language_dfs = {}
for language in data["language"].unique():
    language_dfs[language] = data[data["language"] == language].copy()

# Count occurrences of each language
language_counts = Counter(data["language"])

# Print message with total counts
#message = "\n".join([f"Language: {lang}, Count: {count}" for lang, count in language_counts.items()])
#print("Language detection complete. Total counts by language:")
#print(message)

# Initialize spell checker
spell = SpellChecker()

def fix_typos(text):
    """
    Correct common typos in the text using a spell checker.
    :param text: Input string to process.
    :return: Corrected string.
    """
    if not isinstance(text, str):
        return text

    # Split text into words, correct each word, and rejoin
    words = text.split()
    corrected_words = [
        spell.correction(word) if spell.correction(word) else word
        for word in words
        if word is not None
    ]
    return ' '.join(corrected_words)

# Attempt to implement spellcheck
#language_dfs['en']["about_me"] = language_dfs['en']["about_me"].apply(fix_typos)

# Define dictionaries for obfuscated numerals
numeral_mappings = {
    "en": {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
    },
    "es": {
        "cero": "0", "uno": "1", "dos": "2", "tres": "3", "cuatro": "4",
        "cinco": "5", "seis": "6", "siete": "7", "ocho": "8", "nueve": "9"
    },
    "ru": {
        "ноль": "0", "один": "1", "два": "2", "три": "3", "четыре": "4", 
        "пять": "5", "шесть": "6", "семь": "7", "восемь": "8", "девять": "9",
        "одиннадцать": "11", "сорок": "40"
        },
    "pt": {
        "zero": "0",
        "um": "1", "uma": "1", "dois": "2", "duas": "2", "três": "3", "tres": "3",
        "quatro": "4", "cinco": "5", "seis": "6", "sete": "7", "oito": "8", "nove": "9",
        "dez": "10", "vinte": "20"
        },
    "uk": {
        "нуль": "0", "ноль": "0", "один": "1", "два": "2", "три": "3", "чотири": "4",
        "п’ять": "5", "пять": "5", "шість": "6", "сім": "7", "вісім": "8", "дев’ять": "9", 
        "девять": "9", "десять": "10"
        }
}

def normalize_obfuscated_text(text, language):
    """Normalize obfuscated text by replacing numbers written in words with digits.
    
    :param text: Input string to process.
    :param language: Language code ("en" for English, "es" for Spanish).
    :return: Normalized string with original casing preserved.
    """
    if not isinstance(text, str):
        return text

    # Get the appropriate numeral mapping for the language
    mapping = numeral_mappings.get(language, {})
    if not mapping:
        raise ValueError(f"Unsupported language code: {language}")

    # Tokenize text while preserving spaces
    words = re.findall(r'\b\w+\b|\s+|[^\w\s]', text)

    # Replace words in the mapping while preserving the original casing
    normalized_words = []
    for word in words:
        lower_word = word.lower()
        if lower_word in mapping:
            normalized_words.append(mapping[lower_word])
        else:
            normalized_words.append(word)

    # Combine back into a single string
    normalized_text = ''.join(normalized_words)

    return normalized_text

# Apply the function to the appropriate language's DataFrame
for lang, df in language_dfs.items():
    if lang in numeral_mappings:  # Only process languages with numeral mappings
        df["about_me"] = df["about_me"].apply(lambda x: normalize_obfuscated_text(x, lang))

# Function to expand contractions in text
def expand_contractions(text):
    if isinstance(text, str):
        return contractions.fix(text)
    return text

# Apply contraction expansion to the text column
language_dfs['en']["about_me"] = language_dfs['en']["about_me"].apply(expand_contractions)

processed_data = pd.concat(language_dfs.values(), ignore_index=True)

# Save the updated data to a new CSV file
output_path = r"C:\Users\Admin\Documents\Study\Test_tasks\fine_tune_data.csv"
processed_data.to_csv(output_path, index=False)






