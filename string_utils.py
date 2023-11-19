import itertools
import string
from tqdm import tqdm

def most_common_string(strings):
    """
    This function takes a list of strings and returns the most common string.
    If there is a tie, it returns the first one among the most common ones.
    """
    from collections import Counter

    # Count the frequency of each string
    string_counts = Counter(strings)

    # Find the string(s) with the highest frequency
    max_count = max(string_counts.values())
    most_common_strings = [string for string, count in string_counts.items() if count == max_count]

    # Return the first one among the most common strings
    return most_common_strings[0]

def generate_candidate_words(word_list):
    """
    Generate a set of candidate words based on the given word list.
    This function generates variations by adding, removing, or substituting characters.
    """
    #alphabet = string.ascii_letters + string.digits + string.punctuation
    
    alphabet = set("".join([word for word in word_list]))
    
    
    candidates = set(word_list)  # Start with the original words

    for word in word_list:
        # Additions
        for i in range(len(word) + 1):
            for char in alphabet:
                candidates.add(word[:i] + char + word[i:])

        # Deletions
        for i in range(len(word)):
            candidates.add(word[:i] + word[i + 1:])

        # Substitutions
        for i in range(len(word)):
            for char in alphabet:
                candidates.add(word[:i] + char + word[i + 1:])
    return set(candidates)

def find_min_average_distance_word(word_list):
    """
    Find the word that has the minimum average Levenshtein distance to all words in the list.
    """
    candidates = generate_candidate_words(word_list)
    min_total_distance = float('inf')
    best_word = None
    
    #print(len(candidates))
    for candidate in tqdm(candidates):
        #print(candidate)
        total_distance = sum(levenshtein_distance(candidate, word) for word in word_list)

        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_word = candidate

    return best_word

def levenshtein_distance(s1, s2):
    """
    Compute the Levenshtein distance between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]