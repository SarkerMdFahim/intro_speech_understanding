def words2characters(words):
    """
    This function converts a list of words into a list of characters.
    """

    characters = []

    for word in words:
        # Convert each element to string
        word = str(word)

        # Split into characters and append
        for char in word:
            characters.append(char)

    return characters


# Input list
words = ['hello', 1.234, True]

# Function call
result = words2characters(words)

# Printing output
print("Input:", words)
print("Output:", result)
