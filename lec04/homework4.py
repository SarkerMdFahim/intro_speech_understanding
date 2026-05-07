def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dictionary mapping date tuples to lists of names

    @return:
    birthday - the next birthday date
    list_of_names - names of people having birthday on that date
    '''

    # Sort all birthday dates
    sorted_birthdays = sorted(birthdays.keys())

    # Find the next birthday after the given date
    for birthday in sorted_birthdays:
        if birthday > date:
            return birthday, birthdays[birthday]

    # If no later birthday exists, return the first birthday of next year
    first_birthday = sorted_birthdays[0]
    return first_birthday, birthdays[first_birthday]


# Example dictionary of birthdays
birthdays = {
    (1, 10): ["Alice"],
    (3, 15): ["Bob", "Charlie"],
    (7, 20): ["David"],
    (12, 25): ["Emma"]
}

# Given date
date = (3, 16)

# Function call
birthday, list_of_names = next_birthday(date, birthdays)

# Printing output
print("Given Date:", date)
print("Next Birthday Date:", birthday)
print("People:", list_of_names)
