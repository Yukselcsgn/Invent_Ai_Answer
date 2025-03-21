def find_longest_unique_substrings(s):
    char_set = set()
    left = 0
    max_length = 0
    substrings = set()

    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        char_set.add(s[right])
        current_length = right - left + 1

        if current_length > max_length:
            max_length = current_length
            substrings = {s[left:right + 1]}
        elif current_length == max_length:
            substrings.add(s[left:right + 1])

    return substrings, max_length


if __name__ == "__main__":
    input_str = input("input: ").strip()
    substrings, length = find_longest_unique_substrings(input_str)

    for substring in substrings:
        print(f"output: {substring} length: {length}")