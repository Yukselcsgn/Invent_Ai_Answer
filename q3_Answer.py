def find_longest_unique_substring(s):
    char_set = set()
    left = 0
    max_length = 0
    result = ""

    for right in range(len(s)):
        # If character is already in our set, shrink window from left
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        char_set.add(s[right])
        current_length = right - left + 1

        # Update result if we found a longer substring
        if current_length > max_length:
            max_length = current_length
            result = s[left:right + 1]

    return result, max_length


if __name__ == "__main__":
    input_str = input("input: ").strip()
    substring, length = find_longest_unique_substring(input_str)
    print(f"output: {substring} length: {length}")