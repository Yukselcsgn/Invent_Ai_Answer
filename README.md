# Q3 Time and Space Complexity

## The Program
The program finds the longest non-repetitive subsequence using sliding window and two-pointer techniques.

---

## 1. How the Algorithm Works

While scanning the array from beginning to end with the right pointer, it advances the left pointer when necessary, thus preventing repeated sub-arrays from being found in the main array. The working steps of the algorithm are as follows:

- The right pointer scans the string from beginning to end and adds each character to a `set` data structure.
- If the character is repeated, the left cursor moves to remove the repeated character.
- At each step, the length of the current subsequence is calculated, and the longest subsequence is updated.
- This process continues until the end of the array, and the longest non-repeating sub-array is obtained.

---

## 2. Time Complexity Analysis

- The algorithm performs in **O(n)** time complexity because the right pointer scans the string only once.
- As mentioned above, the left cursor moves only when necessary. Since a character can be added to the set at most once and removed at most once, it moves at most **n** times in total. This results in **O(n)** time complexity.
- Insertion, removal, and search operations on the `set` data structure run with an average time complexity of **O(1)**.
- Since each character is processed only once, the time complexity of the algorithm is **O(n)**.

---

## 3. Space Complexity Analysis

- Since the `set` data structure stores only non-repeating characters, it takes **O(n)** space in the worst case when all characters are distinct.
- The `max_substring` variable, which stores the longest substring, can also contain a maximum of **n** characters.
- The worst-case space complexity of the algorithm is determined as **O(n)**.