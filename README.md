#Q3 COMPLEXITY INTERPRETATION

## Time and Space Complexity Summary

### How the Algorithm Works:
- Uses the sliding window and two-pointer technique.  
- The right pointer scans the string while the left pointer moves when a duplicate is found.  
- Updates the longest unique substring at each step.  

### Time Complexity (O(n))
- The right pointer scans the string once → **O(n)**.  
- The left pointer moves only when necessary, at most **O(n)** times.  
- Set operations (insert, remove, search) run in **O(1)** on average.  
- Overall time complexity: **O(n)**.  

### Space Complexity (O(n))
- The set stores unique characters → **O(n)** in the worst case.  
- The longest substring can also have up to **n** characters.  
- Overall space complexity: **O(n)**.

