# Largest Product of Three

## Problem Statement
Given an array of integers, this algorithm calculates the largest product that can be obtained by multiplying any three integers in the array.

## Approach
1. **Sorting**: The array is sorted in ascending order.
2. **Product Calculation**:
   - The product of the three largest numbers is calculated.
   - The product of the two smallest numbers and the largest number is calculated (in case of negative numbers).
3. The maximum of these two products is returned.

## Time Complexity
- The algorithm sorts the array, leading to a time complexity of \(O(n \log n)\).

## Space Complexity
- The space complexity is \(O(1)\) (not counting the space used for sorting).

## Usage
Run the script to see the largest product of three numbers in the example array.

```bash
python largest_product.py
