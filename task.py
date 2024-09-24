def maximum_product_of_three(nums):
    # Sort the array
    nums.sort()
    
    # Get the maximum product of the three largest numbers
    max1 = nums[-1] * nums[-2] * nums[-3]
    
    # Get the maximum product of two smallest and the largest number
    max2 = nums[0] * nums[1] * nums[-1]
    
    # Return the maximum of the two products
    return max(max1, max2)

# Example usage
if __name__ == "__main__":
    nums = [1, 10, 2, 6, 5, 3]
    print("The largest product of any three numbers is:", maximum_product_of_three(nums))