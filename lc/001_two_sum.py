class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for idx, number in enumerate(nums):
            remainder = target - number
            if (remainder in nums) and (nums.index(remainder) != idx):
                return [idx, nums.index(remainder)]
