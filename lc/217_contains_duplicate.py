class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        if sorted(list(set(nums))) != sorted(nums):
            return True
        else:
            return False
