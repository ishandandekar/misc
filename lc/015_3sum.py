class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res_ = list()

        for i, ele1 in enumerate(nums):
            for j, ele2 in enumerate(nums):
                for k, ele3 in enumerate(nums):
                    if i == j and j == k and i == k:
                        continue
                    if ele1 + ele2 + ele3 == 0:
                        res_.append(sorted([ele1, ele2, ele3]))

        res = {tuple(triplet) for triplet in res_}
        res = [list(triplet) for triplet in res]
        return res
