from functools import reduce
from operator import mul

def give_prod(lst) -> int:
    return reduce(mul, lst)

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:

        output = list()
        product = give_prod(nums)

        for idx, element in enumerate(nums):
            if element == 0:
                copied_lst = nums.copy()
                copied_lst.pop(idx)
                output.append(give_prod(copied_lst))

            else:
                output.append(int(product/element))

        return output
