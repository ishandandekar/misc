from collections import Counter

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:

        counter_ = Counter(nums)
        counter_sorted_items_ = sorted(
            counter_.items(), key=lambda x: x[1], reverse=True
        )
        counter_keys_ = [item[0] for item in counter_sorted_items_]
        return counter_keys_[:k]
