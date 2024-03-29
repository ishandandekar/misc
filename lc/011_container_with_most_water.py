class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r, res, h = 0, len(height) - 1, 0, max(height)
        
        while l < r:
            res = max(res, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]:
                l += 1
            elif height[r] <= height[l]:
                r -= 1
            
            if (r-l) * h <= res:
                break 
        return res
