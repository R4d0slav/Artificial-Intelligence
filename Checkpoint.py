import pygame
from typing import Tuple

class Checkpoint:
    def __init__(self, x1:int, y1:int, x2:int, y2:int, color:Tuple[int, int, int] = (0, 0, 255)) -> None:
        self.color = color
        self.start = (x1, y1)
        self.end = (x2, y2)
        self.active = 1
    
    def reset(self) -> None:
        self.active = 1
        self.color = (0, 0, 255)

    def draw(self, screen:pygame.display) -> None:
        pygame.draw.line(screen, self.color, self.start, self.end, 3)

    def intersect(self, rect:pygame.Rect) -> bool:
        left = self._intersect(self.start, self.end, rect.topleft, rect.bottomleft)
        right = self._intersect(self.start, self.end, rect.topright, rect.bottomright)
        top = self._intersect(self.start, self.end, rect.topleft, rect.topright)
        bottom = self._intersect(self.start, self.end, rect.bottomleft, rect.bottomright)
        
        return left or right or top or bottom or rect.collidepoint(self.start) or rect.collidepoint(self.end)

    def _intersect(self, line1_start:Tuple[int, int], line1_end:Tuple[int, int], line2_start:Tuple[int, int], line2_end:Tuple[int, int]) -> bool:
        dir1 = (line1_end[0] - line1_start[0], line1_end[1] - line1_start[1])
        dir2 = (line2_end[0] - line2_start[0], line2_end[1] - line2_start[1])
        
        det = dir1[0] * dir2[1] - dir1[1] * dir2[0]

        if det == 0:
            return False

        d1 = ((line2_start[0] - line1_start[0]) * dir2[1] - (line2_start[1] - line1_start[1]) * dir2[0]) / det
        d2 = ((line2_start[0] - line1_start[0]) * dir1[1] - (line2_start[1] - line1_start[1]) * dir1[0]) / det
        
        return 0 <= d1 <= 1 and 0 <= d2 <= 1