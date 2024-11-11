from collections import deque
from typing import Any

class RollingBufferMixin:
    """Mixin that provides rolling buffer functionality to any class with a buffer attribute."""
    
        
    def add_value(self, value: Any) -> None:
        """
        Add a new value to the buffer. If buffer is full, oldest value is automatically removed.
        
        Args:
            value: Value to add to the buffer
        """
        self.buffer.appendleft(value)
        
    def get_values(self) -> list:
        """
        Get all values in the buffer, newest to oldest.
        
        Returns:
            list: All values currently in the buffer
        """
        return list(self.buffer)
    
    def get_latest(self) -> Any:
        """
        Get the most recent value.
        
        Returns:
            The most recent value or None if buffer is empty
        """
        return self.buffer[0] if self.buffer else None
    
    def unique_count(self) -> int:
        """
        Get the number of unique values in the buffer.
        
        Returns:
            int: Number of unique values
        """
        return len(set(self.buffer))
    
    def get_unique_values(self) -> set:
        """
        Get all unique values in the buffer.
        
        Returns:
            set: Set of unique values
        """
        return set(self.buffer)
    
    def get_pace_list (self):
        s = list(self.get_unique_values())
        if len(s)>=2:
            pace_list = [round(s[i] - s[i + 1], 5 )  for i in range(len(s)-1) ]
        else :
            pace_list = None
        return pace_list
 