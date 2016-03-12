import heapq


class PriorityQueue(object):

    def __init__(self):
        self._queue = []
        self._index = 0

    def __len__(self):
        return len(self._queue)

    def push(self, item, priority):
        setattr(item, '_heapindex', self._index)
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._update_index()
        self._index += 1

    def pop(self):
        el = heapq.heappop(self._queue)[-1]
        el._heapindex = -1
        self._update_index()
        return el

    def front(self):
        return self._queue[0][-1]

    def _update_index(self):
        if self._queue:
            for i, el in enumerate(zip(*self._queue)[2]):
                el._heapindex = i

