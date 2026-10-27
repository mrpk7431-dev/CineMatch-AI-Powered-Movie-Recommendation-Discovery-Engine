import time

class CacheService:
    def __init__(self, expiration_seconds=3600):
        self.cache = {}
        self.expiration_seconds = expiration_seconds

    def get(self, key):
        if key in self.cache:
            item = self.cache[key]
            if time.time() - item['timestamp'] < self.expiration_seconds:
                return item['data']
            else:
                del self.cache[key]
        return None

    def set(self, key, data):
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
