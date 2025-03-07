# CoCo
LM mediator. Help LMs understand you.

## Quickstart
1. Create virtual env
    * `pip install requirements.txt`
2. Start Redis DB - [Redis Quickstart](https://redis.io/learn/howtos/quick-start)
    * `docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest`
    * `docker exec -it redis-stack redis-cli`
3. `/backend`
    * `python backend.py`
4. `/frontend`
    * `python frontend.py`
