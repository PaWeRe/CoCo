# CoCo
LM mediator. Help LMs understand you.

## Quickstart
1. Create virtual env and .env file 
    * `/backend/.env` with at least `OPENAI_API_KEY` and `WANDB_API_KEY`
    * `pip install requirements.txt`
2. Start Redis DB - [Redis Quickstart](https://redis.io/learn/howtos/quick-start)
    * `docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest`
    * `docker exec -it redis-stack redis-cli`
    * `LRANGE request_queue 0 -1` and `LRANGE response_queue 0 -1` to check the status of the queues on Redis
3. `/backend`
    * `python backend.py` (execute from within /backend for .env to be found)
4. `/frontend` (execute from within /frontend for .env to be found, can changed easily)
    * `python frontend.py`

## For the Cursor Demo
1. Set up ngrok public address for Cursor to call our backend
    * `ngrok http http://localhost:8000`
2. Add it as `<ngrok-address>/v1` to the base model URL under OpenAI Cursor Models.
3. `@coco_delegate` works with in-line editing and works without the frontend
4. `@coco_collab` should be used in the chat (for time-out reasons) and calls the frontend