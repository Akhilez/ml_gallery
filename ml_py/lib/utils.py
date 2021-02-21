def fire_and_forget(f):
    def wrapped(*args, **kwargs):
        import asyncio

        return asyncio.get_event_loop().run_in_executor(None, f, *args, *kwargs)

    return wrapped
