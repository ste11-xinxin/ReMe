import asyncio

from agentscope.message import Msg

from reme.reme_light import ReMeLight


async def main():
    reme = ReMeLight(
        default_as_llm_config={"model_name": "qwen3.5-plus"},
        default_file_store_config={
            "backend": "local",
            "fts_enabled": True,
            "vector_enabled": False,
        },
        enable_load_env=True,
    )
    await reme.start()

    messages = [
        Msg(name="user", role="user", content="我喜欢早上喝咖啡工作。"),
        Msg(
            name="assistant",
            role="assistant",
            content="好的，我记住了。你偏好在上午喝咖啡开始工作。",
        ),
    ]

    result = await reme.summary_memory(messages=messages, language="zh")
    print("summary_memory result:")
    print(result)

    search_result = await reme.memory_search(query="用户的工作习惯", max_results=5)
    print("memory_search result:")
    print(search_result)

    await reme.close()


if __name__ == "__main__":
    asyncio.run(main())
