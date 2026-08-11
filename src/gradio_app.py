import json
import os
from collections.abc import AsyncIterator

import gradio as gr
import httpx

API_BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000/api/v1")
GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7861"))


def format_answer(
    answer: str,
    sources: list[str],
    chunks_used: int,
) -> str:
    output = answer

    if chunks_used:
        output += f"\n\nRetrieved chunks: {chunks_used}"

    if sources:
        output += "\n\n### Sources"
        for source in sources:
            source_name = source.rsplit("/", maxsplit=1)[-1]
            output += f"\n- [{source_name}]({source})"

    return output


async def stream_response(
    query: str,
    top_k: int,
) -> AsyncIterator[str]:
    if not query.strip():
        yield "Please enter a question."
        return

    yield "Searching relevant research papers..."

    answer = ""
    sources: list[str] = []
    chunks_used = 0

    try:
        async with httpx.AsyncClient(timeout=240.0) as client:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/stream",
                json={"query": query, "top_k": top_k},
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    event = json.loads(line.removeprefix("data: "))

                    if "sources" in event:
                        sources = event["sources"]
                        chunks_used = event.get("chunks_used", 0)
                        if chunks_used:
                            yield "Relevant context found. Generating the answer..."

                    if "chunk" in event:
                        answer += event["chunk"]
                        yield format_answer(answer, sources, chunks_used)

                    if event.get("done"):
                        answer = event.get("answer", answer)
                        yield format_answer(answer, sources, chunks_used)
                        return

    except httpx.HTTPStatusError as error:
        yield f"The RAG API returned an error: {error.response.status_code}"
    except httpx.RequestError:
        yield f"Could not connect to the RAG API at {API_BASE_URL}."


def create_interface() -> gr.Blocks:
    with gr.Blocks(title="Agentic Research Assistant") as interface:
        gr.Markdown(
            """
            # Agentic Research Assistant

            Ask questions about the research papers indexed by the application.
            """
        )

        query_input = gr.Textbox(
            label="Question",
            placeholder="How do vision-language models understand 3D geometry?",
            lines=2,
        )
        top_k_input = gr.Slider(
            minimum=1,
            maximum=10,
            value=3,
            step=1,
            label="Chunks to retrieve",
        )
        submit_button = gr.Button("Ask", variant="primary")
        answer_output = gr.Markdown(value="Enter a question to begin.")

        submit_button.click(
            fn=stream_response,
            inputs=[query_input, top_k_input],
            outputs=answer_output,
            show_progress="full",
        )
        query_input.submit(
            fn=stream_response,
            inputs=[query_input, top_k_input],
            outputs=answer_output,
            show_progress="full",
        )

    return interface


def main() -> None:
    interface = create_interface()
    interface.queue().launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=False,
    )


if __name__ == "__main__":
    main()
